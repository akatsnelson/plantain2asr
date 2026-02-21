import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import numpy as np
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from .base_trainer import BaseTrainer
from ..dataloaders.base import BaseASRDataset

class CTCCharTokenizer:
    """
    Minimal char-level tokenizer for CTC training/decoding based on model vocabulary.
    This bypasses non-standard GigaAM tokenizers that may not implement encode/decode.
    """
    def __init__(self, vocabulary: List[str], blank_id: Optional[int] = None):
        if not vocabulary:
            raise ValueError("CTCCharTokenizer: vocabulary is empty")
        self.vocab = vocabulary
        self.char2id = {ch: i for i, ch in enumerate(vocabulary)}
        # If model has an extra class for CTC blank, it's usually the last id
        self.blank_id = blank_id if blank_id is not None else len(vocabulary)

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        # basic cleanup: remove markers, normalize ё
        text = re.sub(r"#.*?#", " ", text)
        text = re.sub(r"\*.*?\*", " ", text)
        text = text.replace("ё", "е")
        # keep only letters/digits/spaces/hyphens; turn punctuation into spaces
        text = re.sub(r"[^\wа-яёa-z\d\s-]", " ", text, flags=re.UNICODE)
        text = text.replace("-", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def encode(self, text: str) -> List[int]:
        text = self.normalize_text(text)
        if not text:
            return []
        ids: List[int] = []
        for ch in text:
            # keep spaces and known chars only
            if ch in self.char2id:
                ids.append(self.char2id[ch])
            elif ch == " " and " " in self.char2id:
                ids.append(self.char2id[" "])
            # else: skip unknown symbols
        return ids

    def decode(self, ids: List[int]) -> str:
        out_chars: List[str] = []
        for i in ids:
            if i is None:
                continue
            if i == -100:
                continue
            if i == self.blank_id:
                continue
            if 0 <= i < len(self.vocab):
                out_chars.append(self.vocab[i])
        # collapse multiple spaces
        return re.sub(r"\s+", " ", "".join(out_chars)).strip()

class GigaAMDataset(Dataset):
    def __init__(self, dataset: BaseASRDataset, tokenizer: Any):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sample_rate = 16000

    def _load_audio_any(self, path: str) -> tuple[np.ndarray, int]:
        try:
            wav, sr = torchaudio.load(path)
            wav = wav.detach().cpu()
            if wav.ndim == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0).numpy()
            return wav.astype(np.float32, copy=False), int(sr)
        except Exception:
            import wave
            import contextlib
            try:
                with contextlib.closing(wave.open(path, "rb")) as wf:
                    sr = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    n_frames = wf.getnframes()
                    raw = wf.readframes(n_frames)

                if sampwidth == 2:
                    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    raise ValueError(f"Unsupported WAV width: {sampwidth}")

                if n_channels > 1:
                    x = x.reshape(-1, n_channels).mean(axis=1)
                return x, sr
            except Exception as e:
                raise RuntimeError(f"Failed to load audio '{path}': {e}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        try:
            waveform, sr = self._load_audio_any(sample.audio_path)
            if sr != self.sample_rate:
                wav_t = torch.from_numpy(waveform).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                wav_t = resampler(wav_t)
                waveform = wav_t.squeeze(0).numpy()
        except Exception as e:
            print(f"⚠️ Audio load error '{sample.audio_path}': {e}")
            return None 

        input_ids: List[int] = []
        if sample.text:
            try:
                # We expect tokenizer.encode -> List[int]
                if hasattr(self.tokenizer, "encode"):
                    input_ids = list(self.tokenizer.encode(sample.text))
                elif callable(self.tokenizer):
                    out = self.tokenizer(sample.text)
                    input_ids = list(getattr(out, "input_ids", out))
            except Exception as e:
                print(f"⚠️ Tokenization error for '{sample.id}': {e}")
                input_ids = []

        return {
            "input_values": torch.from_numpy(waveform),
            "labels": torch.tensor(input_ids, dtype=torch.long),
            "reference_text": sample.text  # Для декодинга референсов
        }

def gigaam_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    batch = [b for b in batch if b["input_values"].shape[0] > 0 and len(b["labels"]) > 0]
    if not batch: return None

    audio_list = [b["input_values"] for b in batch]
    input_values = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
    
    label_list = [b["labels"] for b in batch]
    labels = pad_sequence(label_list, batch_first=True, padding_value=-100)
    
    reference_texts = [b["reference_text"] for b in batch]

    return {
        "input_values": input_values, 
        "labels": labels,
        "reference_texts": reference_texts
    }

class GigaAMTrainer(BaseTrainer):
    def _extract_ctc_vocab(self) -> Tuple[List[str], Optional[int]]:
        """
        Extract vocabulary and blank id from GigaAM config if available.
        In GigaAM-v3-ctc checkpoints, vocab length is often num_classes-1 (blank is extra class).
        """
        vocab: Optional[List[str]] = None
        num_classes: Optional[int] = None

        def deep_get(obj: Any, path: List[str]):
            cur = obj
            for key in path:
                if cur is None:
                    return None
                if isinstance(cur, dict):
                    cur = cur.get(key)
                else:
                    cur = getattr(cur, key, None)
            return cur

        cfg = getattr(self.model_wrapper.model, "config", None)
        cfg_root = getattr(cfg, "cfg", None) if cfg is not None else None

        # try dict/attr paths
        vocab = (
            deep_get(cfg_root, ["model", "cfg", "decoding", "vocabulary"])
            or deep_get(cfg_root, ["cfg", "model", "cfg", "decoding", "vocabulary"])
        )
        num_classes = (
            deep_get(cfg_root, ["model", "cfg", "head", "num_classes"])
            or deep_get(cfg_root, ["cfg", "model", "cfg", "head", "num_classes"])
        )

        # fallback: try inner model cfg
        inner = getattr(self.model_wrapper, "_inner_model", None)
        vocab = vocab or deep_get(inner, ["cfg", "decoding", "vocabulary"]) or deep_get(inner, ["cfg", "cfg", "decoding", "vocabulary"])
        num_classes = num_classes or deep_get(inner, ["cfg", "head", "num_classes"]) or deep_get(inner, ["cfg", "cfg", "head", "num_classes"])

        if not vocab:
            # safe fallback: space + russian alphabet (no ё)
            vocab = [" "] + list("абвгдежзийклмнопрстуфхцчшщъыьэюя")

        blank_id = None
        if isinstance(num_classes, int) and num_classes == len(vocab) + 1:
            blank_id = len(vocab)
        return vocab, blank_id

    def prepare_training(self, train_dataset: BaseASRDataset):
        print("🔧 Preparing GigaAM training components...")
        
        model = self.model_wrapper.model
        tokenizer = None
        
        def find_tokenizer_recursive(obj, depth=0):
            if depth > 3: return None
            if hasattr(obj, 'tokenizer') and obj.tokenizer is not None: return obj.tokenizer
            if hasattr(obj, 'processor') and hasattr(obj.processor, 'tokenizer'): return obj.processor.tokenizer
            if hasattr(obj, "__dict__"):
                for key, val in obj.__dict__.items():
                    if isinstance(val, (torch.nn.Module, object)) and not isinstance(val, (str, int, float, list, dict)):
                        if key.startswith("_"): continue 
                        found = find_tokenizer_recursive(val, depth + 1)
                        if found: return found
            return None

        print("   🔍 Searching for tokenizer in model...")
        tokenizer = find_tokenizer_recursive(model)
        if tokenizer is None and hasattr(self.model_wrapper, "_inner_model"):
             tokenizer = find_tokenizer_recursive(self.model_wrapper._inner_model)

        if tokenizer is None:
            print("   ⚠️ Tokenizer not found in model object. Trying to load AutoProcessor from HF...")
            try:
                from transformers import AutoProcessor
                proc = AutoProcessor.from_pretrained("ai-sage/GigaAM-v3", trust_remote_code=True)
                tokenizer = proc.tokenizer
                print("   ✅ Loaded AutoProcessor from 'ai-sage/GigaAM-v3'")
            except Exception as e:
                print(f"   ❌ AutoProcessor load failed: {e}")

        if tokenizer is None:
            print("   ⚠️ Trying generic Russian Wav2Vec2 tokenizer...")
            try:
                from transformers import Wav2Vec2Processor
                proc = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
                tokenizer = proc.tokenizer
                print("   ✅ Generic tokenizer loaded.")
            except Exception as e:
                print(f"   ❌ Generic tokenizer load failed: {e}")

        if tokenizer is None:
            raise ValueError("❌ Could not find ANY tokenizer. Training impossible.")
        
        # Build robust char-level CTC tokenizer from model vocab (for training labels + decoding)
        vocab, blank_id = self._extract_ctc_vocab()
        self.ctc_tokenizer = CTCCharTokenizer(vocab, blank_id=blank_id)
        self.tokenizer = self.ctc_tokenizer  # use for training/eval in this trainer
        print(f"   ✅ Tokenizer ready: CTCCharTokenizer(vocab={len(vocab)}, blank_id={self.ctc_tokenizer.blank_id})")

        self.train_ds_torch = GigaAMDataset(train_dataset, self.tokenizer)
        self.train_loader = DataLoader(
            self.train_ds_torch, 
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=gigaam_collate_fn,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True
        )
        
        if self.eval_dataset:
            self.val_ds_torch = GigaAMDataset(self.eval_dataset, self.tokenizer)
            self.val_loader = DataLoader(
                self.val_ds_torch,
                batch_size=self.config.per_device_eval_batch_size,
                collate_fn=gigaam_collate_fn,
                num_workers=self.config.dataloader_num_workers
            )
            
        self.optimizer = torch.optim.AdamW(
            self.model_wrapper.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        inner = self.model_wrapper._inner_model if hasattr(self.model_wrapper, "_inner_model") else self.model_wrapper.model
        if hasattr(inner, "freeze_feature_encoder"):
            inner.freeze_feature_encoder()
            print("   ❄️ Feature encoder frozen.")

    def train_step(self, batch: Any) -> Dict[str, float]:
        if batch is None: return {'loss': 0.0}
        
        input_values = batch["input_values"].to(self.device)
        labels = batch["labels"].to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model_wrapper.model(input_values=input_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_wrapper.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {'loss': loss.item()}

    def eval_step(self, batch: Any) -> Dict[str, Any]:
        if batch is None:
            # Happens when collate_fn filtered out the whole batch (e.g., empty labels)
            return {'loss': 0.0}

        input_values = batch["input_values"].to(self.device)
        labels = batch["labels"].to(self.device)
        reference_texts = batch["reference_texts"]
        
        logits = None
        with torch.no_grad():
            outputs = self.model_wrapper.model(input_values=input_values, labels=labels)
            loss = outputs.loss.item()
            
            # Debug: проверим структуру outputs
            if not hasattr(self, "_debug_outputs_checked"):
                print(f"\n🔍 DEBUG: outputs type = {type(outputs)}")
                print(f"🔍 DEBUG: outputs keys/attrs = {dir(outputs) if hasattr(outputs, '__dir__') else 'N/A'}")
                self._debug_outputs_checked = True
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                if not hasattr(self, "_debug_logits_shape"):
                    print(f"🔍 DEBUG: logits.shape = {logits.shape}")
                    self._debug_logits_shape = True
            elif isinstance(outputs, tuple) and len(outputs) > 1:
                logits = outputs[1]
                if not hasattr(self, "_debug_logits_tuple"):
                    print(f"🔍 DEBUG: logits from tuple, shape = {logits.shape if hasattr(logits, 'shape') else 'N/A'}")
                    self._debug_logits_tuple = True

        preds = []
        refs = []
        
        # Decoding
        if logits is not None and logits.numel() > 0:
            pred_ids = torch.argmax(logits, dim=-1) # [batch, time]
            
            # Decode Predictions (CTC greedy)
            for i in range(len(pred_ids)):
                ids = pred_ids[i].tolist()
                # CTC collapse: remove consecutive duplicates and blank (0)
                collapsed = [x for j, x in enumerate(ids) if j == 0 or x != ids[j-1]]
                collapsed = [x for x in collapsed if x != 0] 
                
                try:
                    if hasattr(self.tokenizer, "decode"):
                        decoded = self.tokenizer.decode(collapsed)
                        preds.append(decoded.strip() if isinstance(decoded, str) else "")
                        
                        # Debug: print first prediction
                        if not hasattr(self, "_debug_first_pred"):
                            print(f"🔍 DEBUG: First prediction: ids={collapsed[:20]}... -> text='{decoded[:50]}'")
                            self._debug_first_pred = True
                    else:
                        preds.append("")
                except Exception as e:
                    if not hasattr(self, "_warned_decode_error"):
                        print(f"⚠️ Decode error: {e}")
                        self._warned_decode_error = True
                    preds.append("")
        else:
            if not hasattr(self, "_warned_no_logits"):
                print("\n⚠️ No logits found in outputs. Are you training an RNNT model with a CTC trainer?")
                self._warned_no_logits = True
            preds = [""] * len(labels)

        # References (from batch metadata, not from decoding labels)
        refs = reference_texts

        return {'loss': loss, 'preds': preds, 'refs': refs}

    def save_checkpoint(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model_wrapper.model.save_pretrained(path)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(path)
        print(f"💾 Checkpoint saved: {path}")
