import torch
import torchaudio
import numpy as np
import wave
import contextlib
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import List, Dict, Union, Any, Optional

from ..dataloaders.base import BaseASRDataset
from ..utils.logging import get_logger

TARGET_SAMPLE_RATE = 16000
logger = get_logger(__name__)


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono float32 numpy waveform via torchaudio."""
    wav_t = torch.from_numpy(waveform).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(wav_t).squeeze(0).numpy()


def load_audio_any(path: str) -> tuple[np.ndarray, int]:
    """
    Load audio as mono float32 numpy array plus sample rate.

    Primary path uses `torchaudio.load`; fallback handles plain WAV files without
    requiring additional runtime codecs.
    """
    last_err = None

    try:
        wav, sr = torchaudio.load(path)
        wav = wav.detach().cpu()
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0).numpy()
        return wav.astype(np.float32, copy=False), int(sr)
    except Exception as exc:
        last_err = exc

    try:
        with contextlib.closing(wave.open(path, "rb")) as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 1:
            x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            x = (x - 128.0) / 128.0
        elif sampwidth == 2:
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            x = (
                b[:, 0].astype(np.int32)
                | (b[:, 1].astype(np.int32) << 8)
                | (b[:, 2].astype(np.int32) << 16)
            )
            x = (x ^ 0x800000) - 0x800000
            x = x.astype(np.float32) / 8388608.0
        elif sampwidth == 4:
            x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

        if n_channels > 1:
            x = x.reshape(-1, n_channels).mean(axis=1)

        return x.astype(np.float32, copy=False), int(sr)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load audio '{path}' via torchaudio and WAV fallback. "
            f"Last torchaudio error: {last_err}. WAV error: {exc}"
        ) from exc

class HFWrappedDataset(Dataset):
    """
    Адаптер, который превращает наш BaseASRDataset в Dataset для HuggingFace Trainer.
    Выполняет загрузку аудио и токенизацию "на лету".
    """
    def __init__(self, dataset: BaseASRDataset, processor: Any):
        self.dataset = dataset
        self.processor = processor
        self.sampling_rate = 16000 # Стандарт для большинства моделей (GigaAM, Whisper, Wav2Vec2)

    def _load_audio_any(self, path: str) -> tuple[np.ndarray, int]:
        return load_audio_any(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        
        # 1. Загружаем аудио
        # torchaudio возвращает (waveform, sr)
        # waveform: [channels, time]
        try:
            speech_array, sr = self._load_audio_any(sample.audio_path)

            if sr != self.sampling_rate:
                speech_array = _resample(speech_array, sr, self.sampling_rate)
            
            # 2. Процессинг аудио (Feature Extraction)
            # Возвращает BatchEncoding, берем input_values
            input_values = self.processor(
                speech_array, 
                sampling_rate=self.sampling_rate
            ).input_values[0]
            
            # 3. Процессинг текста (Tokenization)
            if sample.text:
                with self.processor.as_target_processor():
                    labels = self.processor(sample.text).input_ids
            else:
                labels = None

            return {
                "input_values": input_values,
                "labels": labels
            }
            
        except Exception as e:
            logger.warning("Error processing sample %s: %s", sample.id, e)
            # Возвращаем пустой сэмпл или None, коллатор должен это обработать
            # Для простоты пока кидаем ошибку, в идеале надо фильтровать битые файлы ДО обучения
            raise e


class CTCCharTokenizer:
    """
    Minimal char-level tokenizer for CTC training/decoding based on model vocabulary.
    """

    def __init__(self, vocabulary: List[str], blank_id: Optional[int] = None):
        import re

        if not vocabulary:
            raise ValueError("CTCCharTokenizer: vocabulary is empty")

        self._re = re
        self.vocab = vocabulary
        self.char2id = {ch: i for i, ch in enumerate(vocabulary)}
        self.blank_id = blank_id if blank_id is not None else len(vocabulary)

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = self._re.sub(r"#.*?#", " ", text)
        text = self._re.sub(r"\*.*?\*", " ", text)
        text = text.replace("ё", "е")
        text = self._re.sub(r"[^\wа-яёa-z\d\s-]", " ", text, flags=self._re.UNICODE)
        text = text.replace("-", " ")
        text = self._re.sub(r"\s+", " ", text).strip()
        return text

    def encode(self, text: str) -> List[int]:
        text = self.normalize_text(text)
        if not text:
            return []
        ids = []
        for ch in text:
            if ch in self.char2id:
                ids.append(self.char2id[ch])
            elif ch == " " and " " in self.char2id:
                ids.append(self.char2id[" "])
        return ids

    def decode(self, ids: List[int]) -> str:
        out_chars = []
        for idx in ids:
            if idx in (None, -100, self.blank_id):
                continue
            if 0 <= idx < len(self.vocab):
                out_chars.append(self.vocab[idx])
        return self._re.sub(r"\s+", " ", "".join(out_chars)).strip()


class CharCTCAudioDataset(Dataset):
    def __init__(
        self,
        dataset: BaseASRDataset,
        tokenizer: CTCCharTokenizer,
        sample_rate: int = 16000,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        try:
            waveform, sr = load_audio_any(sample.audio_path)
            if sr != self.sample_rate:
                waveform = _resample(waveform, sr, self.sample_rate)
        except Exception as exc:
            logger.warning("Audio load error for %s: %s", sample.audio_path, exc)
            return None

        labels = self.tokenizer.encode(sample.text or "")
        return {
            "input_values": torch.from_numpy(waveform),
            "labels": torch.tensor(labels, dtype=torch.long),
            "reference_text": sample.text or "",
        }


def ctc_audio_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    batch = [item for item in batch if item["input_values"].shape[0] > 0 and len(item["labels"]) > 0]
    if not batch:
        return None

    input_values = pad_sequence(
        [item["input_values"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    labels = pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-100,
    )
    return {
        "input_values": input_values,
        "labels": labels,
        "reference_texts": [item["reference_text"] for item in batch],
    }

@dataclass
class DataCollatorCTCWithPadding:
    """
    Стандартный коллатор для CTC моделей (Wav2Vec2, HuBERT, GigaAM-CTC).
    Паддит аудио и лейблы до максимальной длины в батче.
    """
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch
