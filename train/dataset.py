import torch
import torchaudio
import numpy as np
import wave
import contextlib
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict, Union, Any, Optional

from ..dataloaders.base import BaseASRDataset

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
        """
        Load audio as mono float32 numpy array + sample rate.

        Primary: torchaudio.load
        Fallback: pure-Python WAV reader (no torchcodec dependency).
        """
        # 1) torchaudio (fast path)
        try:
            wav, sr = torchaudio.load(path)
            # wav: [channels, time]
            wav = wav.detach().cpu()
            if wav.ndim == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0).numpy()
            return wav.astype(np.float32, copy=False), int(sr)
        except ImportError as e:
            # Newer torchaudio may require torchcodec. We'll fallback below.
            last_err = e
        except Exception as e:
            last_err = e

        # 2) Pure WAV reader fallback
        try:
            with contextlib.closing(wave.open(path, "rb")) as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            if sampwidth == 1:
                # 8-bit unsigned PCM
                x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                x = (x - 128.0) / 128.0
            elif sampwidth == 2:
                x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 3:
                # 24-bit PCM packed little-endian -> int32
                b = np.frombuffer(raw, dtype=np.uint8)
                b = b.reshape(-1, 3)
                x = (b[:, 0].astype(np.int32) |
                     (b[:, 1].astype(np.int32) << 8) |
                     (b[:, 2].astype(np.int32) << 16))
                # sign extension
                x = (x ^ 0x800000) - 0x800000
                x = x.astype(np.float32) / 8388608.0
            elif sampwidth == 4:
                # Most commonly 32-bit PCM int
                x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

            if n_channels > 1:
                x = x.reshape(-1, n_channels).mean(axis=1)

            return x.astype(np.float32, copy=False), int(sr)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio '{path}' via torchaudio and WAV fallback. Last torchaudio error: {last_err}. WAV error: {e}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        
        # 1. Загружаем аудио
        # torchaudio возвращает (waveform, sr)
        # waveform: [channels, time]
        try:
            speech_array, sr = self._load_audio_any(sample.audio_path)

            # Ресемплинг если нужно (используем torchaudio, если доступен)
            if sr != self.sampling_rate:
                try:
                    # torchaudio expects Tensor; our loader returns numpy
                    wav_t = torch.from_numpy(speech_array).unsqueeze(0)  # [1, time]
                    resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                    wav_t = resampler(wav_t)
                    speech_array = wav_t.squeeze(0).numpy()
                except Exception as e:
                    raise RuntimeError(
                        f"Resampling failed (sr={sr} -> {self.sampling_rate}) for '{sample.audio_path}'. "
                        f"Install torchcodec or ensure torchaudio resample works. Error: {e}"
                    )
            
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
            print(f"⚠️ Error processing sample {sample.id}: {e}")
            # Возвращаем пустой сэмпл или None, коллатор должен это обработать
            # Для простоты пока кидаем ошибку, в идеале надо фильтровать битые файлы ДО обучения
            raise e

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
