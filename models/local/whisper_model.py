import torch
import warnings
from pathlib import Path
from typing import Union, List
from ..base import BaseASRModel

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa
except ImportError:
    WhisperProcessor = None


class WhisperModel(BaseASRModel):
    def __init__(
        self,
        model_name: str = "bond005/whisper-large-v3-ru-podlodka",
        device: str = "cuda",
        batch_size: int = 16,
    ):
        if WhisperProcessor is None:
            raise ImportError("transformers or librosa not installed")

        if device == "cuda" and not torch.cuda.is_available():
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self._name = f"Whisper-{model_name.split('/')[-1]}"

        print(f"📥 Loading Whisper ({model_name}) on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(model_name)

        # bf16 быстрее fp16 на Ampere+ (RTX 30xx/40xx/50xx)
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.dtype = dtype
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        print(f"   dtype={dtype}, batch_size={batch_size}")

    @property
    def name(self) -> str:
        return self._name

    def _load_audio(self, path: Union[str, Path]) -> "np.ndarray":
        audio, _ = librosa.load(str(path), sr=16000)
        return audio

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        return self.transcribe_batch([audio_path])[0]

    def transcribe_batch(self, audio_paths: List[Union[str, Path]]) -> List[str]:
        import numpy as np

        audios = [self._load_audio(p) for p in audio_paths]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs = self.processor(
                audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )

        input_features = inputs.input_features.to(self.device, dtype=self.dtype)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                language="russian",
            )

        transcriptions = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return [t.strip() for t in transcriptions]
