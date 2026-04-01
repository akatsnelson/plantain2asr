import warnings
from pathlib import Path
from typing import List, Union
from ..base import BaseASRModel
from ...utils.logging import get_logger

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa
except ImportError:
    WhisperProcessor = None
    WhisperForConditionalGeneration = None
    librosa = None

_WHISPER_INSTALL_HINT = (
    "Install one of the supported extras:\n"
    "  pip install plantain2asr[whisper]\n"
    "  pip install plantain2asr[asr-cpu]\n"
    "  pip install plantain2asr[asr-gpu]"
)

logger = get_logger(__name__)


def _resolve_torch_device(device: str) -> str:
    from ...utils.device import resolve_torch_device

    return resolve_torch_device(
        device, backend_name="Whisper", install_hint=_WHISPER_INSTALL_HINT,
    )


class WhisperModel(BaseASRModel):
    def __init__(
        self,
        model_name: str = "bond005/whisper-large-v3-ru-podlodka",
        device: str = "auto",
        batch_size: int = 16,
    ):
        if WhisperProcessor is None or WhisperForConditionalGeneration is None or librosa is None:
            raise ImportError(f"Whisper dependencies are not installed. {_WHISPER_INSTALL_HINT}")

        self.device = _resolve_torch_device(device)
        self.model_name = model_name
        self.batch_size = batch_size
        self._name = f"Whisper-{model_name.split('/')[-1]}"

        logger.info("Loading Whisper (%s) on %s", model_name, self.device)
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
        logger.info("Whisper dtype=%s batch_size=%s", dtype, batch_size)

    @property
    def name(self) -> str:
        return self._name

    def training_not_supported_reason(self) -> str:
        return (
            f"{self.name} does not support plantain-style fine-tuning yet. Whisper needs a "
            "separate seq2seq training backend, so the current train API intentionally fails "
            "fast instead of pretending that CTC training is available."
        )

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
