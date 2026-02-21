import torch
from pathlib import Path
from typing import Union
from ..base import BaseASRModel

try:
    from nemo.collections.asr.models import EncDecMultiTaskModel
except ImportError:
    EncDecMultiTaskModel = None

class CanaryModel(BaseASRModel):
    def __init__(self, model_name: str = "nvidia/canary-1b", device: str = "cuda"):
        if EncDecMultiTaskModel is None:
            raise ImportError("nemo_toolkit not installed")
            
        # Smart device selection
        if device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        self._name = "Canary-1B"
        
        print(f"📥 Loading Canary-1B on {self.device}...")
        self.model = EncDecMultiTaskModel.from_pretrained(model_name=model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        # Canary поддерживает список файлов
        # transcribe() возвращает список текстов
        return self.model.transcribe([str(audio_path)])[0]

    def transcribe_batch(self, audio_paths: list) -> list:
        # Батчинг из коробки
        return self.model.transcribe([str(p) for p in audio_paths])
