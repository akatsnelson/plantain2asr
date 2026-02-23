import torch
from typing import Union, List
from pathlib import Path
from ..base import BaseASRModel

try:
    import gigaam
except ImportError:
    gigaam = None

class GigaAMv2(BaseASRModel):
    """
    Обертка для GigaAM v2.
    Поддерживает варианты: 'v2_ctc', 'v2_rnnt'.
    """
    def __init__(self, model_name: str = "v2_ctc", device: str = "cuda"):
        if gigaam is None:
            raise ImportError("Library 'gigaam' not installed. pip install gigaam")
            
        # Smart device selection
        if device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self._name = f"GigaAM-{model_name}"
        
        print(f"⏳ Loading {self._name} on {self.device}...")

        # PyTorch >= 2.6: weights_only=True по умолчанию, но чекпоинт GigaAM v2
        # содержит omegaconf-объекты — патчим torch.load на время загрузки.
        _orig_load = torch.load

        def _load_unsafe(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        torch.load = _load_unsafe

        # cuDNN LSTM flatten_parameters несовместимо с CUDA 13.x.
        cudnn_was_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            self.model = gigaam.load_model(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
        finally:
            torch.load = _orig_load
            torch.backends.cudnn.enabled = cudnn_was_enabled

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        result = self.model.transcribe(str(audio_path))
        if isinstance(result, list):
            return result[0] if result else ""
        return str(result)
