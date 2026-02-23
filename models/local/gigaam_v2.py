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

        # PyTorch >= 2.6 требует явного разрешения omegaconf-типов в чекпоинте
        try:
            import torch.serialization as _ts
            from omegaconf import DictConfig, ListConfig
            _ts.add_safe_globals([DictConfig, ListConfig])
        except Exception:
            pass

        # cuDNN LSTM flatten_parameters несовместимо с CUDA 13.x.
        # Отключаем cuDNN на время загрузки — на inference не влияет критично.
        cudnn_was_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            self.model = gigaam.load_model(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
        finally:
            torch.backends.cudnn.enabled = cudnn_was_enabled

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        return self.transcribe_batch([audio_path])[0]

    def transcribe_batch(self, audio_paths: List[Union[str, Path]]) -> List[str]:
        # GigaAM поддерживает батчинг "из коробки" через transcribe
        paths = [str(p) for p in audio_paths]
        
        # v2 API может отличаться, проверим типичный вызов
        # Обычно: model.transcribe(paths) -> List[str]
        return self.model.transcribe(paths)
