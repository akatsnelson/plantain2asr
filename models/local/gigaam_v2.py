from typing import Union
from pathlib import Path
from ..base import BaseASRModel
from ...utils.logging import get_logger

try:
    import torch
except ImportError:
    torch = None

try:
    import gigaam
except ImportError:
    gigaam = None

_SUPPORTED_V2_MODELS = {"v2_ctc", "v2_rnnt"}
_GIGAAM_INSTALL_HINT = (
    "Install the backend with one of:\n"
    "  pip install plantain2asr[gigaam-v2]\n"
    "  pip install plantain2asr[gigaam]      # shared v2+v3 environment"
)

logger = get_logger(__name__)


def _resolve_torch_device(device: str) -> str:
    from ...utils.device import resolve_torch_device

    return resolve_torch_device(
        device, backend_name="GigaAM v2", install_hint=_GIGAAM_INSTALL_HINT,
    )


class GigaAMv2(BaseASRModel):
    """
    Обертка для GigaAM v2.
    Поддерживает варианты: 'v2_ctc', 'v2_rnnt'.
    """
    def __init__(self, model_name: str = "v2_ctc", device: str = "auto"):
        if gigaam is None:
            raise ImportError(f"Library 'gigaam' is not installed. {_GIGAAM_INSTALL_HINT}")
        if model_name not in _SUPPORTED_V2_MODELS:
            raise ValueError(
                f"Unsupported GigaAM v2 variant '{model_name}'. "
                f"Supported variants: {sorted(_SUPPORTED_V2_MODELS)}"
            )

        self.model_name = model_name
        self.device = _resolve_torch_device(device)

        self._name = f"GigaAM-{model_name}"
        
        logger.info("Loading %s on %s", self._name, self.device)

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

    def training_not_supported_reason(self) -> str:
        if self.model_name == "v2_ctc":
            return (
                f"{self.name} is inference-ready, but online fine-tuning is not wired into "
                "the current plantain2asr train pipeline yet. Use GigaAM v3 CTC for the "
                "supported training path in this release."
            )
        return (
            f"{self.name} uses an RNNT head, while the current plantain2asr train pipeline "
            "supports only the CTC training path."
        )

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        path = str(audio_path)
        try:
            result = self.model.transcribe(path)
        except Exception as e:
            if "Too long" in str(e) or "transcribe_longform" in str(e):
                result = self.model.transcribe_longform(path)
            else:
                raise
        if isinstance(result, list):
            return " ".join(result) if result else ""
        return str(result)
