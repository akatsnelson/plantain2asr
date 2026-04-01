import inspect
from pathlib import Path
from typing import Optional, Union
from ..base import BaseASRModel
from ...utils.logging import get_logger

_TONE_INSTALL_HINT = (
    "Install one of the supported extras:\n"
    "  pip install plantain2asr[tone]      # standalone CPU install\n"
    "  pip install plantain2asr[tone-gpu]  # standalone NVIDIA GPU install\n"
    "  pip install plantain2asr[asr-cpu]   # shared CPU environment\n"
    "  pip install plantain2asr[asr-gpu]   # shared GPU environment"
)

logger = get_logger(__name__)


class ToneModel(BaseASRModel):
    """
    T-one — streaming CTC ASR для русского языка (телефония).

    Установка:
        pip install plantain2asr[tone]

    Использование:
        model = Models.Tone()
        ds >> model
    """

    def __init__(self, model_name: Optional[str] = None, device: str = "auto"):
        try:
            from tone import StreamingCTCPipeline, read_audio
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                f"T-one dependencies are not installed: {e}\n{_TONE_INSTALL_HINT}"
            )

        self.model_name = model_name
        self._name = "T-One"
        self._read_audio = read_audio

        available = ort.get_available_providers()
        providers, provider_label = self._select_providers(device, available)
        self.device = provider_label
        logger.info("T-One provider: %s", provider_label)

        patched_wrapper = None
        original_from_local = None
        try:
            try:
                import tone.onnx_wrapper as _ow

                original_from_local = _ow.StreamingCTCModel.from_local

                @classmethod
                def _from_local_with_providers(cls, model_path):
                    sess = ort.InferenceSession(model_path, providers=providers)
                    return cls(sess)

                _ow.StreamingCTCModel.from_local = _from_local_with_providers
                patched_wrapper = _ow
            except Exception:
                patched_wrapper = None

            if model_name and Path(model_name).exists():
                logger.info("Loading T-One from local path: %s", model_name)
                self.pipeline = StreamingCTCPipeline.from_local(model_name)
            else:
                source_label = model_name or "default Hugging Face weights"
                logger.info("Loading T-One from %s", source_label)
                self.pipeline = self._load_remote_pipeline(StreamingCTCPipeline, model_name)
        except Exception as e:
            raise RuntimeError(
                "Failed to load the T-One pipeline. "
                "Check the selected provider, installed extra, and model weights source.\n"
                f"Requested source: {model_name or 'default'}\n"
                f"Selected provider: {provider_label}\n"
                f"Original error: {e}"
            ) from e
        finally:
            if patched_wrapper is not None and original_from_local is not None:
                patched_wrapper.StreamingCTCModel.from_local = original_from_local

    @property
    def name(self) -> str:
        return self._name

    def training_not_supported_reason(self) -> str:
        return (
            "T-One is currently exposed as an inference-only backend in plantain2asr. "
            "Its ONNX streaming runtime does not map onto the shared online fine-tuning "
            "pipeline yet."
        )

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        # T-one API: read_audio → forward_offline → list[Phrase]
        audio = self._read_audio(str(audio_path))
        phrases = self.pipeline.forward_offline(audio)
        if not phrases:
            return ""
        if isinstance(phrases, list):
            return " ".join(p.text for p in phrases if hasattr(p, "text")).strip()
        if hasattr(phrases, "text"):
            return phrases.text
        return str(phrases)

    @staticmethod
    def _select_providers(device: str, available: list[str]) -> tuple[list[str], str]:
        if device == "auto":
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
            if "CoreMLExecutionProvider" in available:
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"], "mps"
            return ["CPUExecutionProvider"], "cpu"

        if device == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDA provider requested for T-One, but onnxruntime-gpu is unavailable. "
                    f"Available providers: {available or ['none']}"
                )
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"

        if device == "mps":
            if "CoreMLExecutionProvider" not in available:
                raise RuntimeError(
                    "CoreML provider requested for T-One, but it is unavailable. "
                    f"Available providers: {available or ['none']}"
                )
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"], "mps"

        if device == "cpu":
            return ["CPUExecutionProvider"], "cpu"

        raise ValueError("device must be one of: auto, cuda, mps, cpu")

    @staticmethod
    def _load_remote_pipeline(StreamingCTCPipeline, model_name: Optional[str]):
        if not model_name:
            return StreamingCTCPipeline.from_hugging_face()

        from_hf = StreamingCTCPipeline.from_hugging_face
        try:
            signature = inspect.signature(from_hf)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            params = signature.parameters
            if "repo_id" in params:
                return from_hf(repo_id=model_name)
            if "model_name" in params:
                return from_hf(model_name=model_name)
            if len(params) >= 1:
                return from_hf(model_name)

        return from_hf()
