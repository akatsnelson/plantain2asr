from pathlib import Path
from typing import Union
from ..base import BaseASRModel


class ToneModel(BaseASRModel):
    """
    T-one — streaming CTC ASR для русского языка (телефония).

    Установка:
        pip install git+https://github.com/voicekit-team/T-one.git

    Использование:
        model = Models.Tone()
        ds >> model
    """

    def __init__(self, model_name: str = None, device: str = "cuda"):
        try:
            from tone import StreamingCTCPipeline, read_audio
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                f"T-one не установлен: {e}\n"
                "Установи: pip install git+https://github.com/voicekit-team/T-one.git"
            )

        self._name = "T-One"
        self._read_audio = read_audio

        # Выбираем ONNX-провайдер под доступное железо
        available = ort.get_available_providers()
        if device in ("cuda", "auto") and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("   ✅ T-One: CUDA")
        elif device in ("mps", "auto") and "CoreMLExecutionProvider" in available:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            print("   ✅ T-One: CoreML (MPS)")
        else:
            providers = ["CPUExecutionProvider"]
            print("   ⚠️  T-One: CPU")

        # Патчим from_local чтобы пробросить провайдеры
        try:
            import tone.onnx_wrapper as _ow
            _orig_from_local = _ow.StreamingCTCModel.from_local.__func__

            @classmethod
            def _from_local_with_providers(cls, model_path):
                sess = ort.InferenceSession(model_path, providers=providers)
                return cls(sess)

            _ow.StreamingCTCModel.from_local = _from_local_with_providers
        except Exception:
            pass  # если API изменился — работаем без патча

        if model_name and Path(model_name).exists():
            print(f"📥 Loading T-One from {model_name}...")
            self.pipeline = StreamingCTCPipeline.from_local(model_name)
        else:
            print("📥 Loading T-One from HuggingFace...")
            self.pipeline = StreamingCTCPipeline.from_hugging_face()

    @property
    def name(self) -> str:
        return self._name

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
