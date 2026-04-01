import difflib
from typing import Optional, List
from .base import BaseASRModel

class Models:
    """
    Фабрика для удобного создания и листинга ASR моделей.
    Использует lazy imports (импорт внутри метода), чтобы не грузить Torch/CUDA
    при простом обращении к классу.
    """

    @staticmethod
    def _canonical_names() -> List[str]:
        return [
            "GigaAM_v3",
            "GigaAM_v2",
            "Whisper",
            "Vosk",
            "Canary",
            "Tone",
            "SaluteSpeech",
        ]

    @classmethod
    def _aliases(cls) -> dict:
        aliases = {}
        for canonical in cls._canonical_names():
            normalized = canonical.lower().replace("-", "").replace("_", "")
            aliases[canonical] = canonical
            aliases[canonical.lower()] = canonical
            aliases[normalized] = canonical
        aliases.update(
            {
                "gigaamv3": "GigaAM_v3",
                "gigaamv2": "GigaAM_v2",
                "gigaam-v3": "GigaAM_v3",
                "gigaam-v2": "GigaAM_v2",
                "salute": "SaluteSpeech",
                "salutespeech": "SaluteSpeech",
            }
        )
        return aliases

    @classmethod
    def _resolve_name(cls, name: str) -> str:
        aliases = cls._aliases()
        key = name.lower().replace("-", "").replace("_", "")
        if name in aliases:
            return aliases[name]
        if key in aliases:
            return aliases[key]

        candidates = cls._canonical_names()
        suggestions = difflib.get_close_matches(name, candidates, n=3, cutoff=0.45)
        suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(
            f"Unknown model '{name}'. Available: {', '.join(candidates)}.{suggestion_text}"
        )

    @staticmethod
    def list() -> List[str]:
        """Возвращает список доступных для создания моделей (названия методов)"""
        return Models._canonical_names()

    # === Local Models ===

    @staticmethod
    def GigaAM_v3(model_name: str = "e2e_rnnt", device: str = "auto") -> BaseASRModel:
        """
        SberDevices GigaAM v3 (CTC/RNNT).
        Options: 'e2e_rnnt' (default), 'e2e_ctc', 'rnnt', 'ctc', 'turbo'
        """
        from .local.gigaam_v3 import GigaAMv3
        return GigaAMv3(model_name=model_name, device=device)

    @staticmethod
    def GigaAM_v2(model_name: str = "v2_ctc", device: str = "auto") -> BaseASRModel:
        """
        SberDevices GigaAM v2.
        Options: 'v2_ctc', 'v2_rnnt'
        """
        from .local.gigaam_v2 import GigaAMv2
        return GigaAMv2(model_name=model_name, device=device)

    @staticmethod
    def Whisper(
        model_name: str = "bond005/whisper-large-v3-ru-podlodka",
        device: str = "auto",
        batch_size: int = 16,
    ) -> BaseASRModel:
        """
        OpenAI Whisper (HuggingFace implementation).
        """
        from .local.whisper_model import WhisperModel
        return WhisperModel(model_name=model_name, device=device, batch_size=batch_size)

    @staticmethod
    def Vosk(model_path: str = "models/vosk-model-ru-0.42") -> BaseASRModel:
        """
        AlphaCephei Vosk (Offline Kaldi-based).
        """
        from .local.vosk_model import VoskModel
        return VoskModel(model_path=model_path)
    
    @staticmethod
    def Canary(model_name: str = "nvidia/canary-1b", device: str = "cuda") -> BaseASRModel:
        """
        NVIDIA Canary (NeMo).
        """
        from .local.canary_model import CanaryModel
        return CanaryModel(model_name=model_name, device=device)
    
    @staticmethod
    def Tone(model_name: Optional[str] = None, device: str = "auto") -> BaseASRModel:
        """
        T-one RussianTone.

        model_name may be:
        - None: load the default Hugging Face weights
        - a local directory: load a local exported T-one pipeline
        - a Hugging Face repo id, if the installed tone package supports it
        """
        from .local.tone_model import ToneModel
        return ToneModel(model_name=model_name, device=device)

    # === Remote Models ===

    @staticmethod
    def SaluteSpeech(api_key: Optional[str] = None) -> BaseASRModel:
        """
        Sber SaluteSpeech API.
        Requires SALUTE_AUTH_DATA env var or api_key.
        """
        from .remote.salutespeech import SaluteSpeechModel
        return SaluteSpeechModel(auth_data=api_key)

    @staticmethod
    def create(name: str, **kwargs) -> BaseASRModel:
        """Универсальный метод создания по строковому имени"""
        canonical = Models._resolve_name(name)
        return getattr(Models, canonical)(**kwargs)
