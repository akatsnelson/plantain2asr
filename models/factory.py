from typing import Optional, List, Dict, Any
from .base import BaseASRModel

class Models:
    """
    Фабрика для удобного создания и листинга ASR моделей.
    Использует lazy imports (импорт внутри метода), чтобы не грузить Torch/CUDA
    при простом обращении к классу.
    """

    @staticmethod
    def list() -> List[str]:
        """Возвращает список доступных для создания моделей (названия методов)"""
        return [
            "GigaAM_v3",
            "GigaAM_v2", 
            "Whisper",
            "Vosk",
            "Canary",
            "Tone",
            "SaluteSpeech"
        ]

    # === Local Models ===

    @staticmethod
    def GigaAM_v3(model_name: str = "e2e_rnnt", device: str = "cuda") -> BaseASRModel:
        """
        SberDevices GigaAM v3 (CTC/RNNT).
        Options: 'e2e_rnnt' (default), 'e2e_ctc', 'rnnt', 'ctc', 'turbo'
        """
        from .local.gigaam_v3 import GigaAMv3
        return GigaAMv3(model_name=model_name, device=device)

    @staticmethod
    def GigaAM_v2(model_name: str = "v2_ctc", device: str = "cuda") -> BaseASRModel:
        """
        SberDevices GigaAM v2.
        Options: 'v2_ctc', 'v2_rnnt'
        """
        from .local.gigaam_v2 import GigaAMv2
        return GigaAMv2(model_name=model_name, device=device)

    @staticmethod
    def Whisper(model_name: str = "bond005/whisper-large-v3-ru-podlodka", device: str = "cuda") -> BaseASRModel:
        """
        OpenAI Whisper (HuggingFace implementation).
        """
        from .local.whisper_model import WhisperModel
        return WhisperModel(model_name=model_name, device=device)

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
    def Tone(model_name: str = "T-one/russiantone-large", device: str = "cuda") -> BaseASRModel:
        """
        T-one RussianTone.
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
        if hasattr(Models, name):
            return getattr(Models, name)(**kwargs)
        raise ValueError(f"Unknown model: {name}. Available: {Models.list()}")
