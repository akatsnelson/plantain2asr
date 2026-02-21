import torch
from pathlib import Path
from typing import Union
from ..base import BaseASRModel

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa
except ImportError:
    WhisperProcessor = None

class WhisperModel(BaseASRModel):
    def __init__(self, model_name: str = "bond005/whisper-large-v3-ru-podlodka", device: str = "cuda"):
        if WhisperProcessor is None:
            raise ImportError("transformers or librosa not installed")
            
        # Smart device selection
        if device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        self._name = f"Whisper-{model_name.split('/')[-1]}"
        
        print(f"📥 Loading Whisper ({model_name}) on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        
        # Determine dtype
        dtype = torch.float32
        if self.device == "cuda" or self.device == "mps":
            dtype = torch.float16
            
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        # Загрузка и ресемплинг
        audio, _ = librosa.load(str(audio_path), sr=16000)
        
        # Токенизация
        input_features = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        if self.device == "cuda" or self.device == "mps":
            input_features = input_features.half()
            
        # Генерация
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, language="russian")
            
        # Декодирование
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
