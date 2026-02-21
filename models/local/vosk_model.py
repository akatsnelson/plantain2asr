import sys
import wave
import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Union

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
except ImportError:
    Model = None

from ..base import BaseASRModel

class VoskModel(BaseASRModel):
    def __init__(self, model_path: str = "models/vosk-model-ru-0.42"):
        if Model is None:
            raise ImportError("vosk not installed. pip install vosk")
            
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Vosk model not found at {self.model_path}")
            
        print(f"📥 Loading Vosk model from {self.model_path}...")
        SetLogLevel(-1) # Mute logs
        self.model = Model(str(self.model_path))
        self._name = "Vosk"

    @property
    def name(self) -> str:
        return self._name

    def _convert_audio(self, input_path: str) -> str:
        """Конвертирует в WAV 16kHz Mono (требование Vosk)"""
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-ar', '16000',
            '-ac', '1',
            '-f', 'wav',
            '-y',
            temp_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return temp_path

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        audio_path = str(audio_path)
        temp_wav = None
        
        try:
            # Vosk требует WAV 16kHz Mono. Если файл другой - конвертируем.
            # Для надежности конвертируем всегда (ffmpeg быстрый).
            temp_wav = self._convert_audio(audio_path)
            
            wf = wave.open(temp_wav, "rb")
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)
            
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    part = json.loads(rec.Result())
                    results.append(part.get("text", ""))
            
            final_part = json.loads(rec.FinalResult())
            results.append(final_part.get("text", ""))
            
            return " ".join([r for r in results if r])
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return ""
        finally:
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)
