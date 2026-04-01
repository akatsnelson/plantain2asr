import sys
import wave
import json
import subprocess
import tempfile
import os
import shutil
from pathlib import Path
from typing import Union

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
except ImportError:
    Model = None

from ..base import BaseASRModel
from ...utils.logging import get_logger

_VOSK_INSTALL_HINT = (
    "Install one of the supported extras:\n"
    "  pip install plantain2asr[vosk]\n"
    "  pip install plantain2asr[asr-cpu]\n"
    "  pip install plantain2asr[asr-gpu]"
)

logger = get_logger(__name__)

class VoskModel(BaseASRModel):
    def __init__(self, model_path: str = "models/vosk-model-ru-0.42"):
        if Model is None:
            raise ImportError(f"Vosk dependencies are not installed. {_VOSK_INSTALL_HINT}")
            
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Vosk model not found at {self.model_path}")
            
        logger.info("Loading Vosk model from %s", self.model_path)
        SetLogLevel(-1) # Mute logs
        self.model = Model(str(self.model_path))
        self._name = "Vosk"

    @property
    def name(self) -> str:
        return self._name

    def training_not_supported_reason(self) -> str:
        return (
            "Vosk is currently available only for inference in plantain2asr. "
            "The library does not yet provide a Kaldi/Vosk training backend behind the "
            "shared `dataset >> trainer` pipeline."
        )

    def _convert_audio(self, input_path: str) -> str:
        """Конвертирует в WAV 16kHz Mono (требование Vosk)"""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg is required to convert audio for Vosk when the input file is not "
                "16kHz mono WAV."
            )

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

    def _can_read_as_vosk_wav(self, audio_path: str) -> bool:
        try:
            with wave.open(audio_path, "rb") as wf:
                return (
                    wf.getframerate() == 16000
                    and wf.getnchannels() == 1
                    and wf.getsampwidth() == 2
                )
        except wave.Error:
            return False

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        audio_path = str(audio_path)
        temp_wav = None
        
        try:
            if self._can_read_as_vosk_wav(audio_path):
                source_path = audio_path
            else:
                temp_wav = self._convert_audio(audio_path)
                source_path = temp_wav

            with wave.open(source_path, "rb") as wf:
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
            logger.warning("Error transcribing %s: %s", audio_path, e)
            return ""
        finally:
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)
