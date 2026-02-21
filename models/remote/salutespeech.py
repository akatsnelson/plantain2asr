import os
from pathlib import Path
from typing import Union, Optional

from .salutespeech_api import SaluteSpeechAPI
from ..base import BaseASRModel


class SaluteSpeechModel(BaseASRModel):
    """
    Sber SaluteSpeech cloud ASR model.

    Requires a SaluteSpeech authorization key, which can be passed directly
    or via the ``SALUTE_AUTH_DATA`` environment variable.

    Example::

        import os
        os.environ["SALUTE_AUTH_DATA"] = "<your_base64_key>"

        from plantain2asr.models import Models
        model = Models.SaluteSpeech()
    """

    def __init__(self, auth_data: Optional[str] = None, model: str = "general"):
        """
        Args:
            auth_data: Base64-encoded authorization key from SberDevices Studio.
                       Falls back to the ``SALUTE_AUTH_DATA`` environment variable.
            model: Recognition model — ``general`` (≥8 kHz) or ``callcenter`` (8 kHz).
        """
        key = auth_data or os.environ.get("SALUTE_AUTH_DATA")
        if not key:
            raise ValueError(
                "SaluteSpeech requires an authorization key. "
                "Pass it as `auth_data` or set the SALUTE_AUTH_DATA environment variable."
            )
        self.client = SaluteSpeechAPI(key)
        self.model_type = model
        self._name = f"SaluteSpeech-{model}"

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        return self.client.transcribe_async(str(audio_path), model=self.model_type, verbose=False)
