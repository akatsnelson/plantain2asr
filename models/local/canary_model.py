import logging
import torch
from pathlib import Path
from typing import Union, List
from ..base import BaseASRModel

try:
    from nemo.collections.asr.models import EncDecMultiTaskModel
except ImportError:
    EncDecMultiTaskModel = None


def _suppress_nemo_logs():
    for name in logging.root.manager.loggerDict:
        if "nemo" in name.lower():
            logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel(logging.ERROR)
    except Exception:
        pass


class CanaryModel(BaseASRModel):
    def __init__(self, model_name: str = "nvidia/canary-1b", device: str = "cuda"):
        if EncDecMultiTaskModel is None:
            raise ImportError("nemo_toolkit not installed")

        if device == "cuda" and not torch.cuda.is_available():
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self._name = "Canary-1B"

        _suppress_nemo_logs()
        print(f"📥 Loading Canary-1B on {self.device}...")
        self.model = EncDecMultiTaskModel.from_pretrained(model_name=model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Подавляем логи после загрузки (NeMo добавляет хендлеры в __init__)
        _suppress_nemo_logs()

    @property
    def name(self) -> str:
        return self._name

    def _do_transcribe(self, paths: List[str]) -> List[str]:
        _suppress_nemo_logs()
        results = self.model.transcribe(
            paths,
            batch_size=1,
            task="asr",
            source_lang="ru",
            target_lang="ru",
            pnc="no",
        )
        if not results:
            return [""] * len(paths)
        # NeMo может вернуть список строк или список Hypothesis
        out = []
        for r in results:
            if isinstance(r, str):
                out.append(r)
            elif hasattr(r, "text"):
                out.append(r.text)
            else:
                out.append(str(r))
        return out

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        res = self._do_transcribe([str(audio_path)])
        return res[0] if res else ""

    def transcribe_batch(self, audio_paths: List[Union[str, Path]]) -> List[str]:
        return self._do_transcribe([str(p) for p in audio_paths])
