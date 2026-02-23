import io
import logging
import os
import torch
from contextlib import contextmanager
from pathlib import Path
from typing import Union, List
from ..base import BaseASRModel

try:
    from nemo.collections.asr.models import EncDecMultiTaskModel
except ImportError:
    EncDecMultiTaskModel = None


@contextmanager
def _quiet_nemo():
    """Глушим весь вывод NeMo: и logging, и stderr-спам."""
    # Поднимаем уровень всех nemo-логгеров
    nemo_loggers = [
        logging.getLogger(n)
        for n in list(logging.root.manager.loggerDict)
        if "nemo" in n.lower()
    ]
    old_levels = [(lg, lg.level) for lg in nemo_loggers]
    for lg in nemo_loggers:
        lg.setLevel(logging.CRITICAL)

    # Глушим nemo.utils.logging напрямую
    try:
        from nemo.utils import logging as _nlog
        old_nlog = _nlog.logger.level
        _nlog.logger.setLevel(logging.CRITICAL)
    except Exception:
        old_nlog = None

    # Перехватываем stderr (NeMo пишет туда минуя logging)
    old_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)

    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        for lg, lv in old_levels:
            lg.setLevel(lv)
        if old_nlog is not None:
            try:
                from nemo.utils import logging as _nlog
                _nlog.logger.setLevel(old_nlog)
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

        print(f"📥 Loading Canary-1B on {self.device}...")
        with _quiet_nemo():
            self.model = EncDecMultiTaskModel.from_pretrained(model_name=model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
        print(f"✅ Canary-1B loaded")

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        with _quiet_nemo():
            results = self.model.transcribe(
                [str(audio_path)],
                batch_size=1,
                task="asr",
                source_lang="ru",
                target_lang="ru",
                pnc="no",
            )
        if not results:
            return ""
        r = results[0]
        if isinstance(r, str):
            return r
        return getattr(r, "text", str(r))
