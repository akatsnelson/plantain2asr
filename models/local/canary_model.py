import io
import logging
import os
import torch
from contextlib import contextmanager
from pathlib import Path
from typing import Union, List
from ..base import BaseASRModel
from ...utils.logging import get_logger

try:
    from nemo.collections.asr.models import EncDecMultiTaskModel
except ImportError:
    EncDecMultiTaskModel = None

logger = get_logger(__name__)


@contextmanager
def _quiet_nemo():
    """Глушим весь вывод NeMo: logging + sys.stdout/stderr + OS-level fd."""
    # 1. Python logging — все nemo-логгеры на CRITICAL
    nemo_loggers = [
        logging.getLogger(n)
        for n in list(logging.root.manager.loggerDict)
        if "nemo" in n.lower()
    ]
    old_levels = [(lg, lg.level) for lg in nemo_loggers]
    for lg in nemo_loggers:
        lg.setLevel(logging.CRITICAL)

    try:
        from nemo.utils import logging as _nlog
        old_nlog = _nlog.logger.level
        _nlog.logger.setLevel(logging.CRITICAL)
        # Убираем хендлеры NeMo-логгера
        old_handlers = _nlog.logger.handlers[:]
        _nlog.logger.handlers.clear()
    except Exception:
        old_nlog = None
        old_handlers = []

    # 2. Перехватываем sys.stdout и sys.stderr (Python-уровень)
    import sys
    old_sout, old_serr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()

    # 3. OS-уровень: fd 1 и fd 2 → /dev/null
    old_fd1, old_fd2 = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    try:
        yield
    finally:
        # Восстанавливаем fd
        os.dup2(old_fd1, 1); os.close(old_fd1)
        os.dup2(old_fd2, 2); os.close(old_fd2)
        # Восстанавливаем sys streams
        sys.stdout, sys.stderr = old_sout, old_serr
        # Восстанавливаем logging
        for lg, lv in old_levels:
            lg.setLevel(lv)
        if old_nlog is not None:
            try:
                from nemo.utils import logging as _nlog
                _nlog.logger.setLevel(old_nlog)
                _nlog.logger.handlers[:] = old_handlers
            except Exception:
                pass


class CanaryModel(BaseASRModel):
    def __init__(self, model_name: str = "nvidia/canary-1b", device: str = "cuda"):
        if EncDecMultiTaskModel is None:
            raise ImportError("nemo_toolkit not installed")

        self.model_name = model_name
        if device == "cuda" and not torch.cuda.is_available():
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self._name = "Canary-1B"

        logger.info("Loading Canary-1B on %s", self.device)
        with _quiet_nemo():
            self.model = EncDecMultiTaskModel.from_pretrained(model_name=model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
        logger.info("Canary-1B loaded")

    @property
    def name(self) -> str:
        return self._name

    def training_not_supported_reason(self) -> str:
        return (
            f"{self.name} is not trainable through the shared plantain2asr trainer yet. "
            "Canary needs a dedicated NeMo multitask training backend rather than the "
            "current CTC-oriented online fine-tune path."
        )

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
