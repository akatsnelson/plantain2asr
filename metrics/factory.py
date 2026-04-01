import difflib
from typing import List, Optional, TYPE_CHECKING
from .base import BaseMetric
from .composite import CompositeMetric

from .simple.wer import WER
from .simple.cer import CER
from .simple.mer import MER
from .simple.wil import WIL
from .simple.wip import WIP
from .simple.accuracy import Accuracy
from .simple.idr import IDR
from .simple.length_ratio import LengthRatio
try:
    from .complex.pos_analysis import PosErrorAnalysis
    from .complex.bert_score import BERTScore
    _HAS_COMPLEX = True
except ImportError:
    _HAS_COMPLEX = False

if TYPE_CHECKING:
    from ..normalization.base import BaseNormalizer


class Metrics:
    """
    Фабрика метрик plantain2asr.

    Типичное использование:
        # Нормализуем данные ДО метрик — рекомендуемый способ:
        dataset >> DagrusNormalizer() >> Metrics.composite()

        # Нормализация внутри метрик (устаревший стиль):
        Metrics.composite(normalizer=SimpleNormalizer())
    """

    @classmethod
    def _unknown_metric_error(cls, name: str) -> ValueError:
        available = cls.list()
        suggestions = difflib.get_close_matches(name.lower(), available, n=3, cutoff=0.45)
        suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return ValueError(
            f"Unknown metric '{name}'. Available: {', '.join(available)}.{suggestion_text}"
        )

    _REGISTRY = {
        "wer":          WER,
        "cer":          CER,
        "mer":          MER,
        "wil":          WIL,
        "wip":          WIP,
        "accuracy":     Accuracy,
        "idr":          IDR,
        "length_ratio": LengthRatio,
        **({
            "pos_analysis": PosErrorAnalysis,
            "bert_score":   BERTScore,
        } if _HAS_COMPLEX else {}),
    }

    @classmethod
    def list(cls) -> List[str]:
        """Список доступных метрик."""
        return list(cls._REGISTRY.keys())

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseMetric:
        """Создаёт экземпляр одной метрики по имени."""
        key = name.lower()
        if key in cls._REGISTRY:
            return cls._REGISTRY[key](**kwargs)
        raise cls._unknown_metric_error(name)

    @classmethod
    def composite(
        cls,
        names: Optional[List[str]] = None,
        normalizer: Optional['BaseNormalizer'] = None,
    ) -> CompositeMetric:
        """
        Создаёт CompositeMetric из списка метрик.

        Args:
            names:      Список метрик по имени. None → стандартный набор
                        (wer, cer, mer, wil, wip, accuracy, idr, length_ratio).
            normalizer: Нормализатор, передаётся в каждую метрику.
                        Рекомендуется применять нормализатор на уровне датасета,
                        а не здесь.
        """
        if not names or "all" in names:
            names = ["wer", "cer", "mer", "wil", "wip", "accuracy", "idr", "length_ratio"]

        selected = []
        unknown = []
        for name in names:
            key = name.lower()
            if key in cls._REGISTRY:
                selected.append(cls._REGISTRY[key])
            else:
                unknown.append(name)

        if unknown:
            if len(unknown) == 1:
                raise cls._unknown_metric_error(unknown[0])
            raise ValueError(
                "Unknown metrics: "
                + ", ".join(unknown)
                + f". Available: {', '.join(cls.list())}."
            )

        return CompositeMetric(metrics=selected, normalizer=normalizer)

    # Псевдоним для обратной совместимости
    @classmethod
    def create_composite(cls, names=None, normalizer=None, **_ignored) -> CompositeMetric:
        """Псевдоним для composite(). Параметр do_clean удалён."""
        return cls.composite(names=names, normalizer=normalizer)

    # === Shortcuts ===

    @staticmethod
    def WER(**kwargs) -> WER:
        return WER(**kwargs)

    @staticmethod
    def CER(**kwargs) -> CER:
        return CER(**kwargs)

    @staticmethod
    def MER(**kwargs) -> MER:
        return MER(**kwargs)

    @staticmethod
    def Accuracy(**kwargs) -> Accuracy:
        return Accuracy(**kwargs)

    @staticmethod
    def BERTScore(model_name: str = "bert-base-multilingual-cased", **kwargs):
        """Семантическая метрика через BERT. Тяжёлая — загружает модель."""
        if not _HAS_COMPLEX:
            raise ImportError("BERTScore requires: pip install plantain2asr[analysis]")
        return BERTScore(model_name=model_name, **kwargs)  # noqa: F821
