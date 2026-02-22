from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

from ..core.processor import Processor

if TYPE_CHECKING:
    from ..normalization.base import BaseNormalizer
    from ..dataloaders.base import BaseASRDataset


class BaseMetric(Processor):
    """
    Абстрактный базовый класс для всех метрик.

    Нормализация текста выполняется на уровне датасета через пайплайн:
        dataset >> DagrusNormalizer() >> Metrics.composite()

    Если нужна нормализация на уровне метрики (редкий случай), передайте
    normalizer явно: WER(normalizer=SimpleNormalizer()).
    """

    def __init__(self, normalizer: Optional['BaseNormalizer'] = None):
        self._normalizer = normalizer

    @property
    def normalizer(self) -> Optional['BaseNormalizer']:
        return self._normalizer

    @property
    @abstractmethod
    def name(self) -> str:
        """Название метрики."""
        pass

    def normalize(self, text: str) -> str:
        """Нормализует текст через текущий нормализатор (или возвращает как есть)."""
        if self._normalizer is None:
            return text
        return self._normalizer.normalize_ref(text)

    @abstractmethod
    def calculate(self, reference: str, hypothesis: str) -> float:
        """Считает метрику для одной пары строк."""
        pass

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> float:
        """Считает усредненную метрику для батча."""
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses lists must have same length")
        total = sum(self.calculate(r, h) for r, h in zip(references, hypotheses))
        return total / len(references) if references else 0.0

    # ===== Processor API =====

    def apply_to(self, dataset: 'BaseASRDataset') -> 'BaseASRDataset':
        """Интеграция с pipeline >>. Считает метрику для всего датасета."""
        dataset._apply_metric(self)
        return dataset
