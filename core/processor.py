"""
Processor — единый интерфейс для всех компонентов пайплайна.

Любой объект, реализующий apply_to(dataset) -> dataset, является Processor.
Это делает >> оператор универсальным: достаточно реализовать один метод.

Иерархия:
    Processor (ABC)
    ├── BaseASRModel    — транскрибирует аудио, записывает hypothesis
    ├── BaseMetric      — считает метрики, записывает в metrics{}
    ├── BaseNormalizer  — нормализует text и hypothesis в данных
    └── BaseAnalyzer    — анализирует данные, сохраняет self.report
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataloaders.base import BaseASRDataset


class Processor(ABC):
    """
    Базовый интерфейс для всех компонентов пайплайна plantain2asr.

    Контракт:
        - apply_to(dataset) получает датасет
        - apply_to(dataset) возвращает датасет (тот же или новый view)
        - Сайд-эффекты (файлы, отчёты) — внутри apply_to
        - Результаты хранятся в self.report (для Analyzer-классов)

    Это позволяет строить цепочки:
        dataset >> ModelA >> Metrics.composite() >> DiffVisualizer("out.html")
    """

    @abstractmethod
    def apply_to(self, dataset: 'BaseASRDataset') -> 'BaseASRDataset':
        """
        Применяет процессор к датасету.

        Args:
            dataset: Входной датасет.

        Returns:
            Датасет (тот же объект или нормализованный view).
            НИКОГДА не возвращает Report — для цепочки >> важно получить dataset.
        """
        pass
