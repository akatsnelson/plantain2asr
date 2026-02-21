from typing import List, Dict, Type
from .base import BaseMetric
from .simple.wer import WER
from .simple.cer import CER
from .simple.mer import MER
from .simple.wil import WIL
from .simple.wip import WIP
from .simple.accuracy import Accuracy
from .simple.idr import IDR
from .simple.length_ratio import LengthRatio

class CompositeMetric:
    """
    Агрегатор для расчета нескольких метрик одновременно.
    Позволяет получить полный отчет по качеству распознавания.
    """

    name: str = "Composite"

    def __init__(self, metrics: List[Type[BaseMetric]] = None, do_clean: bool = False):
        """
        Args:
            metrics: Список классов метрик для расчета.
                     Если None, используются все стандартные (WER, CER, MER, WIL, WIP, Acc, IDR, LR).
            do_clean: Флаг нормализации текста для всех метрик.
        """
        if metrics is None:
            metrics = [WER, CER, MER, WIL, WIP, Accuracy, IDR, LengthRatio]
            
        # Инициализируем экземпляры метрик
        self.metrics = [m(do_clean=do_clean) for m in metrics]
        self.do_clean = do_clean

    def calculate(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Считает все метрики для одной пары.
        Returns:
            Dict: { 'WER': 12.5, 'CER': 5.4, ... }
        """
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.calculate(reference, hypothesis)
        return results

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Считает все метрики для батча.
        """
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.calculate_batch(references, hypotheses)
        return results
