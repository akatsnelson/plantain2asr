from typing import List, Optional, Union
from .base import BaseMetric
from .composite import CompositeMetric

# Import metric classes
from .simple.wer import WER
from .simple.cer import CER
from .simple.mer import MER
from .simple.wil import WIL
from .simple.wip import WIP
from .simple.accuracy import Accuracy
from .simple.idr import IDR
from .simple.length_ratio import LengthRatio
from .complex.pos_analysis import PosErrorAnalysis
from .complex.bert_score import BERTScore

class Metrics:
    """
    Фабрика для метрик.
    Позволяет создавать как отдельные метрики, так и композитные наборы.
    """
    
    _REGISTRY = {
        "wer": WER,
        "cer": CER,
        "mer": MER,
        "wil": WIL,
        "wip": WIP,
        "accuracy": Accuracy,
        "idr": IDR,
        "length_ratio": LengthRatio,
        "pos_analysis": PosErrorAnalysis,
        "bert_score": BERTScore
    }

    @classmethod
    def list(cls) -> List[str]:
        """Список доступных метрик"""
        return list(cls._REGISTRY.keys())

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseMetric:
        """Создает экземпляр одной метрики по имени"""
        key = name.lower()
        if key in cls._REGISTRY:
            return cls._REGISTRY[key](**kwargs)
        raise ValueError(f"Unknown metric: {name}")

    @classmethod
    def create_composite(cls, names: Optional[List[str]] = None, do_clean: bool = True) -> CompositeMetric:
        """
        Создает CompositeMetric из списка названий.
        Аналог старой функции create_metrics.
        """
        selected_metrics = []
        
        # Стандартный набор (без тяжелых метрик вроде BERTScore)
        if not names or "all" in names:
            names = ["wer", "cer", "mer", "wil", "wip", "accuracy", "idr", "length_ratio"]

        for name in names:
            key = name.lower()
            if key in cls._REGISTRY:
                # Передаем классы в CompositeMetric (он сам их инстанцирует)
                selected_metrics.append(cls._REGISTRY[key])
            else:
                print(f"⚠️ Warning: Metric '{name}' not found. Skipping.")

        return CompositeMetric(metrics=selected_metrics, do_clean=do_clean)

    # === Shortcuts ===
    
    @staticmethod
    def WER(**kwargs) -> WER: return WER(**kwargs)
    
    @staticmethod
    def CER(**kwargs) -> CER: return CER(**kwargs)
    
    @staticmethod
    def MER(**kwargs) -> MER: return MER(**kwargs)
    
    @staticmethod
    def Accuracy(**kwargs) -> Accuracy: return Accuracy(**kwargs)

    @staticmethod
    def BERTScore(model_name: str = "bert-base-multilingual-cased", **kwargs) -> BERTScore:
        """
        Semantic similarity metric using BERT embeddings.
        Heavy metric! Loads a model.
        """
        return BERTScore(model_name=model_name, **kwargs)
