from abc import ABC, abstractmethod
from typing import List
import re

class BaseMetric(ABC):
    """
    Абстрактный базовый класс для всех метрик.
    Поддерживает опциональную нормализацию текста.
    """
    
    def __init__(self, do_clean: bool = False):
        """
        Args:
            do_clean: Если True, применяет normalize() к текстам перед расчетом.
        """
        self.do_clean = do_clean

    @property
    @abstractmethod
    def name(self) -> str:
        """Название метрики"""
        pass

    def normalize(self, text: str) -> str:
        """
        Предварительная очистка текста.
        Срабатывает, только если self.do_clean = True.
        """
        if not text:
            return ""
            
        if not self.do_clean:
            return text

        # 1. Приведение к нижнему регистру (стандарт для WER)
        text = text.lower()

        # 2. Удаление технических маркеров
        # Удаляет #нрзб#, *говорит на другом языке* и т.д.
        text = re.sub(r'#.*?#', '', text)
        text = re.sub(r'\*.*?\*', '', text)
        
        # 3. Удаление всех знаков препинания и спецсимволов (оставляем только буквы и пробелы)
        # Оставляем кириллицу (а-яё), латиницу (a-z), цифры (\d) и пробелы (\s)
        text = re.sub(r'[^\wа-яёa-z\d\s]', '', text, flags=re.UNICODE)
        
        # 4. Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @abstractmethod
    def calculate(self, reference: str, hypothesis: str) -> float:
        """Считает метрику для одной пары строк."""
        pass

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> float:
        """Считает усредненную метрику для батча."""
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses lists must have same length")
            
        total_score = 0.0
        count = 0
        
        for ref, hyp in zip(references, hypotheses):
            total_score += self.calculate(ref, hyp)
            count += 1
            
        return total_score / count if count > 0 else 0.0
