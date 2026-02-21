from typing import List
import jiwer
from ..base import BaseMetric

class WER(BaseMetric):
    def __init__(self, do_clean: bool = True):
        """
        Word Error Rate metric using jiwer.
        
        Args:
            do_clean: If True (default), normalizes text (lowercase, remove punctuation).
        """
        super().__init__(do_clean)

    @property
    def name(self) -> str:
        return "WER"

    def calculate(self, reference: str, hypothesis: str) -> float:
        reference = self.normalize(reference)
        hypothesis = self.normalize(hypothesis)

        # Защита от пустых референсов
        if not reference.strip():
            return 100.0 if hypothesis.strip() else 0.0
        
        if not hypothesis.strip():
            return 100.0
            
        score = jiwer.wer(reference, hypothesis)
        return float(score) * 100.0

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> float:
        # Нормализуем списки, если включено
        if self.do_clean:
            references = [self.normalize(r) for r in references]
            hypotheses = [self.normalize(h) for h in hypotheses]

        # jiwer.wer поддерживает батчи напрямую
        score = jiwer.wer(references, hypotheses)
        return float(score) * 100.0
