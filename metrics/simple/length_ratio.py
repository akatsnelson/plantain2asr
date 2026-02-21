from typing import List
from ..base import BaseMetric

class LengthRatio(BaseMetric):
    """
    Oтношение длины гипотезы к длине референса (в словах).
    LR = len(hyp) / len(ref)
    LR < 1: Модель удаляет слова (Deletions)
    LR > 1: Модель вставляет слова (Insertions / Hallucinations)
    """
    
    @property
    def name(self) -> str:
        return "LengthRatio"

    def calculate(self, reference: str, hypothesis: str) -> float:
        if self.do_clean:
            reference = self.normalize(reference)
            hypothesis = self.normalize(hypothesis)
            
        len_ref = len(reference.split())
        len_hyp = len(hypothesis.split())
        
        if len_ref == 0:
            return 0.0 if len_hyp == 0 else float('inf') # Или какое-то большое число
            
        return len_hyp / len_ref

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> float:
        total_ref = 0
        total_hyp = 0
        
        for ref, hyp in zip(references, hypotheses):
            if self.do_clean:
                ref = self.normalize(ref)
                hyp = self.normalize(hyp)
                
            total_ref += len(ref.split())
            total_hyp += len(hyp.split())
            
        if total_ref == 0:
            return 0.0
            
        return total_hyp / total_ref
