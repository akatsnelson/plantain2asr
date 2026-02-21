from typing import List, Dict
from ..base import BaseMetric
from ..utils.alignment import align_words

class IDR(BaseMetric):
    """
    Insertion, Deletion, Substitution Rate.
    Возвращает словарь с разбивкой ошибок.
    """
    
    @property
    def name(self) -> str:
        return "IDR"

    def calculate(self, reference: str, hypothesis: str) -> Dict[str, float]:
        if self.do_clean:
            reference = self.normalize(reference)
            hypothesis = self.normalize(hypothesis)
            
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Если оба пустые
        if not ref_words and not hyp_words:
            return {"Insertion": 0.0, "Deletion": 0.0, "Substitution": 0.0}
            
        # Если референс пустой (все слова - вставки)
        if not ref_words:
            return {"Insertion": 100.0, "Deletion": 0.0, "Substitution": 0.0}

        alignment = align_words(ref_words, hyp_words)
        
        counts = {"ins": 0, "del": 0, "sub": 0}
        for op, _, _ in alignment:
            if op in counts:
                counts[op] += 1
                
        total_words = len(ref_words)
        
        return {
            "Insertion": (counts["ins"] / total_words) * 100.0,
            "Deletion": (counts["del"] / total_words) * 100.0,
            "Substitution": (counts["sub"] / total_words) * 100.0
        }

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        agg_counts = {"ins": 0, "del": 0, "sub": 0}
        total_words = 0
        
        for ref, hyp in zip(references, hypotheses):
            if self.do_clean:
                ref = self.normalize(ref)
                hyp = self.normalize(hyp)
                
            ref_w = ref.split()
            hyp_w = hyp.split()
            
            alignment = align_words(ref_w, hyp_w)
            
            for op, _, _ in alignment:
                if op in agg_counts:
                    agg_counts[op] += 1
            
            total_words += len(ref_w)
            
        if total_words == 0:
            return {"Insertion": 0.0, "Deletion": 0.0, "Substitution": 0.0}
            
        return {
            "Insertion": (agg_counts["ins"] / total_words) * 100.0,
            "Deletion": (agg_counts["del"] / total_words) * 100.0,
            "Substitution": (agg_counts["sub"] / total_words) * 100.0
        }
