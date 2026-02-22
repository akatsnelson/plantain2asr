from typing import List, Dict
from ..base import BaseMetric
from ..utils.alignment import align_words


class IDR(BaseMetric):
    """
    Insertion / Deletion / Substitution Rate.
    Возвращает словарь с разбивкой ошибок.
    """

    @property
    def name(self) -> str:
        return "IDR"

    def calculate(self, reference: str, hypothesis: str) -> Dict[str, float]:
        if self._normalizer is not None:
            reference = self.normalize(reference)
            hypothesis = self.normalize(hypothesis)

        ref_words = reference.split()
        hyp_words = hypothesis.split()

        if not ref_words and not hyp_words:
            return {"Insertion": 0.0, "Deletion": 0.0, "Substitution": 0.0}
        if not ref_words:
            return {"Insertion": 100.0, "Deletion": 0.0, "Substitution": 0.0}

        alignment = align_words(ref_words, hyp_words)
        counts = {"ins": 0, "del": 0, "sub": 0}
        for op, _, _ in alignment:
            if op in counts:
                counts[op] += 1

        total = len(ref_words)
        return {
            "Insertion":    (counts["ins"] / total) * 100.0,
            "Deletion":     (counts["del"] / total) * 100.0,
            "Substitution": (counts["sub"] / total) * 100.0,
        }

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        agg = {"ins": 0, "del": 0, "sub": 0}
        total = 0

        for ref, hyp in zip(references, hypotheses):
            if self._normalizer is not None:
                ref = self.normalize(ref)
                hyp = self.normalize(hyp)

            ref_w = ref.split()
            hyp_w = hyp.split()
            for op, _, _ in align_words(ref_w, hyp_w):
                if op in agg:
                    agg[op] += 1
            total += len(ref_w)

        if total == 0:
            return {"Insertion": 0.0, "Deletion": 0.0, "Substitution": 0.0}

        return {
            "Insertion":    (agg["ins"] / total) * 100.0,
            "Deletion":     (agg["del"] / total) * 100.0,
            "Substitution": (agg["sub"] / total) * 100.0,
        }
