from typing import List, Optional, TYPE_CHECKING
import jiwer
from ..base import BaseMetric

if TYPE_CHECKING:
    from ...normalization.base import BaseNormalizer


class MER(BaseMetric):
    """
    Match Error Rate (jiwer).

    MER = (S + D + I) / (H + S + D + I)
        где H — совпадения, S — замены, D — пропуски, I — вставки.
    """

    @property
    def name(self) -> str:
        return "MER"

    def calculate(self, reference: str, hypothesis: str) -> float:
        reference = self.normalize(reference)
        hypothesis = self.normalize(hypothesis)

        if not reference.strip():
            return 100.0 if hypothesis.strip() else 0.0

        return jiwer.mer(reference, hypothesis) * 100.0

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> float:
        if self._normalizer is not None:
            references = [self.normalize(r) for r in references]
            hypotheses = [self.normalize(h) for h in hypotheses]

        return jiwer.mer(references, hypotheses) * 100.0
