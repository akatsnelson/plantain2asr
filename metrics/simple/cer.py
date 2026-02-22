from typing import List, Optional, TYPE_CHECKING
import jiwer
from ..base import BaseMetric

if TYPE_CHECKING:
    from ...normalization.base import BaseNormalizer


class CER(BaseMetric):
    """Character Error Rate (jiwer)."""

    def __init__(self, normalizer: Optional['BaseNormalizer'] = None):
        super().__init__(normalizer=normalizer)

    @property
    def name(self) -> str:
        return "CER"

    def calculate(self, reference: str, hypothesis: str) -> float:
        reference = self.normalize(reference)
        hypothesis = self.normalize(hypothesis)

        if not reference.strip():
            return 100.0 if hypothesis.strip() else 0.0
        if not hypothesis.strip():
            return 100.0

        return float(jiwer.cer(reference, hypothesis)) * 100.0

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> float:
        if self._normalizer is not None:
            references = [self.normalize(r) for r in references]
            hypotheses = [self.normalize(h) for h in hypotheses]

        return float(jiwer.cer(references, hypotheses)) * 100.0
