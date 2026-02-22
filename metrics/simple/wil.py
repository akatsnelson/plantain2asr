from typing import List, Optional, TYPE_CHECKING
import jiwer
from ..base import BaseMetric

if TYPE_CHECKING:
    from ...normalization.base import BaseNormalizer


class WIL(BaseMetric):
    """
    Word Information Lost (jiwer).

    WIL = 1 - WIP = 1 - (H/N_ref) * (H/N_hyp)
    """

    @property
    def name(self) -> str:
        return "WIL"

    def calculate(self, reference: str, hypothesis: str) -> float:
        reference = self.normalize(reference)
        hypothesis = self.normalize(hypothesis)

        if not reference.strip():
            return 100.0 if hypothesis.strip() else 0.0

        return jiwer.wil(reference, hypothesis) * 100.0

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> float:
        if self._normalizer is not None:
            references = [self.normalize(r) for r in references]
            hypotheses = [self.normalize(h) for h in hypotheses]

        return jiwer.wil(references, hypotheses) * 100.0
