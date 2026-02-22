from typing import List, Optional, TYPE_CHECKING
import jiwer
from ..base import BaseMetric

if TYPE_CHECKING:
    from ...normalization.base import BaseNormalizer


class WIP(BaseMetric):
    """
    Word Information Preserved (jiwer).

    WIP = (H/N_ref) * (H/N_hyp),  H — число совпадений.
    """

    @property
    def name(self) -> str:
        return "WIP"

    def calculate(self, reference: str, hypothesis: str) -> float:
        reference = self.normalize(reference)
        hypothesis = self.normalize(hypothesis)

        if not reference.strip():
            return 0.0 if hypothesis.strip() else 100.0

        return jiwer.wip(reference, hypothesis) * 100.0

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> float:
        if self._normalizer is not None:
            references = [self.normalize(r) for r in references]
            hypotheses = [self.normalize(h) for h in hypotheses]

        return jiwer.wip(references, hypotheses) * 100.0
