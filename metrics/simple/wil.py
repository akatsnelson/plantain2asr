from typing import List
from torchmetrics.text import WordInfoLost as TorchWIL
from ..base import BaseMetric

class WIL(BaseMetric):
    def __init__(self, do_clean: bool = False):
        super().__init__(do_clean)
        self.metric = TorchWIL()

    @property
    def name(self) -> str:
        return "WIL"

    def calculate(self, reference: str, hypothesis: str) -> float:
        reference = self.normalize(reference)
        hypothesis = self.normalize(hypothesis)

        if not reference.strip():
            return 100.0 if hypothesis.strip() else 0.0
            
        score = self.metric([hypothesis], [reference])
        return float(score) * 100.0

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> float:
        if self.do_clean:
            references = [self.normalize(r) for r in references]
            hypotheses = [self.normalize(h) for h in hypotheses]

        score = self.metric(hypotheses, references)
        return float(score) * 100.0
