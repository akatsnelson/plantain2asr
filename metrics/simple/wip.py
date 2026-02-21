from typing import List
from torchmetrics.text import WordInfoPreserved as TorchWIP
from ..base import BaseMetric

class WIP(BaseMetric):
    def __init__(self, do_clean: bool = False):
        super().__init__(do_clean)
        self.metric = TorchWIP()

    @property
    def name(self) -> str:
        return "WIP"

    def calculate(self, reference: str, hypothesis: str) -> float:
        reference = self.normalize(reference)
        hypothesis = self.normalize(hypothesis)

        if not reference.strip():
            return 0.0 if hypothesis.strip() else 100.0 # Preserved is 0 if empty ref but not empty hyp? 
            # Or usually 1.0 (100%) if both empty (perfect match).
            # Torchmetrics handles this, but let's stick to consistent logic:
            # If ref is empty and hyp is empty -> Perfect -> WIP=100
            # If ref is empty and hyp is NOT empty -> Insertion -> WIP=0
        
        score = self.metric([hypothesis], [reference])
        return float(score) * 100.0

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> float:
        if self.do_clean:
            references = [self.normalize(r) for r in references]
            hypotheses = [self.normalize(h) for h in hypotheses]

        score = self.metric(hypotheses, references)
        return float(score) * 100.0
