# Custom Metric

A metric computes a quality score from a reference transcript and a model hypothesis.
Implement `BaseMetric` and it integrates into `dataset >> MyMetric()` automatically.

## Base class contract

```python
from plantain2asr.metrics.base import BaseMetric

class BaseMetric(ABC):
    @property
    def name(self) -> str: ...                                  # abstract

    def calculate(self, reference: str, hypothesis: str) -> float: ...   # abstract

    # Optional: efficient batch aggregate (default: mean over calculate())
    def calculate_batch(self, references, hypotheses) -> float: ...
```

## Minimal example: Syllable Error Rate

```python
from plantain2asr.metrics.base import BaseMetric

VOWELS = set("аеёиоуыэюя")

def syllables(text: str) -> int:
    return sum(1 for c in text.lower() if c in VOWELS)

class SyllableErrorRate(BaseMetric):
    """Absolute difference in syllable count, normalised to reference length."""

    @property
    def name(self) -> str:
        return "SER"

    def calculate(self, reference: str, hypothesis: str) -> float:
        ref_s = syllables(reference)
        hyp_s = syllables(hypothesis)
        if ref_s == 0:
            return 0.0
        return abs(ref_s - hyp_s) / ref_s * 100.0
```

Apply it:

```python
from plantain2asr import GolosDataset, SimpleNormalizer

ds   = GolosDataset("data/golos")
norm = ds >> SimpleNormalizer()
norm >> SyllableErrorRate()
```

## Efficient batch metric

Override `calculate_batch` when you have a vectorized implementation:

```python
import jiwer
from plantain2asr.metrics.base import BaseMetric

class CharBigramError(BaseMetric):
    @property
    def name(self) -> str:
        return "CBE"

    def calculate(self, reference: str, hypothesis: str) -> float:
        ref_bg  = set(zip(reference, reference[1:]))
        hyp_bg  = set(zip(hypothesis, hypothesis[1:]))
        if not ref_bg:
            return 0.0
        return len(ref_bg - hyp_bg) / len(ref_bg) * 100.0

    def calculate_batch(self, references, hypotheses) -> float:
        scores = [self.calculate(r, h) for r, h in zip(references, hypotheses)]
        return sum(scores) / len(scores) if scores else 0.0
```

## Using results

After `norm >> SyllableErrorRate()`, each sample has:

```python
sample.asr_results["GigaAM-v3-e2e-rnnt"]["metrics"]["SER"]   # float
```

Or via pandas:

```python
df = norm.to_pandas()
df.groupby("model")["SER"].mean()
```
