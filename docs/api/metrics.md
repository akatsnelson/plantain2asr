# Metrics

## Metrics factory

```python
from plantain2asr import Metrics
```

| Factory method | Description |
|---|---|
| `Metrics.composite()` | Computes all metrics below in one pass (fast batch mode) |
| `Metrics.WER()` | Word Error Rate |
| `Metrics.CER()` | Character Error Rate |
| `Metrics.MER()` | Match Error Rate |
| `Metrics.WIL()` | Word Information Lost |
| `Metrics.WIP()` | Word Information Preserved |
| `Metrics.Accuracy()` | 1 − MER |
| `Metrics.IDR()` | Insertion / Deletion Ratio |
| `Metrics.LengthRatio()` | Hypothesis / Reference length ratio |
| `Metrics.BERTScore()` | Semantic similarity (requires `analysis` extra) |
| `Metrics.POSAnalysis()` | Error breakdown by POS tag (requires `analysis` extra) |

**Usage:**

```python
norm >> Metrics.composite()       # recommended: all metrics, one batch jiwer call
norm >> Metrics.WER()             # single metric
```

Results are stored per-sample:

```python
sample.asr_results["GigaAM-v3-e2e-rnnt"]["metrics"]["WER"]   # float (0–100)
```

---

## BaseMetric

```python
from plantain2asr.metrics.base import BaseMetric
```

**Interface:**

```python
class BaseMetric(ABC):
    @property
    def name(self) -> str: ...

    def calculate(self, reference: str, hypothesis: str) -> float: ...
    def calculate_batch(self, references: list, hypotheses: list) -> float: ...
```

See [Custom Metric](../extending/custom_metric.md) for a full example.

---

## CompositeMetric

Computes WER, CER, MER, WIL, WIP, Accuracy, IDR, LengthRatio in a single
`jiwer.process_words` + `jiwer.process_characters` call per model.
~8× faster than computing metrics individually.

```python
from plantain2asr.metrics.composite import CompositeMetric

metric = CompositeMetric()
norm >> metric
```
