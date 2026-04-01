# Metrics

Metrics are computed on normalized dataset views and stored per model, per sample.

## `Metrics` factory

```python
from plantain2asr import Metrics
```

| Factory method | Description |
|---|---|
| `Metrics.composite()` | Recommended multi-metric batch evaluation |
| `Metrics.WER()` | Word Error Rate |
| `Metrics.CER()` | Character Error Rate |
| `Metrics.MER()` | Match Error Rate |
| `Metrics.WIL()` | Word Information Lost |
| `Metrics.WIP()` | Word Information Preserved |
| `Metrics.Accuracy()` | `1 - MER` |
| `Metrics.IDR()` | Insertion / Deletion Ratio |
| `Metrics.LengthRatio()` | Hypothesis length divided by reference length |
| `Metrics.BERTScore()` | Semantic similarity, requires `analysis` |
| `Metrics.POSAnalysis()` | POS-tag error analysis, requires `analysis` |

Typical usage:

```python
norm >> Metrics.composite()
norm >> Metrics.WER()
metric = Metrics.get("cer")
```

Unknown metric names fail fast with helpful suggestions.

Stored shape:

```python
sample.asr_results["GigaAM-v3-e2e-rnnt"]["metrics"]["WER"]
```

## `BaseMetric`

```python
from plantain2asr.metrics.base import BaseMetric
```

```python
class BaseMetric(ABC):
    @property
    def name(self) -> str: ...

    def calculate(self, reference: str, hypothesis: str) -> float: ...
    def calculate_batch(self, references: list, hypotheses: list) -> float: ...
```

Use `calculate_batch` when a vectorized or aggregated implementation is cheaper than per-sample calls.

## `CompositeMetric`

`CompositeMetric` computes the core metrics in a single batch-oriented pass and is the default recommendation for almost every evaluation workflow.

```python
from plantain2asr.metrics.composite import CompositeMetric

norm >> CompositeMetric()
```

See [Custom Metric](../extending/custom_metric.md) for extension examples.
