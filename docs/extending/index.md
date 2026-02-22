# Extending plantain2asr

plantain2asr is built around **four abstract base classes**. To extend the library you subclass one of them — no other files need to be changed.

| What you want to add | Base class | Guide |
|---|---|---|
| Text normalization rules | `BaseNormalizer` | [Custom Normalizer](custom_normalizer.md) |
| A new ASR model | `BaseASRModel` | [Custom Model](custom_model.md) |
| A new quality metric | `BaseMetric` | [Custom Metric](custom_metric.md) |
| A new report tab | `BaseSection` | [Custom Report Section](custom_section.md) |

All four base classes live in `plantain2asr` and can be imported directly:

```python
from plantain2asr import BaseNormalizer, BaseSection
from plantain2asr.models.base import BaseASRModel
from plantain2asr.metrics.base import BaseMetric
```

Every component integrates into the `>>` pipeline automatically once the base class contract is fulfilled.
