# Normalization

Normalization is applied at the **dataset level** — it creates a new dataset view,
the original is never mutated.

```python
norm = dataset >> MyNormalizer()
```

## BaseNormalizer

```python
from plantain2asr import BaseNormalizer
```

**Interface:**

```python
class BaseNormalizer(ABC):
    def normalize_ref(self, text: str) -> str: ...   # abstract
    def normalize_hyp(self, text: str) -> str: ...   # default: calls normalize_ref
    def normalize_pair(self, ref, hyp) -> tuple: ... # convenience
```

- `normalize_ref` handles reference transcripts (may contain corpus annotations)
- `normalize_hyp` handles model output. Override when ref/hyp need different treatment.

---

## SimpleNormalizer

General-purpose Russian normalizer. Suitable for most datasets.

```python
from plantain2asr import SimpleNormalizer

norm = dataset >> SimpleNormalizer()
```

**What it does:**
- Lowercase
- Strip punctuation
- `ё` → `е`
- Collapse whitespace

---

## DagrusNormalizer

Normalizer tailored for the DaGRuS corpus annotation format.

```python
from plantain2asr import DagrusNormalizer

norm = dataset >> DagrusNormalizer(remove_fillers=False, strip_punctuation=True)
```

**Constructor:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `remove_fillers` | `bool` | `False` | Remove filler words (ага, угу, мм…) |
| `strip_punctuation` | `bool` | `True` | Strip punctuation |

**What it does (in addition to SimpleNormalizer):**
- Removes corpus annotations: `[laugh]`, `[noise]`, `{word*}` (unclear pronunciation)
- Removes unmarked events: "говорит на другом языке" etc.
- Optionally removes filler words
- Colloquialism normalization: "щас" → "сейчас", "ваще" → "вообще" etc.
- `ё` → `е`
