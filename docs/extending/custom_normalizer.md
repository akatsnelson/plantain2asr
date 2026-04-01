# Custom Normalizer

Normalization transforms reference transcripts and model hypotheses before metric calculation.
A normalizer applied at the **dataset level** — `dataset >> MyNormalizer()` — creates a new dataset view;
the original is never mutated.

## Base class contract

```python
from plantain2asr import BaseNormalizer

class BaseNormalizer(ABC):
    def normalize_ref(self, text: str) -> str: ...   # abstract — must implement
    def normalize_hyp(self, text: str) -> str: ...   # optional, defaults to normalize_ref
```

`normalize_ref` handles reference transcripts (may contain corpus-specific annotations).
`normalize_hyp` handles model output (by default it calls `normalize_ref`).

## Minimal example

Strip punctuation and lowercase:

```python
import re
from plantain2asr import BaseNormalizer

class StripPunctNormalizer(BaseNormalizer):
    _RE = re.compile(r"[^\w\s]", re.UNICODE)

    def normalize_ref(self, text: str) -> str:
        text = text.lower()
        text = self._RE.sub("", text)
        text = text.replace("ё", "е")
        return " ".join(text.split())
```

Use it:

```python
from plantain2asr import GolosDataset, Metrics, Models

ds   = GolosDataset("data/golos")
ds   >> Models.GigaAM_v3()

norm = ds >> StripPunctNormalizer()   # creates a normalized view
norm >> Metrics.composite()
```

## Different handling for ref and hyp

Corpus annotations (`[laugh]`, `{unclear*}`) exist only in references — remove them in `normalize_ref`
but leave `normalize_hyp` unchanged:

```python
import re
from plantain2asr import BaseNormalizer

class AnnotatedCorpusNormalizer(BaseNormalizer):
    _ANNOT = re.compile(r"\[.*?\]|\{.*?\*\}")  # [laugh], {word*}

    def normalize_ref(self, text: str) -> str:
        text = self._ANNOT.sub("", text)
        return text.lower().strip()

    def normalize_hyp(self, text: str) -> str:
        return text.lower().strip()
```

## Pipeline composition

Multiple normalizers can be composed by chaining:

```python
norm1 = ds >> StripPunctNormalizer()
norm2 = norm1 >> AnnotatedCorpusNormalizer()   # applied on top
```

## Full production example: `DagrusNormalizer`

See the implementation in the repository source:
[plantain2asr/normalization/dagrus.py](https://github.com/akatsnelson/plantain2asr/blob/main/plantain2asr/normalization/dagrus.py)

It handles DaGRuS corpus annotations, fillers, colloquialisms, and `е`/`ё` equivalence.
