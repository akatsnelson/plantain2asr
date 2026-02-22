# plantain2asr

[![PyPI version](https://img.shields.io/pypi/v/plantain2asr.svg)](https://pypi.org/project/plantain2asr/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue.svg)](https://akatsnelson.github.io/plantain2asr)

**Benchmarking and analysis framework for Russian ASR models.**

Pipeline API: load a dataset, apply models, normalize text, compute metrics, explore results — all in one consistent `>>` interface.

```python
from plantain2asr import GolosDataset, Models, DagrusNormalizer, Metrics, ReportServer

ds   = GolosDataset("data/golos")          # auto-downloads if missing
ds   >> Models.GigaAM_v3()                 # run inference (results cached)
norm = ds >> DagrusNormalizer()            # normalize text — creates a view
norm >> Metrics.composite()               # WER, CER, MER, WIL, WIP, Accuracy…
ReportServer(norm, audio_dir="data/golos").serve()  # interactive browser report
```

---

## Install

```bash
# Core — dataset loading + WER/CER metrics (no GPU required)
pip install plantain2asr

# + GigaAM v2/v3 and T-one models
pip install plantain2asr[gigaam]

# + Whisper (HuggingFace)
pip install plantain2asr[whisper]

# + deep analysis tools (pandas, bert-score, POS-analysis…)
pip install plantain2asr[analysis]

# Everything at once
pip install plantain2asr[all]
```

---

## Quick Start

### Load a dataset

```python
from plantain2asr import GolosDataset, DagrusDataset, NeMoDataset

# GOLOS test set — auto-downloads on first run (~2.5 GB)
ds = GolosDataset("data/golos")

# DaGRuS (Dagestani Russian Speech corpus)
ds = DagrusDataset("data/dagrus")

# Any NeMo-format JSONL manifest
ds = NeMoDataset("data/my_dataset")
```

### Apply a model

```python
from plantain2asr import Models

model = Models.GigaAM_v3()                        # GigaAM v3 e2e-RNNT (default)
ds >> model
```

Results accumulate in `sample.asr_results[model.name]`. Run multiple models on the same dataset to compare side by side.

Available models:

| Call | `model.name` stored | Extra | Device |
|---|---|---|---|
| `Models.GigaAM_v3()` | `GigaAM-v3-e2e_rnnt` | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v3(model_name="e2e_ctc")` | `GigaAM-v3-e2e_ctc` | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v3(model_name="rnnt")` | `GigaAM-v3-rnnt` | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v2(model_name="v2_rnnt")` | `GigaAM-v2_rnnt` | `gigaam` | CUDA / MPS / CPU |
| `Models.Whisper()` | `Whisper-whisper-large-v3-ru-podlodka` | `whisper` | CUDA / MPS / CPU |
| `Models.Tone()` | `T-One` | `gigaam` | CUDA |
| `Models.Vosk(model_path=…)` | `Vosk` | `vosk` | CPU |
| `Models.Canary()` | `Canary-1B` | `canary` | CUDA |
| `Models.SaluteSpeech()` | `SaluteSpeech` | — | cloud |

### Normalize text

```python
from plantain2asr import SimpleNormalizer, DagrusNormalizer

# General Russian: lowercase, strip punctuation, ё→е
norm = ds >> SimpleNormalizer()

# DaGRuS-specific: handles [laugh], fillers (ага, угу), colloquialisms
norm = ds >> DagrusNormalizer()
```

Normalization creates a new dataset **view** — `ds` stays untouched.

### Compute metrics

```python
from plantain2asr import Metrics

norm >> Metrics.composite()   # WER, CER, MER, WIL, WIP, Accuracy, IDR, LengthRatio
```

### Explore results

```python
# Pandas DataFrame — one row per (sample, model)
df = norm.to_pandas()
df.groupby("model")[["WER", "CER", "Accuracy"]].mean().sort_values("WER")

# Interactive browser report: metrics table + error frequency + diff view
from plantain2asr import ReportServer
ReportServer(norm, audio_dir="data/golos").serve()
```

### Load pre-computed results

Run inference on a GPU machine, transfer JSONL files, load here:

```python
ds.load_model_results("GigaAM-v3-e2e_rnnt", "results/GigaAM-v3-e2e_rnnt_results.jsonl")
```

JSONL format: one JSON object per line — `{"audio_path": "…", "hypothesis": "…", "time": 1.23}`

---

## Filter and slice

```python
short    = ds.filter(lambda s: s.duration < 5.0)
crowd    = ds.filter(lambda s: s.meta["subset"] == "crowd")
farfield = ds.filter(lambda s: s.meta["subset"] == "farfield")
top10    = ds.take(10)
```

---

## Extending

plantain2asr is built around four abstract base classes — subclass any of them.

### Custom normalizer

```python
from plantain2asr import BaseNormalizer

class MyNormalizer(BaseNormalizer):
    def normalize_ref(self, text: str) -> str:
        return text.lower().replace("ё", "е")

    def normalize_hyp(self, text: str) -> str:
        return text.lower().replace("ё", "е")

norm = ds >> MyNormalizer()
```

### Custom model

```python
from plantain2asr.models.base import BaseASRModel

class MyModel(BaseASRModel):
    @property
    def name(self) -> str:
        return "MyModel"

    def transcribe(self, audio_path: str) -> str:
        return "transcribed text"

ds >> MyModel()
```

### Custom metric

```python
from plantain2asr.metrics.base import BaseMetric

class SyllableErrorRate(BaseMetric):
    @property
    def name(self) -> str:
        return "SER"

    def calculate(self, reference: str, hypothesis: str) -> float:
        ref_syls = sum(1 for c in reference if c in "аеёиоуыэюя")
        hyp_syls = sum(1 for c in hypothesis if c in "аеёиоуыэюя")
        return abs(ref_syls - hyp_syls) / max(ref_syls, 1) * 100

norm >> SyllableErrorRate()
```

### Custom report section

```python
from plantain2asr import BaseSection, ReportServer

class LengthSection(BaseSection):
    @property
    def name(self) -> str:   return "length"
    @property
    def title(self) -> str:  return "Length Stats"
    @property
    def icon(self) -> str:   return "📏"

    def compute(self, dataset) -> dict:
        return {s.id: {"words": len(s.text.split())} for s in dataset}

    def js_function(self) -> str:
        return "function render_length() { /* your JS */ }"

ReportServer(norm, sections=[LengthSection()]).serve()
```

Full documentation: [akatsnelson.github.io/plantain2asr](https://akatsnelson.github.io/plantain2asr)

---

## License

MIT — Artem Katsnelson
