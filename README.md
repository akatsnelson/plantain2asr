# 🌱 plantain2asr

[![PyPI version](https://img.shields.io/pypi/v/plantain2asr.svg)](https://pypi.org/project/plantain2asr/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue.svg)](https://akatsnelson.github.io/plantain2asr)

**Benchmarking and analysis framework for Russian ASR models.**

Pipeline API that lets you load a dataset, apply models, normalize text, compute metrics and explore results — all in a consistent `>>` interface.

```python
from plantain2asr import GolosDataset, Models, SimpleNormalizer, Metrics, ReportServer

ds   = GolosDataset("data/golos")          # auto-downloads if missing
ds   >> Models.GigaAM_v3()                 # run inference
norm = ds >> SimpleNormalizer()            # normalize text
norm >> Metrics.composite()               # WER, CER, MER, WIL, WIP, Accuracy…
norm.to_pandas()                           # pandas DataFrame for further analysis
ReportServer(norm, audio_dir="data/golos").serve()  # interactive browser report
```

---

## Install

```bash
# Core — dataset loading + WER/CER metrics (no GPU required)
pip install plantain2asr

# + GigaAM v2/v3 models
pip install plantain2asr[gigaam]

# + Whisper (HuggingFace)
pip install plantain2asr[whisper]

# + deep analysis tools (pandas, bert-score, POS-analysis…)
pip install plantain2asr[analysis]

# Everything
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

ds >> Models.GigaAM_v3()                          # GigaAM v3 e2e-RNNT (default)
ds >> Models.GigaAM_v3(model_name="e2e_ctc")      # GigaAM v3 e2e-CTC
ds >> Models.GigaAM_v3(model_name="rnnt")         # GigaAM v3 RNNT
ds >> Models.GigaAM_v2(model_name="v2_rnnt")      # GigaAM v2
ds >> Models.Whisper()                             # Whisper large-v3 RU
ds >> Models.Tone()                                # T-one RussianTone
ds >> Models.Vosk(model_path="models/vosk-ru")    # Vosk (offline, CPU)
ds >> Models.SaluteSpeech()                        # SaluteSpeech API
```

Results accumulate in `sample.asr_results` — run multiple models on the same dataset to compare them.

### Normalize text

```python
from plantain2asr import SimpleNormalizer, DagrusNormalizer

# General Russian normalization: lowercase, strip punctuation, ё→е
norm = ds >> SimpleNormalizer()

# DaGRuS-specific: handles annotations [laugh], fillers (ага, угу), colloquialisms
norm = ds >> DagrusNormalizer(remove_fillers=False, strip_punctuation=True)
```

Normalization creates a new dataset **view** — the original `ds` is untouched.

### Compute metrics

```python
from plantain2asr import Metrics

norm >> Metrics.composite()   # WER, CER, MER, WIL, WIP, Accuracy, IDR, LengthRatio
```

Metrics are stored per-sample in `sample.asr_results[model]["metrics"]`.

### Explore results

```python
# Pandas DataFrame — one row per (sample, model)
df = norm.to_pandas()
df.groupby("model")[["WER", "CER", "Accuracy"]].mean().sort_values("WER")

# Word-level error breakdown
from plantain2asr import WordErrorAnalyzer
norm >> WordErrorAnalyzer(model_name="GigaAM-v3-e2e-rnnt", top_n=20)

# Interactive browser report: metrics table + error frequency + diff view
from plantain2asr import ReportServer
ReportServer(norm, audio_dir="data/golos").serve()
```

### Load pre-computed results

Run inference on a GPU machine, transfer JSONL files, load here:

```python
ds.load_model_results("GigaAM-v3-rnnt", "results/GigaAM-v3-rnnt_results.jsonl")
```

Format: `{"audio_path": "/any/path/file.wav", "hypothesis": "text", "processing_time": 1.23}`

---

## Filter and slice

```python
# Standard pipeline methods
short = ds.filter(lambda s: s.duration < 5.0)
crowd = ds.filter(lambda s: s.meta["subset"] == "crowd")
top10 = ds.take(10)
```

---

## Extending

plantain2asr is built around four abstract base classes. Subclass any of them to add your own components.

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
        # your inference logic
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
        # your metric logic
        ref_syls = sum(1 for c in reference if c in "аеёиоуыэюя")
        hyp_syls = sum(1 for c in hypothesis if c in "аеёиоуыэюя")
        return abs(ref_syls - hyp_syls) / max(ref_syls, 1) * 100

norm >> SyllableErrorRate()
```

### Custom report section

```python
from plantain2asr import BaseSection

class LengthSection(BaseSection):
    @property
    def name(self) -> str:   return "length"
    @property
    def title(self) -> str:  return "Length Stats"
    @property
    def icon(self) -> str:   return "📏"

    def compute(self, dataset) -> dict:
        return {
            s.id: {"words": len(s.text.split())}
            for s in dataset
        }

    def js_function(self) -> str:
        return "function render_length() { /* your JS */ }"

from plantain2asr import ReportServer
ReportServer(norm, sections=[LengthSection()]).serve()
```

See [full extending guide](https://plantain2asr.readthedocs.io/extending/) for complete examples.

---

## Supported models

| Model | Extra | Device |
|---|---|---|
| GigaAM v3 (e2e-rnnt, e2e-ctc, rnnt, ctc) | `gigaam` | CUDA / MPS / CPU |
| GigaAM v2 (v2-rnnt, v2-ctc) | `gigaam` | CUDA / MPS / CPU |
| Whisper large-v3 RU (HuggingFace) | `whisper` | CUDA / MPS / CPU |
| T-one RussianTone | `gigaam` | CUDA |
| Vosk | `vosk` | CPU |
| NVIDIA Canary | `canary` | CUDA |
| SaluteSpeech API | — | cloud |

---

## License

MIT
