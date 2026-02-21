# 🌿 Plantain2ASR (Подорожник для ASR)

A benchmarking, comparison, and analysis framework for ASR models.
"Applies" models to data, saves history, and heals your experiments.

> **Russian ASR focus** — built-in support for GigaAM, Whisper-RU, Vosk, T-one, SaluteSpeech and others.

---

## 🚀 Features

- **Unified data container (`AudioSample`)** — hypotheses, metrics, and timing live in one object.
- **Smart caching** — script crashed at 99 %? Restart and it resumes from where it stopped.
- **Adaptive batching** — GigaAM v3 and similar models saturate the GPU.
- **One-liner analytics** — export to Pandas in a single call.
- **Functional pipeline** — filter, sort, split, and process datasets with a fluent API.

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/plantain2asr.git
cd plantain2asr

# 2. Install the core package (+ choose optional extras)
pip install -e ".[gigaam,whisper,analysis]"

# Or install everything at once (heavy — includes all backends)
pip install -e ".[all]"
```

### Optional extras

| Extra | What it installs |
|-------|-----------------|
| `gigaam` | `transformers`, `accelerate` — for GigaAM v2/v3 |
| `whisper` | same as above — for Whisper (HuggingFace) |
| `vosk` | `vosk` — for offline Kaldi-based Vosk |
| `analysis` | `pandas`, `numpy`, `seaborn`, `scikit-learn`, `bert-score`, … |
| `train` | `datasets`, `wandb` — for fine-tuning |
| `all` | everything above |

---

## 🛠 Quick Start

### 1. Load a dataset and run models

Your dataset must follow the **NeMo JSONL manifest** format — one JSON object
per line with at minimum an `audio_filepath` key:

```jsonl
{"audio_filepath": "audio/sample_001.wav", "text": "привет мир", "duration": 2.1}
{"audio_filepath": "audio/sample_002.wav", "text": "как дела", "duration": 1.8}
```

```python
from plantain2asr.dataloaders import NeMoDataset
from plantain2asr.models import Models

# Load dataset (rootdir_path must contain manifest.jsonl)
dataset = NeMoDataset(rootdir_path="data/my_dataset", name="MyDataset")

# Create models (lazy-loaded — GPU/CPU is allocated on first use)
models = [
    Models.GigaAM_v3(device="cuda"),          # SberDevices GigaAM v3
    Models.Whisper(device="cuda"),             # Whisper large-v3 RU
    Models.Vosk(model_path="models/vosk-model-ru-0.42"),  # Offline Vosk
]

# Run with caching and metric computation
# Results are incrementally saved to cache/MyDataset/{ModelName}.jsonl
dataset.apply(models, batch_size=16, save_step=100, metrics_list=["wer", "cer"])

# Save final unified results
dataset.save_results("results/benchmark.jsonl")
```

### 2. Analyse results

```python
import pandas as pd
import seaborn as sns

df = dataset.to_pandas()

# Per-model average metrics
print(df.groupby("model")[["wer", "cer", "processing_time"]].mean())

# Visualise
sns.barplot(data=df, x="model", y="wer")
```

### 3. Recompute metrics after the fact

```python
# Ran models but forgot to compute metrics? No problem:
dataset.evaluate_results(metrics_list=["wer", "cer", "accuracy"], force=True)
```

### 4. Functional pipeline

```python
from plantain2asr.utils.functional import Filter, Sort, Split

# Fluent API
train_ds, test_ds = (
    dataset
    >> Filter(lambda s: s.duration < 15)   # drop long utterances
    >> Sort(lambda s: s.duration)           # sort by duration
    >> Split(ratio=0.8, by="duration")      # stratified 80/20 split
)

train_ds.apply(Models.GigaAM_v3())
```

### 5. SaluteSpeech cloud API

```python
import os
os.environ["SALUTE_AUTH_DATA"] = "<your_base64_key_from_sberdevices_studio>"

from plantain2asr.models import Models
model = Models.SaluteSpeech()  # reads key from env
```

---

## 🏗 Architecture

### `AudioSample`
Core data container.
- `id`, `audio_path`, `text`, `duration` — immutable input data.
- `asr_results` — `Dict[ModelName, Result]` with `{hypothesis, processing_time, error, metrics: {wer: …}}`.

### `Models` factory
Centralised access to all ASR models.
- `Models.list()` — list available models.
- `Models.GigaAM_v3(...)`, `Models.Whisper(...)`, `Models.Vosk(...)`, … — create an instance.
- `Models.create("GigaAM_v3", device="cpu")` — create by string name.

### `NeMoDataset` / `BaseASRDataset`
Smart wrapper around a list of `AudioSample` objects.
- `apply(models)` — main entry point, supports batching and caching.
- `filter()`, `sort()`, `take()`, `random_split()`, `stratified_split()` — functional transformations.
- `to_pandas()` — export to DataFrame.
- `save_results()` / `load_results()` — persist and restore full result history.

### Available Models

| Class | Backend | Notes |
|-------|---------|-------|
| `GigaAM_v3` | HuggingFace `ai-sage/GigaAM-v3` | e2e_rnnt / e2e_ctc / rnnt / ctc |
| `GigaAM_v2` | HuggingFace `ai-sage/GigaAM` | v2_ctc / v2_rnnt |
| `WhisperModel` | HuggingFace Transformers | any Whisper checkpoint |
| `VoskModel` | Vosk (Kaldi) | fully offline, CPU |
| `CanaryModel` | NVIDIA NeMo | requires `nemo_toolkit` |
| `ToneModel` | HuggingFace `T-one/russiantone-large` | |
| `SaluteSpeechModel` | Sber REST API | requires API key |

### Available Metrics

`wer`, `cer`, `mer`, `wil`, `wip`, `accuracy`, `length_ratio`, `idr`, `bert_score`, `pos_analysis`

---

## 📄 License

MIT
# plantain2asr
