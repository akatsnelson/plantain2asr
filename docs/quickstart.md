# Quick Start

A complete example: from dataset loading to an interactive browser report.

!!! note "Prerequisites"
    ```bash
    pip install plantain2asr[gigaam,whisper]
    ```
    For the report server, no extra dependencies are needed.

---

## Step 1 — Load a dataset

```python
from plantain2asr import GolosDataset

# GOLOS test set — auto-downloads ~2.5 GB on first run
ds = GolosDataset("data/golos")
print(f"Loaded {len(ds)} samples")

# Filter to a subset
crowd    = ds.filter(lambda s: s.meta["subset"] == "crowd")
farfield = ds.filter(lambda s: s.meta["subset"] == "farfield")
```

!!! tip "DaGRuS corpus"
    ```python
    from plantain2asr import DagrusDataset
    ds = DagrusDataset("data/dagrus")
    ```

---

## Step 2 — Run inference

```python
from plantain2asr import Models

crowd >> Models.GigaAM_v3()   # stores results in sample.asr_results
crowd >> Models.Whisper()     # add more models for comparison
```

!!! info "Caching"
    Results are cached on disk — re-running the same model on the same dataset
    skips already-processed samples. Safe to interrupt and resume.

---

## Step 3 — Normalize

```python
from plantain2asr import SimpleNormalizer

# Creates a new dataset view — crowd is unchanged
norm = crowd >> SimpleNormalizer()
```

Normalization handles: lowercase, punctuation stripping, `ё → е` equalization.

!!! tip "DaGRuS corpus annotations"
    Use `DagrusNormalizer` to also strip `[laugh]`, `{word*}` and filler words:
    ```python
    from plantain2asr import DagrusNormalizer
    norm = ds >> DagrusNormalizer()
    ```

---

## Step 4 — Compute metrics

```python
from plantain2asr import Metrics

norm >> Metrics.composite()
# WER, CER, MER, WIL, WIP, Accuracy, IDR, LengthRatio — all in one fast pass
```

---

## Step 5 — Explore results

=== "Pandas"
    ```python
    df = norm.to_pandas()
    print(df.groupby("model")[["WER", "CER", "Accuracy"]].mean().sort_values("WER"))
    ```

=== "Interactive report"
    ```python
    from plantain2asr import ReportServer
    ReportServer(norm, audio_dir="data/golos").serve()
    ```
    Open **http://localhost:8765** — metrics table, error frequency with audio playback, word-level diff.

=== "Word errors"
    ```python
    from plantain2asr import WordErrorAnalyzer
    norm >> WordErrorAnalyzer(model_name="GigaAM-v3-e2e-rnnt", top_n=20)
    ```

---

## Loading pre-computed results

Run inference on a GPU machine, transfer JSONL, load here:

```python
ds = GolosDataset("data/golos")
ds.load_model_results("GigaAM-v3-rnnt", "results/GigaAM-v3-rnnt_results.jsonl")
```

JSONL format — one line per sample:
```json
{"audio_path": "/any/path/file.wav", "hypothesis": "распознанный текст", "processing_time": 1.23}
```

!!! warning "Match by filename"
    Samples are matched by the **basename** of `audio_path`, not the full path.
    This lets you use results computed on a different machine.
