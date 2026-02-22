# Quick Start

A complete example that runs the full pipeline from dataset loading to an interactive report.

## 1. Install

```bash
pip install plantain2asr[gigaam,analysis]
```

## 2. Load a dataset

```python
from plantain2asr import GolosDataset

# GOLOS test set (auto-downloads ~2.5 GB on first run)
ds = GolosDataset("data/golos")

# Work only with the crowd subset
crowd = ds.filter(lambda s: s.meta["subset"] == "crowd")
```

## 3. Run inference

```python
from plantain2asr import Models

crowd >> Models.GigaAM_v3()    # stores results in sample.asr_results
crowd >> Models.Whisper()      # add more models for comparison
```

Results are cached on disk — re-running the same model on the same dataset is instant.

## 4. Normalize

```python
from plantain2asr import SimpleNormalizer

norm = crowd >> SimpleNormalizer()   # new view; `crowd` is unchanged
```

## 5. Compute metrics

```python
from plantain2asr import Metrics

norm >> Metrics.composite()   # WER, CER, MER, WIL, WIP, Accuracy, IDR, LengthRatio
```

## 6. Explore

```python
# Pandas summary
df = norm.to_pandas()
print(df.groupby("model")[["WER", "CER", "Accuracy"]].mean().sort_values("WER"))

# Interactive browser report
from plantain2asr import ReportServer
ReportServer(norm, audio_dir="data/golos").serve()
```

Open http://localhost:8765 — metrics table, error frequency breakdown, and word-level diff with audio playback.

## 7. Load pre-computed results (optional)

If inference was run on a separate GPU machine:

```python
from plantain2asr import NeMoDataset

ds = NeMoDataset("data/my_corpus")
ds.load_model_results("GigaAM-v3-rnnt", "results/GigaAM-v3-rnnt_results.jsonl")
```

JSONL format: `{"audio_path": "/path/to/file.wav", "hypothesis": "text", "processing_time": 1.23}`
