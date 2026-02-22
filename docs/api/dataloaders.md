# Datasets

## BaseASRDataset

Base class for all datasets. Provides pipeline integration, caching, filtering and metric computation.

```python
from plantain2asr.dataloaders.base import BaseASRDataset
```

**Key methods:**

| Method | Description |
|---|---|
| `filter(fn)` | Returns new dataset with samples matching predicate |
| `take(n)` | Returns first N samples |
| `to_pandas()` | Returns `pd.DataFrame` with one row per (sample, model) |
| `load_model_results(model_name, jsonl_path)` | Loads pre-computed inference results from JSONL file |
| `clone()` | Shallow clone of the dataset |

**Pipeline operator:**

```python
result = dataset >> processor   # applies model / normalizer / metric
```

---

## AudioSample

Data container for a single audio file.

```python
from plantain2asr.dataloaders.types import AudioSample
```

**Fields:**

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique identifier |
| `audio_path` | `str` | Absolute path to audio file |
| `text` | `str` | Reference transcript |
| `duration` | `float \| None` | Duration in seconds |
| `meta` | `dict` | Arbitrary metadata (e.g. `{"subset": "crowd"}`) |
| `asr_results` | `dict` | `{model_name: {"hypothesis": str, "metrics": dict, ...}}` |

---

## DagrusDataset

Loader for the DaGRuS (Dagestani Russian Speech) corpus.

```python
from plantain2asr import DagrusDataset

ds = DagrusDataset("data/dagrus")
```

**Constructor:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `root_dir` | `str` | required | Path to corpus root |
| `limit` | `int \| None` | `None` | Max number of samples to load |

---

## GolosDataset

Loader for the GOLOS test corpus. Auto-downloads on first run.

```python
from plantain2asr import GolosDataset

ds = GolosDataset("data/golos")
# filter by subset
crowd = ds.filter(lambda s: s.meta["subset"] == "crowd")
```

**Constructor:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `root_dir` | `str` | required | Path where corpus is stored |
| `limit` | `int \| None` | `None` | Max number of samples to load |
| `auto_download` | `bool` | `True` | Download if directory not found |

Each sample has `meta["subset"]` = `"crowd"` or `"farfield"`.

---

## NeMoDataset

Loader for datasets in NeMo JSONL manifest format.

```python
from plantain2asr import NeMoDataset

ds = NeMoDataset("data/my_corpus")
```

**Constructor:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `root_dir` | `str` | required | Directory containing `manifest.jsonl` |
| `limit` | `int \| None` | `None` | Max number of samples to load |
