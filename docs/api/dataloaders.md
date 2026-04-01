# Datasets

Datasets are the backbone of the library. They hold samples, model outputs, metrics, and export-ready views.

## BaseASRDataset

```python
from plantain2asr.dataloaders.base import BaseASRDataset
```

Main responsibilities:

- store `AudioSample` objects
- apply processors through `>>`
- cache model outputs
- expose export and tabular helpers
- guard against empty-dataset workflows

Most-used methods:

| Method | What it does |
|---|---|
| `filter(fn)` | Returns a filtered dataset view |
| `take(n)` | Returns the first `n` samples |
| `run_model(model)` | Runs a model directly without writing pipeline syntax |
| `evaluate_metric(metric)` | Computes one metric directly |
| `to_pandas()` | Returns one row per `(sample, model)` |
| `iter_results_rows()` | Iterates flattened result rows |
| `save_csv(path)` | Exports flattened rows to CSV |
| `save_excel(path)` | Exports flattened rows to XLSX |
| `summarize_by_model()` | Builds aggregate metrics by model |
| `load_model_results(name, path)` | Loads precomputed JSONL inference |

Pipeline form:

```python
dataset >> model
dataset >> normalizer
dataset >> metric
```

## AudioSample

```python
from plantain2asr.dataloaders.types import AudioSample
```

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique sample identifier |
| `audio_path` | `str` | Audio file path |
| `text` | `str` | Reference transcript |
| `duration` | `float \| None` | Duration in seconds |
| `meta` | `dict` | Arbitrary metadata |
| `asr_results` | `dict` | Per-model hypotheses and metrics |

## Built-in dataset loaders

### `GolosDataset`

```python
from plantain2asr import GolosDataset

ds = GolosDataset("data/golos")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `root_dir` | `str` | required | Storage directory |
| `limit` | `int \| None` | `None` | Optional sample cap |
| `auto_download` | `bool` | `True` | Download automatically if missing |

Typical metadata: `meta["subset"]` is `"crowd"` or `"farfield"`.

### `DagrusDataset`

```python
from plantain2asr import DagrusDataset

ds = DagrusDataset("data/dagrus")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `root_dir` | `str` | required | Corpus root |
| `limit` | `int \| None` | `None` | Optional sample cap |

### `NeMoDataset`

```python
from plantain2asr import NeMoDataset

ds = NeMoDataset("data/my_corpus")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `root_dir` | `str` | required | Directory with `manifest.jsonl` |
| `limit` | `int \| None` | `None` | Optional sample cap |

## When to use `Experiment` instead

If you want a ready-made research workflow, use `Experiment` on top of a dataset instead of orchestrating individual dataset calls by hand.
