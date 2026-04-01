# Quick Start

This page walks through a complete `>>` pipeline from loading data to viewing results.

## Install

```bash
pip install plantain2asr[asr-cpu]
```

For GPU workflows: `pip install plantain2asr[asr-gpu]`

## 1. Load a dataset

```python
from plantain2asr import GolosDataset

ds = GolosDataset("data/golos")
crowd = ds.filter(lambda s: s.meta["subset"] == "crowd")
```

## 2. Run models via `>>`

```python
from plantain2asr import Models

crowd >> Models.GigaAM_v3()
crowd >> Models.Whisper()
```

Results are cached on disk -- reruns skip already processed samples.

## 3. Normalize via `>>`

```python
from plantain2asr import SimpleNormalizer

norm = crowd >> SimpleNormalizer()
```

The original dataset is untouched; `norm` is a new view with normalized texts.

## 4. Compute metrics via `>>`

```python
from plantain2asr import Metrics

norm >> Metrics.composite()
```

This computes WER, CER, MER, WIL, WIP, Accuracy, IDR, and LengthRatio in a single pass.

## 5. Explore results

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

=== "Static HTML report"
    ```python
    from plantain2asr.reporting.builder import ReportBuilder
    ReportBuilder(norm).save_static_html("artifacts/report.html")
    ```

=== "CSV export"
    ```python
    norm.save_csv("artifacts/results.csv")
    ```

## Full pipeline in one block

```python
from plantain2asr import GolosDataset, Models, SimpleNormalizer, Metrics, ReportServer

ds = GolosDataset("data/golos")

ds >> Models.GigaAM_v3()
ds >> Models.Whisper()

norm = ds >> SimpleNormalizer()
norm >> Metrics.composite()

df = norm.to_pandas()
print(df.groupby("model")[["WER", "CER"]].mean().sort_values("WER"))

ReportServer(norm, audio_dir="data/golos").serve()
```

## Loading precomputed results

If inference was run on another machine, load JSONL results and continue evaluation locally:

```python
ds = GolosDataset("data/golos")
ds.load_model_results("GigaAM-v3-rnnt", "results/GigaAM-v3-rnnt_results.jsonl")
```

One line per sample:

```json
{"audio_path": "/any/path/file.wav", "hypothesis": "recognized text", "processing_time": 1.23}
```

Samples are matched by the basename of `audio_path`, which makes cross-machine reuse practical.

## `Experiment` convenience wrapper

If you want ready-made research scenarios without building the `>>` chain manually,
`Experiment` wraps the same pipeline steps:

```python
from plantain2asr import Experiment, GolosDataset, Models, SimpleNormalizer

experiment = Experiment(
    dataset=GolosDataset("data/golos"),
    models=[Models.GigaAM_v3(), Models.Whisper()],
    normalizer=SimpleNormalizer(),
)

experiment.compare_on_corpus(metrics=["WER", "CER", "Accuracy"])
```

| Method | What it does |
|---|---|
| `compare_on_corpus()` | Run models, normalize, score, return comparison table |
| `leaderboard()` | Rank models by a single metric |
| `prepare_thesis_tables()` | Export CSV tables for thesis/paper |
| `export_appendix_bundle()` | Full package: tables + report + benchmark |
| `benchmark_models()` | Latency, throughput, RTF measurements |
| `save_report_html()` | Static HTML report |

Under the hood, `Experiment` executes the same `>>` steps.
Use it when you want a one-liner; use the pipeline when you want control.
