# Quick Start

This page shows the recommended path for a new user:

1. choose a dataset,
2. run a few models,
3. normalize and score them,
4. export shareable artifacts.

If you want a visual builder first, open the [Interactive Constructor](constructor.html).

## Recommended install

```bash
pip install plantain2asr[asr-cpu]
```

For GPU-heavy local workflows:

```bash
pip install plantain2asr[asr-gpu]
```

Add extras only when needed:

```bash
pip install plantain2asr[analysis]
```

## Recommended workflow: `Experiment`

```python
from plantain2asr import Experiment, GolosDataset, Models, SimpleNormalizer

dataset = GolosDataset("data/golos")

experiment = Experiment(
    dataset=dataset,
    models=[
        Models.GigaAM_v3(),
        Models.Whisper(),
    ],
    normalizer=SimpleNormalizer(),
)
```

### Step 1: compare models on one corpus

```python
comparison = experiment.compare_on_corpus(metrics=["WER", "CER", "Accuracy"])
print(comparison["metrics_by_model"])
```

What you get:

- model inference with caching
- normalized evaluation view
- aggregate metric table by model

### Step 2: inspect the ranking

```python
leaderboard = experiment.leaderboard(metric="WER")
print(leaderboard)
```

### Step 3: export artifacts

```python
experiment.save_leaderboard_csv("artifacts/leaderboard.csv", metric="WER")
experiment.save_report_html("artifacts/report.html")
experiment.export_error_cases("artifacts/error_cases.csv", metric="WER")
```

### Step 4: generate thesis-ready outputs

```python
tables = experiment.prepare_thesis_tables(
    metrics=["WER", "CER", "Accuracy"],
    output_dir="artifacts/thesis",
)

bundle = experiment.export_appendix_bundle(
    output_dir="artifacts/appendix",
    include_report=True,
    include_benchmark=True,
)
```

Use these presets when they match your task:

- `compare_on_corpus()` for general comparison
- `prepare_thesis_tables()` for clean aggregate tables
- `export_appendix_bundle()` for a full deliverable package
- `benchmark_models()` for latency, throughput, and RTF

## Advanced workflow: direct pipeline

Use the pipeline API when you need explicit control over each stage.

### Load a dataset

```python
from plantain2asr import GolosDataset

ds = GolosDataset("data/golos")
crowd = ds.filter(lambda s: s.meta["subset"] == "crowd")
```

### Run inference

```python
from plantain2asr import Models

crowd >> Models.GigaAM_v3()
crowd >> Models.Whisper()
```

Results are cached on disk, so reruns skip already processed samples.

### Normalize and score

```python
from plantain2asr import SimpleNormalizer, Metrics

norm = crowd >> SimpleNormalizer()
norm >> Metrics.composite()
```

### Explore results

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

## Loading precomputed results

If inference was run on another machine, you can load JSONL results and continue evaluation locally:

```python
ds = GolosDataset("data/golos")
ds.load_model_results("GigaAM-v3-rnnt", "results/GigaAM-v3-rnnt_results.jsonl")
```

One line per sample:

```json
{"audio_path": "/any/path/file.wav", "hypothesis": "recognized text", "processing_time": 1.23}
```

Samples are matched by the basename of `audio_path`, which makes cross-machine reuse practical.
