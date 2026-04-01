# 🌱 plantain2asr

[![PyPI version](https://img.shields.io/pypi/v/plantain2asr.svg)](https://pypi.org/project/plantain2asr/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue.svg)](https://akatsnelson.github.io/plantain2asr)

**Benchmarking and analysis framework for Russian ASR models.**

`plantain2asr` lets you compare ASR backends, normalize transcripts, compute metrics, inspect errors, benchmark latency, and export research artifacts without losing the underlying composable pipeline model.

## Start Here

There are three entry points, ordered from simplest to most flexible:

1. **Interactive Constructor**: open the docs constructor and assemble a ready-made chain visually.
2. **`Experiment` facade**: run common research scenarios with a few high-level calls.
3. **`>>` pipeline API**: build custom chains from datasets, models, normalizers, metrics, reports, and analyzers.

Docs: [akatsnelson.github.io/plantain2asr](https://akatsnelson.github.io/plantain2asr)

## Install

```bash
# Core only: datasets, normalization, metrics, reports
pip install plantain2asr

# Common CPU-only local stack
pip install plantain2asr[asr-cpu]

# Common GPU-ready local stack
pip install plantain2asr[asr-gpu]

# Individual model families
pip install plantain2asr[gigaam]
pip install plantain2asr[whisper]
pip install plantain2asr[vosk]
pip install plantain2asr[canary]
pip install plantain2asr[tone]

# Research analysis tools
pip install plantain2asr[analysis]

# Everything
pip install plantain2asr[all]
```

Device selection is automatic where supported: NVIDIA GPU first, then MPS, then CPU.

## Recommended Quick Start

For most research workflows, start with `Experiment`:

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

summary = experiment.compare_on_corpus(metrics=["WER", "CER", "Accuracy"])
leaderboard = experiment.leaderboard(metric="WER")

experiment.save_report_html("artifacts/report.html")
experiment.save_leaderboard_csv("artifacts/leaderboard.csv", metric="WER")

print(summary["metrics_by_model"])
print(leaderboard)
```

Use preset scenarios when you need ready-made research outputs:

- `Experiment.compare_on_corpus()` for straightforward model comparison
- `Experiment.prepare_thesis_tables()` for publication-ready aggregate tables
- `Experiment.export_appendix_bundle()` for a full appendix bundle with exports and optional static report
- `Experiment.benchmark_models()` for latency, throughput, and RTF measurement

## Advanced Pipeline API

If you want full composability, the canonical chain is still:

```python
from plantain2asr import GolosDataset, Models, DagrusNormalizer, Metrics, ReportServer

ds = GolosDataset("data/golos")
ds >> Models.GigaAM_v3()
ds >> Models.Whisper()

norm = ds >> DagrusNormalizer()
norm >> Metrics.composite()

ReportServer(norm, audio_dir="data/golos").serve()
```

Pipeline rules:

- Every step returns a dataset or processor-compatible object.
- Normalization creates a new dataset view and does not mutate the original.
- Model results are cached and safe to resume.
- You can branch at any point with `filter()`, `take()`, or cloned views.

## Supported Models

| Call | Stored name | Extra | Device |
|---|---|---|---|
| `Models.GigaAM_v3()` | `GigaAM-v3-e2e_rnnt` | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v3(model_name="e2e_ctc")` | `GigaAM-v3-e2e_ctc` | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v3(model_name="rnnt")` | `GigaAM-v3-rnnt` | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v3(model_name="ctc")` | `GigaAM-v3-ctc` | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v2(model_name="v2_rnnt")` | `GigaAM-v2_rnnt` | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v2(model_name="v2_ctc")` | `GigaAM-v2_ctc` | `gigaam` | CUDA / MPS / CPU |
| `Models.Whisper()` | `Whisper-whisper-large-v3-ru-podlodka` | `whisper` | CUDA / MPS / CPU |
| `Models.Tone()` | `T-One` | `tone` | CUDA / CPU |
| `Models.Vosk(model_path=...)` | `Vosk` | `vosk` | CPU |
| `Models.Canary()` | `Canary-1B` | `canary` | CUDA |
| `Models.SaluteSpeech()` | `SaluteSpeech` | none | cloud |

You can also resolve models by user-facing names with `Models.create(...)`, including case and separator variants such as `"gigaam_v3"`, `"GigaAM-v3"`, or `"tone"`.

## Typical Research Outputs

- Metrics tables as Python dicts or pandas DataFrames
- Leaderboards sorted by a chosen metric
- Error-case tables and CSV exports
- Static HTML reports for sharing without a running server
- Appendix bundles with thesis-ready artifacts
- Benchmark summaries for CPU, CUDA, or MPS

## Extending

plantain2asr keeps the "plantain" idea of modular composition. If the built-in stack is not enough, extend one of the four base types:

- `BaseASRModel`
- `BaseNormalizer`
- `BaseMetric`
- `BaseSection`

See the docs extending guides for custom components and implementation patterns.

## License

MIT — Artem Katsnelson
