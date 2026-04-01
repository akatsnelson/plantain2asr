# Analysis

Analysis tools help you move from headline metrics to actual error structure.

## `WordErrorAnalyzer`

```python
from plantain2asr import WordErrorAnalyzer

norm >> WordErrorAnalyzer(model_name="GigaAM-v3-e2e-rnnt", top_n=20)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | required | Model to inspect |
| `top_n` | `int` | `20` | Number of top error patterns |

## `DiffVisualizer`

```python
from plantain2asr import DiffVisualizer

norm >> DiffVisualizer(model_name="GigaAM-v3-e2e-rnnt", output="reports/diff.html")
```

This produces a static diff-focused artifact. For broader interactive review, prefer `ReportServer` or `Experiment.save_report_html()`.

## Benchmarking

For runtime measurements, prefer the benchmark layer:

```python
benchmarks = experiment.benchmark_models()
```

This gives latency, throughput, and real-time-factor oriented summaries across supported devices.

## Other analyzers

All of the following require `plantain2asr[analysis]`:

| Class | Description |
|---|---|
| `PerformanceAnalyzer` | RTF and latency analysis |
| `BootstrapAnalyzer` | Bootstrap confidence intervals |
| `AgreementAnalyzer` | Inter-model agreement |
| `TopicAnalyzer` | Error by topic or domain |
| `HallucinationAnalyzer` | Hallucinated word detection |
| `DurationAnalyzer` | Error vs duration correlation |
| `NgramErrorAnalyzer` | N-gram error patterns |
| `CalibrationAnalyzer` | Confidence calibration |
