# Analysis

Analysis tools for deeper investigation of ASR errors.

## WordErrorAnalyzer

Word-level error breakdown: top substitutions, deletions, insertions.

```python
from plantain2asr import WordErrorAnalyzer

analyzer = WordErrorAnalyzer(model_name="GigaAM-v3-e2e-rnnt", top_n=20)
norm >> analyzer
```

**Constructor:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | required | Model to analyze |
| `top_n` | `int` | `20` | Number of top errors to show |

---

## DiffVisualizer

Generates an HTML diff report (static file). Prefer `ReportServer` for interactive use.

```python
from plantain2asr import DiffVisualizer

viz = DiffVisualizer(model_name="GigaAM-v3-e2e-rnnt", output="reports/diff.html")
norm >> viz
```

---

## Other analyzers

| Class | Description |
|---|---|
| `PerformanceAnalyzer` | RTF (Real Time Factor) and latency analysis |
| `BootstrapAnalyzer` | Bootstrap confidence intervals for WER |
| `AgreementAnalyzer` | Inter-model agreement (Cohen's kappa) |
| `TopicAnalyzer` | Error breakdown by topic/domain |
| `HallucinationAnalyzer` | Detects hallucinated words not in reference |
| `DurationAnalyzer` | Error vs audio duration correlation |
| `NgramErrorAnalyzer` | N-gram level error patterns |
| `CalibrationAnalyzer` | Confidence calibration analysis |

All require the `analysis` extra: `pip install plantain2asr[analysis]`
