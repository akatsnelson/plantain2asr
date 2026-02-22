# plantain2asr

**Benchmarking and analysis framework for Russian ASR models.**

Load a dataset, apply models, normalize, compute metrics, explore — all through one consistent `>>` pipeline.

```python
from plantain2asr import GolosDataset, Models, SimpleNormalizer, Metrics, ReportServer

ds   = GolosDataset("data/golos")       # auto-downloads on first run
ds   >> Models.GigaAM_v3()             # inference (cached)
ds   >> Models.Whisper()               # add more models for comparison

norm = ds >> SimpleNormalizer()         # normalize (е/ё, punctuation, case)
norm >> Metrics.composite()            # WER, CER, MER, WIL, Accuracy…

ReportServer(norm, audio_dir="data/golos").serve()  # open browser at localhost:8765
```

---

## How it works

```mermaid
graph LR
    A[Dataset] -->|">> Model()"| B[Inference]
    B -->|">> Normalizer()"| C[Normalized View]
    C -->|">> Metrics()"| D[Scores per sample]
    D -->|"ReportServer"| E[Interactive Report]
```

Each step produces a new dataset — nothing is mutated in place.
You can branch, filter, and compare at any point.

---

## Install

=== "Core only (no GPU)"
    ```bash
    pip install plantain2asr
    ```
    Includes: dataset loading, WER/CER computation, report server.

=== "With GigaAM models"
    ```bash
    pip install plantain2asr[gigaam]
    ```

=== "With Whisper"
    ```bash
    pip install plantain2asr[whisper]
    ```

=== "With analysis tools"
    ```bash
    pip install plantain2asr[analysis]
    ```
    Includes: pandas, BERTScore, POS-analysis, bootstrap CI.

=== "Everything"
    ```bash
    pip install plantain2asr[all]
    ```

---

## Supported models

| Model | Extra | Device |
|---|---|---|
| GigaAM v3 (e2e-rnnt, e2e-ctc, rnnt, ctc) | `gigaam` | CUDA / MPS / CPU |
| GigaAM v2 (v2-rnnt, v2-ctc) | `gigaam` | CUDA / MPS / CPU |
| Whisper large-v3 (HuggingFace) | `whisper` | CUDA / MPS / CPU |
| T-one RussianTone | `gigaam` | CUDA |
| Vosk | `vosk` | CPU |
| NVIDIA Canary | `canary` | CUDA |
| SaluteSpeech API | — | cloud |

---

## Extending

plantain2asr is built on four abstract base classes.
Subclass any of them and it integrates into the pipeline automatically.

| Base class | What it adds |
|---|---|
| `BaseNormalizer` | Text normalization rules |
| `BaseASRModel` | A new ASR model |
| `BaseMetric` | A new quality metric |
| `BaseSection` | A new report tab |

→ [Extending guide](extending/index.md)

---

!!! tip "Next step"
    Follow the [Quick Start](quickstart.md) for a complete runnable example.
