# plantain2asr

Benchmarking and analysis framework for Russian ASR models.

## What it does

- **Loads datasets** — DaGRuS, GOLOS, NeMo-format manifests.
- **Applies ASR models** — GigaAM v2/v3, Whisper, Vosk, Canary, SaluteSpeech.
- **Normalizes text** — pluggable normalizer pipeline, `е`/`ё` equalization, filler removal.
- **Computes metrics** — WER, CER, MER, WIL, WIP, Accuracy, IDR, LengthRatio, BERTScore, POS-breakdown.
- **Visualizes results** — interactive browser report with audio playback, error frequency analysis and word-level diff.

Everything is wired through a single `>>` pipeline operator.

```python
from plantain2asr import GolosDataset, Models, SimpleNormalizer, Metrics, ReportServer

ds   = GolosDataset("data/golos")
ds   >> Models.GigaAM_v3()
norm = ds >> SimpleNormalizer()
norm >> Metrics.composite()
ReportServer(norm, audio_dir="data/golos").serve()
```

## Install

```bash
pip install plantain2asr                   # core only
pip install plantain2asr[gigaam,analysis]  # + GigaAM models + analysis tools
pip install plantain2asr[all]              # everything
```

## Next steps

- [Quick Start](quickstart.md) — full runnable example
- [API Reference](api/dataloaders.md) — all classes and methods
- [Extending](extending/index.md) — add your own models, metrics, normalizers, report sections
