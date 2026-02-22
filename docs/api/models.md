# Models

## Models factory

Centralized access to all supported ASR models.

```python
from plantain2asr import Models
```

**Available models:**

| Factory method | Model | Extra |
|---|---|---|
| `Models.GigaAM_v3()` | GigaAM v3 e2e-RNNT (default) | `gigaam` |
| `Models.GigaAM_v3(model_name="e2e_ctc")` | GigaAM v3 e2e-CTC | `gigaam` |
| `Models.GigaAM_v3(model_name="rnnt")` | GigaAM v3 RNNT | `gigaam` |
| `Models.GigaAM_v3(model_name="ctc")` | GigaAM v3 CTC | `gigaam` |
| `Models.GigaAM_v2(model_name="v2_rnnt")` | GigaAM v2 RNNT | `gigaam` |
| `Models.GigaAM_v2(model_name="v2_ctc")` | GigaAM v2 CTC | `gigaam` |
| `Models.Whisper()` | Whisper large-v3 RU | `whisper` |
| `Models.Tone()` | T-one RussianTone | `gigaam` |
| `Models.Vosk(model_path=...)` | Vosk (offline, CPU) | `vosk` |
| `Models.Canary()` | NVIDIA Canary | `canary` |
| `Models.SaluteSpeech()` | SaluteSpeech cloud API | — |

**Usage:**

```python
ds >> Models.GigaAM_v3()     # inference, results cached automatically
ds >> Models.Whisper()        # add second model for comparison
```

---

## BaseASRModel

Abstract base class for all models. Subclass to add your own.

```python
from plantain2asr.models.base import BaseASRModel
```

**Interface:**

```python
class BaseASRModel(ABC):
    @property
    def name(self) -> str: ...                           # unique model id

    def transcribe(self, audio_path: str) -> str: ...    # single file
    def transcribe_batch(self, paths: list) -> list: ... # batch (optional)

    @property
    def is_e2e(self) -> bool: return False               # True if model outputs punctuation
```

See [Custom Model](../extending/custom_model.md) for a full example.
