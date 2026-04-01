# Models

## `Models` factory

```python
from plantain2asr import Models
```

The factory is the main entry point for built-in backends.

Supported calls:

| Call | Backend | Extra | Device |
|---|---|---|---|
| `Models.GigaAM_v3()` | GigaAM v3 e2e-RNNT | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v3(model_name="e2e_ctc")` | GigaAM v3 e2e-CTC | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v3(model_name="rnnt")` | GigaAM v3 RNNT | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v3(model_name="ctc")` | GigaAM v3 CTC | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v2(model_name="v2_rnnt")` | GigaAM v2 RNNT | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v2(model_name="v2_ctc")` | GigaAM v2 CTC | `gigaam` | CUDA / MPS / CPU |
| `Models.Whisper()` | Whisper large-v3 RU | `whisper` | CUDA / MPS / CPU |
| `Models.Tone()` | T-one | `tone` + T-One source archive | CUDA / CPU |
| `Models.Vosk(model_path=...)` | Vosk | `vosk` | CPU |
| `Models.Canary()` | NVIDIA Canary | `canary` | CUDA |
| `Models.SaluteSpeech()` | SaluteSpeech API | none | cloud |

The factory also supports flexible name resolution:

```python
model = Models.create("gigaam-v3")
model = Models.create("GigaAM_v3")
model = Models.create("tone")
```

Unknown names raise a helpful error with close suggestions.

For T-One, install the runtime extra first and then the source archive:

```bash
pip install plantain2asr[tone]
pip install "tone @ https://github.com/voicekit-team/T-one/archive/3c5b6c015038173840e62cea99e10cdb1c759116.tar.gz"
```

## Usage

```python
ds >> Models.GigaAM_v3()
ds >> Models.Whisper()
```

Model outputs are cached automatically and can be reused across later metric, report, and export steps.

## `BaseASRModel`

```python
from plantain2asr.models.base import BaseASRModel
```

```python
class BaseASRModel(ABC):
    @property
    def name(self) -> str: ...

    def transcribe(self, audio_path: str) -> str: ...
    def transcribe_batch(self, paths: list) -> list: ...

    @property
    def is_e2e(self) -> bool: ...
```

Training-capable models can additionally expose training metadata used by the training layer.

See [Custom Model](../extending/custom_model.md) for extension patterns.
