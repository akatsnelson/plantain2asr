# Модели

## Фабрика `Models`

```python
from plantain2asr import Models
```

Это главный вход в набор встроенных backend-ов.

Поддерживаемые вызовы:

| Вызов | Backend | Extra | Устройство |
|---|---|---|---|
| `Models.GigaAM_v3()` | GigaAM v3 e2e-RNNT | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v3(model_name="e2e_ctc")` | GigaAM v3 e2e-CTC | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v3(model_name="rnnt")` | GigaAM v3 RNNT | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v3(model_name="ctc")` | GigaAM v3 CTC | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v2(model_name="v2_rnnt")` | GigaAM v2 RNNT | `gigaam` | CUDA / MPS / CPU |
| `Models.GigaAM_v2(model_name="v2_ctc")` | GigaAM v2 CTC | `gigaam` | CUDA / MPS / CPU |
| `Models.Whisper()` | Whisper large-v3 RU | `whisper` | CUDA / MPS / CPU |
| `Models.Tone()` | T-one | `tone` | CUDA / CPU |
| `Models.Vosk(model_path=...)` | Vosk | `vosk` | CPU |
| `Models.Canary()` | NVIDIA Canary | `canary` | CUDA |
| `Models.SaluteSpeech()` | SaluteSpeech API | none | облако |

Фабрика также поддерживает гибкое разрешение имён:

```python
model = Models.create("gigaam-v3")
model = Models.create("GigaAM_v3")
model = Models.create("tone")
```

Для неизвестного имени выбрасывается понятная ошибка с подсказками.

## Использование

```python
ds >> Models.GigaAM_v3()
ds >> Models.Whisper()
```

Выводы моделей автоматически кешируются и потом переиспользуются в метриках, отчётах и экспортах.

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

Модели, поддерживающие обучение, могут дополнительно раскрывать training-метаданные для train-слоя.

→ [Своя модель](../extending/custom_model.md)
