# Модели

## Фабрика Models

Централизованный доступ ко всем поддерживаемым ASR-моделям.

```python
from plantain2asr import Models
```

| Метод фабрики | Модель | Extra |
|---|---|---|
| `Models.GigaAM_v3()` | GigaAM v3 e2e-RNNT (по умолчанию) | `gigaam` |
| `Models.GigaAM_v3(model_name="e2e_ctc")` | GigaAM v3 e2e-CTC | `gigaam` |
| `Models.GigaAM_v3(model_name="rnnt")` | GigaAM v3 RNNT | `gigaam` |
| `Models.GigaAM_v3(model_name="ctc")` | GigaAM v3 CTC | `gigaam` |
| `Models.GigaAM_v2(model_name="v2_rnnt")` | GigaAM v2 RNNT | `gigaam` |
| `Models.GigaAM_v2(model_name="v2_ctc")` | GigaAM v2 CTC | `gigaam` |
| `Models.Whisper()` | Whisper large-v3 RU | `whisper` |
| `Models.Tone()` | T-one RussianTone | `gigaam` |
| `Models.Vosk(model_path=...)` | Vosk (офлайн, CPU) | `vosk` |
| `Models.Canary()` | NVIDIA Canary | `canary` |
| `Models.SaluteSpeech()` | SaluteSpeech API | — |

```python
ds >> Models.GigaAM_v3()   # инференс, результаты кешируются автоматически
ds >> Models.Whisper()      # добавить вторую модель для сравнения
```

---

## BaseASRModel

Абстрактный базовый класс для всех моделей. Унаследуйтесь, чтобы добавить свою.

```python
from plantain2asr.models.base import BaseASRModel
```

```python
class BaseASRModel(ABC):
    @property
    def name(self) -> str: ...                           # уникальный id модели

    def transcribe(self, audio_path: str) -> str: ...    # один файл
    def transcribe_batch(self, paths: list) -> list: ... # батч (необязательный)

    @property
    def is_e2e(self) -> bool: return False               # True если модель выдаёт пунктуацию
```

→ [Своя модель](../extending/custom_model.md)
