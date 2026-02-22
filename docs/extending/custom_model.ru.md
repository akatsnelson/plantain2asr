# Своя модель

Модель в plantain2asr — это любой класс, который:

1. Наследует `BaseASRModel`
2. Реализует `name` (свойство) и `transcribe(audio_path)` (метод)
3. Опционально переопределяет `transcribe_batch` для батчевого инференса

После этого `dataset >> MyModel()` работает из коробки.

## Контракт базового класса

```python
from plantain2asr.models.base import BaseASRModel

class BaseASRModel(ABC):
    @property
    def name(self) -> str: ...              # абстрактный — уникальный идентификатор

    def transcribe(self, audio_path) -> str: ...     # абстрактный — один файл
    def transcribe_batch(self, paths) -> list: ...   # необязательный — по умолчанию: цикл

    @property
    def is_e2e(self) -> bool: return False  # True если модель выдаёт пунктуацию
```

## Минимальный пример

```python
from plantain2asr.models.base import BaseASRModel

class EchoModel(BaseASRModel):
    """Возвращает имя файла как транскрипцию (удобно для тестов)."""

    @property
    def name(self) -> str:
        return "EchoModel"

    def transcribe(self, audio_path) -> str:
        from pathlib import Path
        return Path(audio_path).stem
```

## Модель через REST API

```python
import requests
from plantain2asr.models.base import BaseASRModel

class MyAPIModel(BaseASRModel):
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.headers  = {"Authorization": f"Bearer {api_key}"}

    @property
    def name(self) -> str:
        return "MyAPIModel"

    def transcribe(self, audio_path) -> str:
        with open(audio_path, "rb") as f:
            resp = requests.post(self.endpoint, headers=self.headers, files={"audio": f})
        resp.raise_for_status()
        return resp.json()["text"]
```

## Локальная HuggingFace модель

```python
import torch
import librosa
from transformers import pipeline
from plantain2asr.models.base import BaseASRModel

class HFWhisperModel(BaseASRModel):
    def __init__(self, model_id: str = "openai/whisper-large-v3"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            generate_kwargs={"language": "ru"},
        )
        self._name = model_id.split("/")[-1]

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, audio_path) -> str:
        audio, sr = librosa.load(audio_path, sr=16_000)
        result = self.pipe({"array": audio, "sampling_rate": sr})
        return result["text"].strip()
```

!!! info "Кеширование"
    `BaseASRModel.apply_to()` сохраняет результаты в
    `plantain2asr/asr_data/<dataset>/<model>_results.jsonl`
    и пропускает уже обработанные семплы при повторном запуске.
