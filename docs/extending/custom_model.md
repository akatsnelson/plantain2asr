# Custom Model

A model in plantain2asr is any class that:

1. Inherits `BaseASRModel`
2. Implements `name` (property) and `transcribe(audio_path)` (method)
3. Optionally overrides `transcribe_batch` for efficient batched inference

After that, `dataset >> MyModel()` works out of the box.

## Base class contract

```python
from plantain2asr.models.base import BaseASRModel

class BaseASRModel(ABC):
    @property
    def name(self) -> str: ...             # abstract — unique model identifier

    def transcribe(self, audio_path) -> str: ...    # abstract — one file
    def transcribe_batch(self, paths) -> List[str]: # optional — default: loop over transcribe()

    @property
    def is_e2e(self) -> bool: return False  # set True if model outputs punctuation
```

## Minimal example: a dummy model

```python
from plantain2asr.models.base import BaseASRModel

class EchoModel(BaseASRModel):
    """Returns filename stem as transcript (useful for testing)."""

    @property
    def name(self) -> str:
        return "EchoModel"

    def transcribe(self, audio_path) -> str:
        from pathlib import Path
        return Path(audio_path).stem
```

## REST API model

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
            resp = requests.post(
                self.endpoint,
                headers=self.headers,
                files={"audio": f},
            )
        resp.raise_for_status()
        return resp.json()["text"]
```

## Local HuggingFace model

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

## Registering in the Models factory (optional)

If you want `Models.MyModel()` to work, add a static method to the factory:

```python
# plantain2asr/models/factory.py  (or your own extension file)
from plantain2asr.models.factory import Models
from my_package import HFWhisperModel

# Monkey-patch the factory — or subclass it
Models.HFWhisper = staticmethod(lambda **kw: HFWhisperModel(**kw))
```

## Caching

`BaseASRModel.apply_to()` stores results in `plantain2asr/asr_data/<dataset_name>/<model_name>_results.jsonl`
and skips already-processed samples on re-runs. No extra code needed.
