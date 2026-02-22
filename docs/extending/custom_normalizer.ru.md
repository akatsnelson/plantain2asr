# Свой нормализатор

Нормализация преобразует эталонные транскрипции и гипотезы модели перед подсчётом метрик.
Нормализатор применяется на **уровне датасета** — `dataset >> MyNormalizer()` создаёт новый вид;
оригинал не изменяется.

## Контракт базового класса

```python
from plantain2asr import BaseNormalizer

class BaseNormalizer(ABC):
    def normalize_ref(self, text: str) -> str: ...   # абстрактный — обязательно реализовать
    def normalize_hyp(self, text: str) -> str: ...   # необязательный, по умолчанию = normalize_ref
```

`normalize_ref` обрабатывает эталонные транскрипции (могут содержать аннотации корпуса).
`normalize_hyp` обрабатывает вывод модели. По умолчанию вызывает `normalize_ref`.

## Минимальный пример

Убрать пунктуацию, привести к нижнему регистру:

```python
import re
from plantain2asr import BaseNormalizer

class StripPunctNormalizer(BaseNormalizer):
    _RE = re.compile(r"[^\w\s]", re.UNICODE)

    def normalize_ref(self, text: str) -> str:
        text = text.lower()
        text = self._RE.sub("", text)
        text = text.replace("ё", "е")
        return " ".join(text.split())
```

Использование:

```python
norm = ds >> StripPunctNormalizer()
norm >> Metrics.composite()
```

## Разная обработка ref и hyp

Аннотации корпуса (`[laugh]`, `{word*}`) есть только в эталоне — убираем в `normalize_ref`,
`normalize_hyp` оставляем без изменений:

```python
import re
from plantain2asr import BaseNormalizer

class AnnotatedCorpusNormalizer(BaseNormalizer):
    _ANNOT = re.compile(r"\[.*?\]|\{.*?\*\}")

    def normalize_ref(self, text: str) -> str:
        text = self._ANNOT.sub("", text)
        return text.lower().strip()

    def normalize_hyp(self, text: str) -> str:
        return text.lower().strip()
```

## Композиция нормализаторов

```python
norm1 = ds >> StripPunctNormalizer()
norm2 = norm1 >> AnnotatedCorpusNormalizer()   # применяется поверх
```

!!! tip "Полный пример"
    Смотрите `plantain2asr/normalization/dagrus.py` — полноценная реализация с обработкой
    аннотаций DaGRuS, слов-паразитов, коллоквиализмов и эквивалентности е/ё.
