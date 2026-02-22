# Нормализация

Нормализация применяется на **уровне датасета** — создаёт новый вид, оригинал не изменяется.

```python
norm = dataset >> MyNormalizer()
```

## BaseNormalizer

```python
from plantain2asr import BaseNormalizer
```

```python
class BaseNormalizer(ABC):
    def normalize_ref(self, text: str) -> str: ...   # абстрактный
    def normalize_hyp(self, text: str) -> str: ...   # по умолчанию: вызывает normalize_ref
    def normalize_pair(self, ref, hyp) -> tuple: ... # удобный метод для пары
```

- `normalize_ref` — обрабатывает эталонные транскрипции
- `normalize_hyp` — обрабатывает вывод модели. Переопределите, если нужна разная обработка.

---

## SimpleNormalizer

Универсальный нормализатор для русского языка.

```python
from plantain2asr import SimpleNormalizer
norm = dataset >> SimpleNormalizer()
```

**Что делает:**

- Нижний регистр
- Удаление пунктуации
- `ё` → `е`
- Схлопывание пробелов

---

## DagrusNormalizer

Нормализатор, заточенный под формат аннотаций корпуса DaGRuS.

```python
from plantain2asr import DagrusNormalizer
norm = dataset >> DagrusNormalizer(remove_fillers=False, strip_punctuation=True)
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `remove_fillers` | `bool` | `False` | Удалять слова-паразиты (ага, угу, мм…) |
| `strip_punctuation` | `bool` | `True` | Удалять пунктуацию |

**Что делает дополнительно к SimpleNormalizer:**

- Удаляет аннотации: `[laugh]`, `[noise]`, `{word*}`
- Удаляет немаркированные события: "говорит на другом языке" и т.п.
- Опционально удаляет слова-паразиты
- Нормализует коллоквиализмы: "щас" → "сейчас", "ваще" → "вообще" и т.д.
- `ё` → `е`
