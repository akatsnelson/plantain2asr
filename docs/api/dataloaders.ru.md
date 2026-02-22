# Датасеты

## BaseASRDataset

Базовый класс для всех датасетов. Предоставляет интеграцию с пайплайном, кеширование, фильтрацию и расчёт метрик.

**Основные методы:**

| Метод | Описание |
|---|---|
| `filter(fn)` | Возвращает новый датасет с семплами, удовлетворяющими предикату |
| `take(n)` | Возвращает первые N семплов |
| `to_pandas()` | Возвращает `pd.DataFrame` — одна строка на пару (семпл, модель) |
| `load_model_results(model_name, jsonl_path)` | Загружает предвычисленные результаты инференса из JSONL-файла |
| `clone()` | Поверхностная копия датасета |

**Оператор пайплайна:**

```python
result = dataset >> processor   # применяет модель / нормализатор / метрику
```

---

## AudioSample

Контейнер данных для одного аудиофайла.

**Поля:**

| Поле | Тип | Описание |
|---|---|---|
| `id` | `str` | Уникальный идентификатор |
| `audio_path` | `str` | Абсолютный путь к аудиофайлу |
| `text` | `str` | Эталонная транскрипция |
| `duration` | `float \| None` | Длительность в секундах |
| `meta` | `dict` | Произвольные метаданные (напр. `{"subset": "crowd"}`) |
| `asr_results` | `dict` | `{model_name: {"hypothesis": str, "metrics": dict, ...}}` |

---

## DagrusDataset

Загрузчик корпуса DaGRuS (Дагестанская русская речь).

```python
from plantain2asr import DagrusDataset
ds = DagrusDataset("data/dagrus")
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `root_dir` | `str` | обязательный | Путь к корпусу |
| `limit` | `int \| None` | `None` | Максимальное число семплов |

---

## GolosDataset

Загрузчик тестовой части корпуса GOLOS. Автозагружается при первом запуске.

```python
from plantain2asr import GolosDataset
ds = GolosDataset("data/golos")
crowd = ds.filter(lambda s: s.meta["subset"] == "crowd")
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `root_dir` | `str` | обязательный | Путь для хранения корпуса |
| `limit` | `int \| None` | `None` | Максимальное число семплов |
| `auto_download` | `bool` | `True` | Скачать, если директория не найдена |

Каждый семпл содержит `meta["subset"]` = `"crowd"` или `"farfield"`.

---

## NeMoDataset

Загрузчик для датасетов в формате NeMo JSONL-манифеста.

```python
from plantain2asr import NeMoDataset
ds = NeMoDataset("data/my_corpus")
```
