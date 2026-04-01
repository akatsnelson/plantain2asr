# Датасеты

Датасеты - это основа библиотеки. Они хранят семплы, выводы моделей, метрики и export-ready представления.

## BaseASRDataset

```python
from plantain2asr.dataloaders.base import BaseASRDataset
```

Основные обязанности:

- хранить объекты `AudioSample`
- применять процессоры через `>>`
- кешировать выводы моделей
- отдавать табличные и экспортные представления
- не позволять запускать бессмысленные сценарии на пустом датасете

Наиболее полезные методы:

| Метод | Что делает |
|---|---|
| `filter(fn)` | Возвращает отфильтрованное представление датасета |
| `take(n)` | Возвращает первые `n` семплов |
| `run_model(model)` | Запускает модель напрямую без `>>` |
| `evaluate_metric(metric)` | Считает одну метрику напрямую |
| `to_pandas()` | Возвращает таблицу: одна строка на `(семпл, модель)` |
| `iter_results_rows()` | Итерирует плоские строки результатов |
| `save_csv(path)` | Экспортирует строки результатов в CSV |
| `save_excel(path)` | Экспортирует строки результатов в XLSX |
| `summarize_by_model()` | Собирает агрегированные метрики по моделям |
| `load_model_results(name, path)` | Загружает готовый JSONL с инференсом |

Пайплайн-форма:

```python
dataset >> model
dataset >> normalizer
dataset >> metric
```

## AudioSample

```python
from plantain2asr.dataloaders.types import AudioSample
```

| Поле | Тип | Описание |
|---|---|---|
| `id` | `str` | Уникальный идентификатор семпла |
| `audio_path` | `str` | Путь к аудиофайлу |
| `text` | `str` | Эталонная транскрипция |
| `duration` | `float \| None` | Длительность в секундах |
| `meta` | `dict` | Произвольные метаданные |
| `asr_results` | `dict` | Гипотезы и метрики по моделям |

## Встроенные загрузчики

### `GolosDataset`

```python
from plantain2asr import GolosDataset

ds = GolosDataset("data/golos")
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `root_dir` | `str` | обязательный | Директория хранения корпуса |
| `limit` | `int \| None` | `None` | Необязательный лимит семплов |
| `auto_download` | `bool` | `True` | Автозагрузка при отсутствии файлов |

Типичное поле метаданных: `meta["subset"]` равно `"crowd"` или `"farfield"`.

### `DagrusDataset`

```python
from plantain2asr import DagrusDataset

ds = DagrusDataset("data/dagrus")
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `root_dir` | `str` | обязательный | Корень корпуса |
| `limit` | `int \| None` | `None` | Необязательный лимит семплов |

### `NeMoDataset`

```python
from plantain2asr import NeMoDataset

ds = NeMoDataset("data/my_corpus")
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `root_dir` | `str` | обязательный | Директория с `manifest.jsonl` |
| `limit` | `int \| None` | `None` | Необязательный лимит семплов |

## Когда вместо этого брать `Experiment`

Если нужен готовый исследовательский workflow, берите `Experiment` поверх датасета, а не оркестрируйте каждый низкоуровневый вызов вручную.
