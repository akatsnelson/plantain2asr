# Быстрый старт

Эта страница проводит через полный пайплайн `>>` -- от загрузки данных до просмотра результатов.

## Установка

```bash
pip install plantain2asr[asr-cpu]
```

Для GPU: `pip install plantain2asr[asr-gpu]`

## 1. Загрузить датасет

```python
from plantain2asr import GolosDataset

ds = GolosDataset("data/golos")
crowd = ds.filter(lambda s: s.meta["subset"] == "crowd")
```

## 2. Прогнать модели через `>>`

```python
from plantain2asr import Models

crowd >> Models.GigaAM_v3()
crowd >> Models.Whisper()
```

Результаты кешируются на диск -- повторные запуски пропускают уже обработанные семплы.

## 3. Нормализовать через `>>`

```python
from plantain2asr import SimpleNormalizer

norm = crowd >> SimpleNormalizer()
```

Исходный датасет не затронут; `norm` -- новое представление с нормализованными текстами.

## 4. Посчитать метрики через `>>`

```python
from plantain2asr import Metrics

norm >> Metrics.composite()
```

Считает WER, CER, MER, WIL, WIP, Accuracy, IDR и LengthRatio за один проход.

## 5. Исследовать результаты

=== "Pandas"
    ```python
    df = norm.to_pandas()
    print(df.groupby("model")[["WER", "CER", "Accuracy"]].mean().sort_values("WER"))
    ```

=== "Интерактивный отчёт"
    ```python
    from plantain2asr import ReportServer
    ReportServer(norm, audio_dir="data/golos").serve()
    ```

=== "Статический HTML-отчёт"
    ```python
    from plantain2asr.reporting.builder import ReportBuilder
    ReportBuilder(norm).save_static_html("artifacts/report.html")
    ```

=== "CSV-экспорт"
    ```python
    norm.save_csv("artifacts/results.csv")
    ```

## Полный пайплайн одним блоком

```python
from plantain2asr import GolosDataset, Models, SimpleNormalizer, Metrics, ReportServer

ds = GolosDataset("data/golos")

ds >> Models.GigaAM_v3()
ds >> Models.Whisper()

norm = ds >> SimpleNormalizer()
norm >> Metrics.composite()

df = norm.to_pandas()
print(df.groupby("model")[["WER", "CER"]].mean().sort_values("WER"))

ReportServer(norm, audio_dir="data/golos").serve()
```

## Загрузка готовых результатов

Если инференс был выполнен на другой машине, загрузите JSONL и продолжите оценку локально:

```python
ds = GolosDataset("data/golos")
ds.load_model_results("GigaAM-v3-rnnt", "results/GigaAM-v3-rnnt_results.jsonl")
```

Одна строка на семпл:

```json
{"audio_path": "/любой/путь/file.wav", "hypothesis": "распознанный текст", "processing_time": 1.23}
```

Сопоставление идёт по basename `audio_path`, поэтому результаты удобно переносить между машинами.

## Обёртка `Experiment`

Если нужны готовые исследовательские сценарии без ручной сборки `>>` цепочки,
`Experiment` оборачивает те же шаги пайплайна:

```python
from plantain2asr import Experiment, GolosDataset, Models, SimpleNormalizer

experiment = Experiment(
    dataset=GolosDataset("data/golos"),
    models=[Models.GigaAM_v3(), Models.Whisper()],
    normalizer=SimpleNormalizer(),
)

experiment.compare_on_corpus(metrics=["WER", "CER", "Accuracy"])
```

| Метод | Что делает |
|---|---|
| `compare_on_corpus()` | Прогнать модели, нормализовать, оценить, вернуть сравнительную таблицу |
| `leaderboard()` | Рейтинг моделей по одной метрике |
| `prepare_thesis_tables()` | CSV-таблицы для диссертации |
| `export_appendix_bundle()` | Полный пакет: таблицы + отчёт + бенчмарк |
| `benchmark_models()` | Замеры latency, throughput, RTF |
| `save_report_html()` | Статический HTML-отчёт |

Под капотом `Experiment` выполняет те же `>>` шаги.
Используйте его, когда нужен однострочник; используйте пайплайн, когда нужен контроль.
