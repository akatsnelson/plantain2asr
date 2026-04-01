# Быстрый старт

Эта страница показывает рекомендуемый маршрут для нового пользователя:

1. выбрать датасет,
2. прогнать несколько моделей,
3. нормализовать и посчитать метрики,
4. выгрузить готовые артефакты.

Если сначала нужен визуальный сборщик, откройте [Интерактивный конструктор](constructor.html).

## Рекомендуемая установка

```bash
pip install plantain2asr[asr-cpu]
```

Для GPU-ориентированного локального сценария:

```bash
pip install plantain2asr[asr-gpu]
```

Дополнительный исследовательский слой:

```bash
pip install plantain2asr[analysis]
```

## Рекомендуемый workflow: `Experiment`

```python
from plantain2asr import Experiment, GolosDataset, Models, SimpleNormalizer

dataset = GolosDataset("data/golos")

experiment = Experiment(
    dataset=dataset,
    models=[
        Models.GigaAM_v3(),
        Models.Whisper(),
    ],
    normalizer=SimpleNormalizer(),
)
```

### Шаг 1: сравнить модели на одном корпусе

```python
comparison = experiment.compare_on_corpus(metrics=["WER", "CER", "Accuracy"])
print(comparison["metrics_by_model"])
```

Что вы получаете:

- инференс моделей с кешированием
- нормализованный evaluation-view
- агрегированную таблицу метрик по моделям

### Шаг 2: посмотреть рейтинг

```python
leaderboard = experiment.leaderboard(metric="WER")
print(leaderboard)
```

### Шаг 3: выгрузить артефакты

```python
experiment.save_leaderboard_csv("artifacts/leaderboard.csv", metric="WER")
experiment.save_report_html("artifacts/report.html")
experiment.export_error_cases("artifacts/error_cases.csv", metric="WER")
```

### Шаг 4: сделать материалы для диссертации

```python
tables = experiment.prepare_thesis_tables(
    metrics=["WER", "CER", "Accuracy"],
    output_dir="artifacts/thesis",
)

bundle = experiment.export_appendix_bundle(
    output_dir="artifacts/appendix",
    include_report=True,
    include_benchmark=True,
)
```

Используйте эти preset-сценарии по ситуации:

- `compare_on_corpus()` для обычного сравнения моделей
- `prepare_thesis_tables()` для чистых агрегированных таблиц
- `export_appendix_bundle()` для полного пакета артефактов
- `benchmark_models()` для замеров latency, throughput и RTF

## Продвинутый workflow: прямой pipeline

Используйте pipeline API, когда нужен явный контроль над каждым шагом.

### Загрузить датасет

```python
from plantain2asr import GolosDataset

ds = GolosDataset("data/golos")
crowd = ds.filter(lambda s: s.meta["subset"] == "crowd")
```

### Запустить инференс

```python
from plantain2asr import Models

crowd >> Models.GigaAM_v3()
crowd >> Models.Whisper()
```

Результаты кешируются на диск, поэтому повторные запуски пропускают уже обработанные семплы.

### Нормализовать и посчитать метрики

```python
from plantain2asr import SimpleNormalizer, Metrics

norm = crowd >> SimpleNormalizer()
norm >> Metrics.composite()
```

### Исследовать результаты

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

## Загрузка готовых результатов

Если инференс был выполнен на другой машине, можно локально загрузить JSONL и продолжить оценку:

```python
ds = GolosDataset("data/golos")
ds.load_model_results("GigaAM-v3-rnnt", "results/GigaAM-v3-rnnt_results.jsonl")
```

Одна строка на семпл:

```json
{"audio_path": "/любой/путь/file.wav", "hypothesis": "распознанный текст", "processing_time": 1.23}
```

Сопоставление идёт по basename `audio_path`, поэтому результаты удобно переносить между машинами.
