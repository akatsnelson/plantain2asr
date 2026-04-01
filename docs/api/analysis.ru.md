# Анализ

Аналитические инструменты позволяют перейти от итоговых метрик к реальной структуре ошибок.

## `WordErrorAnalyzer`

```python
from plantain2asr import WordErrorAnalyzer

norm >> WordErrorAnalyzer(model_name="GigaAM-v3-e2e-rnnt", top_n=20)
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `model_name` | `str` | обязательный | Какая модель анализируется |
| `top_n` | `int` | `20` | Сколько top-паттернов ошибок показать |

## `DiffVisualizer`

```python
from plantain2asr import DiffVisualizer

norm >> DiffVisualizer(model_name="GigaAM-v3-e2e-rnnt", output="reports/diff.html")
```

Это статический diff-артефакт. Для более широкого интерактивного анализа лучше использовать `ReportServer` или `Experiment.save_report_html()`.

## Бенчмарки

Для замеров runtime лучше использовать benchmark-слой:

```python
benchmarks = experiment.benchmark_models()
```

Он отдаёт сводки по latency, throughput и real-time factor на доступных устройствах.

## Остальные анализаторы

Все следующие классы требуют `plantain2asr[analysis]`:

| Класс | Описание |
|---|---|
| `PerformanceAnalyzer` | Анализ RTF и задержки |
| `BootstrapAnalyzer` | Bootstrap доверительные интервалы |
| `AgreementAnalyzer` | Межмодельное согласие |
| `TopicAnalyzer` | Ошибки по теме или домену |
| `HallucinationAnalyzer` | Обнаружение галлюцинаций |
| `DurationAnalyzer` | Связь ошибок с длительностью |
| `NgramErrorAnalyzer` | Паттерны ошибок на уровне n-грамм |
| `CalibrationAnalyzer` | Анализ калибровки confidence |
