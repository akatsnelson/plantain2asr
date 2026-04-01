# Отчёты

Отчёты доступны в двух формах:

- живой локальный browser-report через `ReportServer`
- переносимый статический HTML через `ReportBuilder.save_static_html()`

## `ReportServer`

```python
from plantain2asr import ReportServer

ReportServer(dataset, audio_dir="data/golos").serve()
ReportServer(dataset, audio_dir="data/golos", port=9000, sections=[MySection()]).serve()
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `dataset` | `BaseASRDataset` | обязательный | Обычно нормализованный датасет с метриками |
| `audio_dir` | `str` | `""` | Корень для раздачи аудиофайлов |
| `port` | `int` | `8765` | HTTP-порт |
| `sections` | `list[BaseSection]` | `None` | Дополнительные вкладки после стандартных |

Стандартные вкладки:

- `Metrics`
- `Error Frequency`
- `Diff`

## `ReportBuilder`

```python
from plantain2asr.reporting.builder import ReportBuilder

builder = ReportBuilder(dataset)
data = builder.build()
builder.save_static_html("artifacts/report.html")
```

`save_static_html()` нужен, когда отчёт надо открыть без локального сервера.

Именно на него также опирается `Experiment.save_report_html()`.

## `BaseSection`

```python
from plantain2asr import BaseSection
```

```python
class BaseSection(ABC):
    @property
    def name(self) -> str: ...
    @property
    def title(self) -> str: ...
    @property
    def icon(self) -> str: ...

    def compute(self, dataset) -> dict: ...
    def js_function(self) -> str: ...

    def panel_html(self) -> str: ...
    def css(self) -> str: ...
```

→ [Своя вкладка отчёта](../extending/custom_section.md)

## Встроенные секции

### `MetricsSection`

Сортируемая таблица метрик с фильтрацией по модели.

### `ErrorFrequencySection`

Топ замен, вставок и удалений с drill-down в примеры и воспроизведением аудио.

### `DiffSection`

Пословное выравнивание эталона и гипотезы с доступом к аудио как в live-, так и в static-режиме.
