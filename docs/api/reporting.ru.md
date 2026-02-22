# Отчёты

Интерактивный отчёт в браузере, запускается локально. Адрес: `http://localhost:8765`.

## ReportServer

```python
from plantain2asr import ReportServer

ReportServer(dataset, audio_dir="data/golos").serve()
ReportServer(dataset, audio_dir="data/golos", port=9000, sections=[MySection()]).serve()
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `dataset` | `BaseASRDataset` | обязательный | Нормализованный датасет с метриками |
| `audio_dir` | `str` | `""` | Корневая директория для раздачи аудио |
| `port` | `int` | `8765` | HTTP-порт |
| `sections` | `list[BaseSection]` | `None` | Дополнительные вкладки после встроенных |

Встроенные вкладки: **Метрики**, **Частота ошибок**, **Diff**.

---

## BaseSection

```python
from plantain2asr import BaseSection
```

```python
class BaseSection(ABC):
    @property
    def name(self) -> str: ...     # уникальный id
    @property
    def title(self) -> str: ...    # заголовок вкладки
    @property
    def icon(self) -> str: ...     # emoji-иконка

    def compute(self, dataset) -> dict: ...    # вызывается при старте → JSON
    def js_function(self) -> str: ...         # JS-строка с функцией render_{name}()

    def panel_html(self) -> str: ...   # необязательный
    def css(self) -> str: ...          # необязательный
```

→ [Своя вкладка отчёта](../extending/custom_section.md)

---

## Встроенные вкладки

### MetricsSection
Таблица метрик с фильтром по модели.

### ErrorFrequencySection
Топ замен/удалений/вставок по частоте. Клик по слову открывает примеры семплов с полным пословным diff и воспроизведением аудио.

### DiffSection
Пословный diff эталона и гипотезы с воспроизведением аудио.
