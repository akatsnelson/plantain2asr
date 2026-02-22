# Своя вкладка отчёта

Интерактивный отчёт состоит из **секций** (вкладок). Каждая секция — это Python-класс, который:

1. Собирает данные из датасета (`compute()`)
2. Предоставляет JavaScript для отрисовки (`js_function()`)
3. Опционально добавляет HTML-структуру (`panel_html()`) или CSS (`css()`)

Добавить новую вкладку = написать один класс. Менять `server.py`, `builder.py` или `template.py` не нужно.

## Контракт базового класса

```python
from plantain2asr import BaseSection

class BaseSection(ABC):
    @property
    def name(self) -> str: ...     # абстрактный — уникальный id, напр. "length"
    @property
    def title(self) -> str: ...    # абстрактный — заголовок вкладки
    @property
    def icon(self) -> str: ...     # абстрактный — emoji-иконка

    def compute(self, dataset) -> dict: ...    # абстрактный — JSON-сериализуемые данные
    def js_function(self) -> str: ...         # абстрактный — JS со функцией render_{name}()

    def panel_html(self) -> str: ...  # необязательный — внутренний HTML (по умолчанию: спиннер)
    def css(self) -> str: ...         # необязательный — CSS секции
```

## Минимальный пример: статистика длины

```python
from plantain2asr import BaseSection

class LengthSection(BaseSection):
    @property
    def name(self) -> str:  return "length"
    @property
    def title(self) -> str: return "Длина"
    @property
    def icon(self) -> str:  return "📏"

    def compute(self, dataset) -> dict:
        rows = []
        for s in dataset:
            for model, res in s.asr_results.items():
                hyp = res.get("hypothesis", "") or ""
                rows.append({
                    "id":    s.id,
                    "model": model,
                    "ref_words": len(s.text.split()),
                    "hyp_words": len(hyp.split()),
                })
        return {"rows": rows}

    def js_function(self) -> str:
        return r"""
function render_length() {
    const rows  = S.data.length.rows.filter(r => r.model === S.activeModel);
    const avg_r = rows.reduce((a,b) => a + b.ref_words, 0) / (rows.length || 1);
    const avg_h = rows.reduce((a,b) => a + b.hyp_words, 0) / (rows.length || 1);
    document.getElementById('length-panel').innerHTML =
        '<p>Среднее слов в эталоне: <b>' + avg_r.toFixed(1) + '</b></p>' +
        '<p>Среднее слов в гипотезе: <b>' + avg_h.toFixed(1) + '</b></p>';
}
"""
```

## Подключение к ReportServer

```python
from plantain2asr import ReportServer

ReportServer(
    norm,
    audio_dir="data/golos",
    sections=[LengthSection()],    # добавляется после встроенных вкладок
).serve()
```

## Глобальные переменные внутри JS

| Переменная | Тип | Описание |
|---|---|---|
| `S.data` | `object` | Данные всех секций (ключ — имя секции) |
| `S.activeModel` | `string` | Текущая выбранная модель |
| `esc(s)` | `function` | HTML-экранирование строки |
| `fmtNum(v)` | `function` | Форматирование float до 2 знаков |

!!! tip "Встроенные примеры"
    Смотрите полные реализации в исходном коде:

    - `plantain2asr/reporting/sections/metrics.py`
    - `plantain2asr/reporting/sections/errors.py`
    - `plantain2asr/reporting/sections/diff.py`
