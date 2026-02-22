# Custom Report Section

The interactive browser report is built from **sections** (tabs). Each section is a Python class that:

1. Collects data from the dataset (`compute()`)
2. Provides the JavaScript that renders it (`js_function()`)
3. Optionally adds HTML structure (`panel_html()`) or CSS (`css()`)

Add a new tab = write one class. No changes to `server.py`, `builder.py`, or `template.py`.

## Base class contract

```python
from plantain2asr import BaseSection

class BaseSection(ABC):
    @property
    def name(self) -> str: ...     # abstract — unique id, e.g. "length"
    @property
    def title(self) -> str: ...    # abstract — tab label, e.g. "Length Stats"
    @property
    def icon(self) -> str: ...     # abstract — emoji icon, e.g. "📏"

    def compute(self, dataset) -> dict: ...    # abstract — JSON-serializable data
    def js_function(self) -> str: ...         # abstract — JS string with render_{name}()

    def panel_html(self) -> str: ...  # optional — inner HTML; default: spinner
    def css(self) -> str: ...         # optional — section-specific CSS
```

`compute()` is called once when the server starts. Its result is served at `/api/{name}` and
available in JavaScript as `S.data['{name}']`.

`js_function()` **must** define a global function named `render_{name}()`.
The template calls it automatically when the user switches to the tab.

## Minimal example: word-count distribution

```python
from plantain2asr import BaseSection

class LengthSection(BaseSection):
    @property
    def name(self) -> str:  return "length"
    @property
    def title(self) -> str: return "Length Stats"
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
        '<p>Avg ref words: <b>' + avg_r.toFixed(1) + '</b></p>' +
        '<p>Avg hyp words: <b>' + avg_h.toFixed(1) + '</b></p>';
}
"""
```

## Register with ReportServer

```python
from plantain2asr import ReportServer

ReportServer(
    norm,
    audio_dir="data/golos",
    sections=[LengthSection()],    # appended after built-in tabs
).serve()
```

## Accessing globals inside JS

Inside `js_function()` you have access to these globals provided by the base template:

| Global | Type | Description |
|---|---|---|
| `S.data` | `object` | All section data (keyed by section name) |
| `S.activeModel` | `string` | Currently selected model |
| `esc(s)` | `function` | HTML-escapes a string |
| `fmtNum(v)` | `function` | Formats a float to 2 decimal places |

## Two-column layout

Override `panel_html()` when you need custom HTML structure (e.g., a sidebar + main area):

```python
def panel_html(self) -> str:
    return """
<div style="display:flex;height:100%;gap:16px">
    <div id="length-sidebar" style="width:260px;overflow-y:auto"></div>
    <div id="length-main"    style="flex:1;overflow-y:auto"></div>
</div>
"""
```

## Adding custom CSS

```python
def css(self) -> str:
    return """
#length-panel .stat { font-size:1.4em; font-weight:bold; color:#4caf50; }
"""
```

## Full built-in examples

See the built-in sections for complete reference implementations:

- `plantain2asr/reporting/sections/metrics.py` — metrics table with model filter
- `plantain2asr/reporting/sections/errors.py` — error frequency + clickable examples
- `plantain2asr/reporting/sections/diff.py` — word-level diff with audio playback
