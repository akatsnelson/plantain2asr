# Reporting

Interactive browser report served locally. Opens at `http://localhost:8765`.

## ReportServer

```python
from plantain2asr import ReportServer

ReportServer(dataset, audio_dir="data/golos").serve()
# or with custom port and sections
ReportServer(dataset, audio_dir="data/golos", port=9000, sections=[MySection()]).serve()
```

**Constructor:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | `BaseASRDataset` | required | Normalized dataset with computed metrics |
| `audio_dir` | `str` | `""` | Root directory for audio file serving |
| `port` | `int` | `8765` | HTTP port |
| `sections` | `list[BaseSection]` | `None` | Additional sections to append after built-in tabs |

Built-in tabs: **Metrics**, **Error Frequency**, **Diff**.

---

## ReportBuilder

Collects and structures data from all sections. Used internally by `ReportServer`.

```python
from plantain2asr.reporting.builder import ReportBuilder

builder = ReportBuilder(dataset, sections=[...])
data = builder.build()   # dict keyed by section name
```

---

## BaseSection

Abstract base class for report tabs.

```python
from plantain2asr import BaseSection
```

**Interface:**

```python
class BaseSection(ABC):
    @property
    def name(self) -> str: ...     # unique id, e.g. "length"
    @property
    def title(self) -> str: ...    # tab label
    @property
    def icon(self) -> str: ...     # emoji icon

    def compute(self, dataset) -> dict: ...    # called once at startup → JSON
    def js_function(self) -> str: ...         # JS string defining render_{name}()

    def panel_html(self) -> str: ...   # optional: inner HTML (default: spinner)
    def css(self) -> str: ...          # optional: section CSS
```

See [Custom Report Section](../extending/custom_section.md) for a full example.

---

## Built-in sections

### MetricsSection

Renders a sortable metrics table with model filter.

### ErrorFrequencySection

Shows substitution/deletion/insertion frequency ranked by count.
Clicking a word opens sample-level examples with full word diff and audio playback.

### DiffSection

Word-level diff view between reference and hypothesis with audio playback.
