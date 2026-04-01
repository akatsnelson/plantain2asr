# Reporting

Reporting is available in two forms:

- live local browser report via `ReportServer`
- shareable static HTML export via `ReportBuilder.save_static_html()`

## `ReportServer`

```python
from plantain2asr import ReportServer

ReportServer(dataset, audio_dir="data/golos").serve()
ReportServer(dataset, audio_dir="data/golos", port=9000, sections=[MySection()]).serve()
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | `BaseASRDataset` | required | Usually a normalized dataset with metrics |
| `audio_dir` | `str` | `""` | Root for serving audio files |
| `port` | `int` | `8765` | HTTP port |
| `sections` | `list[BaseSection]` | `None` | Extra tabs appended after defaults |

Default tabs:

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

Use `save_static_html()` when you need a report that can be opened without running a local server.

This is also what `Experiment.save_report_html()` delegates to.

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

See [Custom Report Section](../extending/custom_section.md) for a full example.

## Built-in sections

### `MetricsSection`

Sortable metrics table with model-aware filtering.

### `ErrorFrequencySection`

Top substitutions, insertions, and deletions with drill-down examples and audio playback.

### `DiffSection`

Word-level alignment between reference and hypothesis, with audio access in both live and static modes.
