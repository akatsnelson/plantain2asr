# Метрики

Метрики считаются на нормализованных представлениях датасета и сохраняются по каждой модели и каждому семплу.

## Фабрика `Metrics`

```python
from plantain2asr import Metrics
```

| Метод фабрики | Описание |
|---|---|
| `Metrics.composite()` | Рекомендуемый батчевый расчёт набора базовых метрик |
| `Metrics.WER()` | Word Error Rate |
| `Metrics.CER()` | Character Error Rate |
| `Metrics.MER()` | Match Error Rate |
| `Metrics.WIL()` | Word Information Lost |
| `Metrics.WIP()` | Word Information Preserved |
| `Metrics.Accuracy()` | `1 - MER` |
| `Metrics.IDR()` | Insertion / Deletion Ratio |
| `Metrics.LengthRatio()` | Длина гипотезы относительно эталона |
| `Metrics.BERTScore()` | Семантическое сходство, требует `analysis` |
| `Metrics.POSAnalysis()` | Анализ ошибок по частям речи, требует `analysis` |

Типичное использование:

```python
norm >> Metrics.composite()
norm >> Metrics.WER()
metric = Metrics.get("cer")
```

Неизвестное имя метрики приводит к явной ошибке с подсказками.

Форма хранения:

```python
sample.asr_results["GigaAM-v3-e2e-rnnt"]["metrics"]["WER"]
```

## `BaseMetric`

```python
from plantain2asr.metrics.base import BaseMetric
```

```python
class BaseMetric(ABC):
    @property
    def name(self) -> str: ...

    def calculate(self, reference: str, hypothesis: str) -> float: ...
    def calculate_batch(self, references: list, hypotheses: list) -> float: ...
```

`calculate_batch` стоит переопределять, когда батчевая или векторизованная реализация дешевле поштучной.

## `CompositeMetric`

`CompositeMetric` считает базовые метрики за один проход и является основным рекомендуемым способом оценки.

```python
from plantain2asr.metrics.composite import CompositeMetric

norm >> CompositeMetric()
```

→ [Своя метрика](../extending/custom_metric.md)
