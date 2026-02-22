# Метрики

## Фабрика Metrics

```python
from plantain2asr import Metrics
```

| Метод фабрики | Описание |
|---|---|
| `Metrics.composite()` | Все метрики ниже за один быстрый проход (рекомендуется) |
| `Metrics.WER()` | Word Error Rate — доля ошибочных слов |
| `Metrics.CER()` | Character Error Rate — доля ошибочных символов |
| `Metrics.MER()` | Match Error Rate |
| `Metrics.WIL()` | Word Information Lost |
| `Metrics.WIP()` | Word Information Preserved |
| `Metrics.Accuracy()` | 1 − MER |
| `Metrics.IDR()` | Insertion / Deletion Ratio |
| `Metrics.LengthRatio()` | Отношение длины гипотезы к длине эталона |
| `Metrics.BERTScore()` | Семантическое сходство (требует `analysis`) |
| `Metrics.POSAnalysis()` | Разбивка ошибок по частям речи (требует `analysis`) |

```python
norm >> Metrics.composite()   # рекомендуется: все метрики, один вызов jiwer
norm >> Metrics.WER()         # одна метрика
```

Результаты доступны в каждом семпле:

```python
sample.asr_results["GigaAM-v3-e2e-rnnt"]["metrics"]["WER"]   # float (0–100)
```

---

## BaseMetric

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

→ [Своя метрика](../extending/custom_metric.md)

---

## CompositeMetric

Вычисляет WER, CER, MER, WIL, WIP, Accuracy, IDR, LengthRatio за один вызов
`jiwer.process_words` + `jiwer.process_characters` на модель.
Примерно в **8 раз быстрее** поштучного подсчёта метрик.

```python
norm >> Metrics.composite()
```
