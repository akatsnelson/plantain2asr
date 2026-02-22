# Своя метрика

Метрика вычисляет оценку качества из эталонной транскрипции и гипотезы модели.
Реализуйте `BaseMetric` — и она встраивается в `dataset >> MyMetric()` автоматически.

## Контракт базового класса

```python
from plantain2asr.metrics.base import BaseMetric

class BaseMetric(ABC):
    @property
    def name(self) -> str: ...                                        # абстрактный

    def calculate(self, reference: str, hypothesis: str) -> float: ... # абстрактный

    # Необязательный: эффективный батчевый агрегат
    def calculate_batch(self, references: list, hypotheses: list) -> float: ...
```

## Минимальный пример: Syllable Error Rate

```python
from plantain2asr.metrics.base import BaseMetric

VOWELS = set("аеёиоуыэюя")

def syllables(text: str) -> int:
    return sum(1 for c in text.lower() if c in VOWELS)

class SyllableErrorRate(BaseMetric):
    """Разница в числе слогов, нормированная к длине эталона."""

    @property
    def name(self) -> str:
        return "SER"

    def calculate(self, reference: str, hypothesis: str) -> float:
        ref_s = syllables(reference)
        hyp_s = syllables(hypothesis)
        if ref_s == 0:
            return 0.0
        return abs(ref_s - hyp_s) / ref_s * 100.0
```

Применение:

```python
norm = ds >> SimpleNormalizer()
norm >> SyllableErrorRate()
```

## Доступ к результатам

После `norm >> SyllableErrorRate()` в каждом семпле доступно:

```python
sample.asr_results["GigaAM-v3-e2e-rnnt"]["metrics"]["SER"]   # float (0–100)
```

Или через pandas:

```python
df = norm.to_pandas()
df.groupby("model")["SER"].mean()
```
