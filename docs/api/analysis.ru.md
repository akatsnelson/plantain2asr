# Анализ

Инструменты для глубокого исследования ошибок ASR.

## WordErrorAnalyzer

Разбивка ошибок по словам: топ замен, удалений, вставок.

```python
from plantain2asr import WordErrorAnalyzer

norm >> WordErrorAnalyzer(model_name="GigaAM-v3-e2e-rnnt", top_n=20)
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `model_name` | `str` | обязательный | Модель для анализа |
| `top_n` | `int` | `20` | Число топ-ошибок |

---

## DiffVisualizer

Генерирует HTML-отчёт с diff (статичный файл). Для интерактивной работы предпочтите `ReportServer`.

```python
from plantain2asr import DiffVisualizer

norm >> DiffVisualizer(model_name="GigaAM-v3-e2e-rnnt", output="reports/diff.html")
```

---

## Остальные анализаторы

Требуют: `pip install plantain2asr[analysis]`

| Класс | Описание |
|---|---|
| `PerformanceAnalyzer` | RTF (Real Time Factor) и анализ задержки |
| `BootstrapAnalyzer` | Bootstrap доверительные интервалы для WER |
| `AgreementAnalyzer` | Межмодельное согласие (каппа Коэна) |
| `TopicAnalyzer` | Разбивка ошибок по теме/домену |
| `HallucinationAnalyzer` | Обнаружение галлюцинаций — слов, которых нет в эталоне |
| `DurationAnalyzer` | Корреляция ошибок с длительностью аудио |
| `NgramErrorAnalyzer` | Паттерны ошибок на уровне n-грамм |
| `CalibrationAnalyzer` | Анализ калибровки уверенности модели |
