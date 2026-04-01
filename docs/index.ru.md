# plantain2asr

**Фреймворк для бенчмаркинга и анализа русскоязычных ASR-моделей.**

Документация теперь построена по одному простому правилу: сначала самый простой вход, потом более низкоуровневые API только там, где они действительно нужны.

## Рекомендуемый маршрут входа

1. Откройте [Интерактивный конструктор](constructor.md), чтобы собрать цепочку визуально.
2. Используйте `Experiment` для типовых исследовательских сценариев.
3. Переходите к `>>`-пайплайну, когда нужна полная модульность и контроль.

## Что даёт plantain2asr

- Локальные и облачные ASR-модели под единым интерфейсом
- Автоматический выбор устройства там, где это поддерживается
- Представления датасета вместо мутаций "на месте"
- Встроенные нормализаторы, метрики, отчёты, анализ и бенчмарки
- Готовые экспортные сценарии для диссертации и приложений
- Модульную архитектуру для своих моделей, метрик и вкладок отчёта

## Самый полезный короткий пример

```python
from plantain2asr import Experiment, GolosDataset, Models, SimpleNormalizer

dataset = GolosDataset("data/golos")

experiment = Experiment(
    dataset=dataset,
    models=[Models.GigaAM_v3(), Models.Whisper()],
    normalizer=SimpleNormalizer(),
)

experiment.compare_on_corpus(metrics=["WER", "CER", "Accuracy"])
experiment.save_report_html("artifacts/report.html")
```

## Установка

=== "Ядро"
    ```bash
    pip install plantain2asr
    ```
    Включает датасеты, нормализацию, метрики, экспорты и отчёты.

=== "Типовой локальный CPU-стек"
    ```bash
    pip install plantain2asr[asr-cpu]
    ```

=== "Типовой локальный GPU-стек"
    ```bash
    pip install plantain2asr[asr-gpu]
    ```

=== "Extras по backend-ам"
    ```bash
    pip install plantain2asr[gigaam]
    pip install plantain2asr[whisper]
    pip install plantain2asr[vosk]
    pip install plantain2asr[canary]
    pip install plantain2asr[tone]
    ```

=== "Исследовательский анализ"
    ```bash
    pip install plantain2asr[analysis]
    ```

=== "Всё сразу"
    ```bash
    pip install plantain2asr[all]
    ```

Логика выбора устройства: сначала CUDA, затем MPS, затем CPU, если backend это поддерживает.

## Ментальная модель

```mermaid
graph LR
    A[Датасет] --> B[Модели]
    B --> C[Нормализатор]
    C --> D[Метрики]
    D --> E[Отчёты и анализ]
    E --> F[Экспорты и артефакты для диссертации]
```

Все эти шаги по-прежнему компонуются как пайплайн. `Experiment` просто оркестрирует те же строительные блоки для типовых исследовательских сценариев.

## Поддерживаемые семейства моделей

| Семейство | Типичный вызов | Extra | Устройство |
|---|---|---|---|
| GigaAM v3 | `Models.GigaAM_v3()` | `gigaam` | CUDA / MPS / CPU |
| GigaAM v2 | `Models.GigaAM_v2()` | `gigaam` | CUDA / MPS / CPU |
| Whisper | `Models.Whisper()` | `whisper` | CUDA / MPS / CPU |
| T-one | `Models.Tone()` | `tone` | CUDA / CPU |
| Vosk | `Models.Vosk(...)` | `vosk` | CPU |
| Canary | `Models.Canary()` | `canary` | CUDA |
| SaluteSpeech | `Models.SaluteSpeech()` | none | облако |

## Если вы заходите впервые

- Идите в [Интерактивный конструктор](constructor.md), если хотите быстро собрать цепочку и сразу увидеть код.
- Идите в [Быстрый старт](quickstart.md), если нужен канонический рабочий сценарий.
- Идите в [Справочник API](api/dataloaders.md), если вы уже знаете, какой блок вам нужен.
- Идите в [Расширение](extending/index.md), если хотите добавить свои компоненты.
