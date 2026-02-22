# plantain2asr

**Фреймворк для бенчмаркинга и анализа русскоязычных ASR-моделей.**

Загрузите датасет, примените модели, нормализуйте, посчитайте метрики, исследуйте результаты — всё через единый пайплайн с оператором `>>`.

```python
from plantain2asr import GolosDataset, Models, SimpleNormalizer, Metrics, ReportServer

ds   = GolosDataset("data/golos")       # автозагрузка при первом запуске
ds   >> Models.GigaAM_v3()             # инференс (кешируется)
ds   >> Models.Whisper()               # добавьте другие модели для сравнения

norm = ds >> SimpleNormalizer()         # нормализация (е/ё, пунктуация, регистр)
norm >> Metrics.composite()            # WER, CER, MER, WIL, Accuracy…

ReportServer(norm, audio_dir="data/golos").serve()  # открыть отчёт в браузере
```

---

## Как это работает

```mermaid
graph LR
    A[Датасет] -->|">> Model()"| B[Инференс]
    B -->|">> Normalizer()"| C[Нормализованный вид]
    C -->|">> Metrics()"| D[Метрики по семплам]
    D -->|"ReportServer"| E[Интерактивный отчёт]
```

Каждый шаг создаёт новый датасет — оригинал никогда не изменяется.
Можно ветвить, фильтровать и сравнивать на любом этапе.

---

## Установка

=== "Только ядро (без GPU)"
    ```bash
    pip install plantain2asr
    ```
    Включает: загрузку датасетов, расчёт WER/CER, сервер отчётов.

=== "С моделями GigaAM"
    ```bash
    pip install plantain2asr[gigaam]
    ```

=== "С Whisper"
    ```bash
    pip install plantain2asr[whisper]
    ```

=== "С инструментами анализа"
    ```bash
    pip install plantain2asr[analysis]
    ```
    Включает: pandas, BERTScore, POS-анализ, bootstrap-доверительные интервалы.

=== "Всё сразу"
    ```bash
    pip install plantain2asr[all]
    ```

---

## Поддерживаемые модели

| Модель | Extra | Устройство |
|---|---|---|
| GigaAM v3 (e2e-rnnt, e2e-ctc, rnnt, ctc) | `gigaam` | CUDA / MPS / CPU |
| GigaAM v2 (v2-rnnt, v2-ctc) | `gigaam` | CUDA / MPS / CPU |
| Whisper large-v3 (HuggingFace) | `whisper` | CUDA / MPS / CPU |
| T-one RussianTone | `gigaam` | CUDA |
| Vosk | `vosk` | CPU |
| NVIDIA Canary | `canary` | CUDA |
| SaluteSpeech API | — | облако |

---

## Расширение

plantain2asr построен на четырёх абстрактных базовых классах.
Унаследуйтесь от любого — и он автоматически встраивается в пайплайн.

| Базовый класс | Что добавляет |
|---|---|
| `BaseNormalizer` | Правила нормализации текста |
| `BaseASRModel` | Новая ASR-модель |
| `BaseMetric` | Новая метрика качества |
| `BaseSection` | Новая вкладка в отчёте |

→ [Руководство по расширению](extending/index.md)

---

!!! tip "Следующий шаг"
    Перейдите к [Быстрому старту](quickstart.md) для полного рабочего примера.
