# Быстрый старт

Полный пример: от загрузки датасета до интерактивного отчёта в браузере.

!!! note "Требования"
    ```bash
    pip install plantain2asr[gigaam,whisper]
    ```
    Для сервера отчётов дополнительные зависимости не нужны.

---

## Шаг 1 — Загрузить датасет

```python
from plantain2asr import GolosDataset

# Тестовая часть GOLOS — автозагрузка ~2.5 ГБ при первом запуске
ds = GolosDataset("data/golos")
print(f"Загружено {len(ds)} семплов")

# Фильтрация по сабсету
crowd    = ds.filter(lambda s: s.meta["subset"] == "crowd")
farfield = ds.filter(lambda s: s.meta["subset"] == "farfield")
```

!!! tip "Корпус DaGRuS"
    ```python
    from plantain2asr import DagrusDataset
    ds = DagrusDataset("data/dagrus")
    ```

---

## Шаг 2 — Запустить инференс

```python
from plantain2asr import Models

crowd >> Models.GigaAM_v3()   # результаты сохраняются в sample.asr_results
crowd >> Models.Whisper()     # добавьте ещё модели для сравнения
```

!!! info "Кеширование"
    Результаты кешируются на диске — повторный запуск той же модели на тех же данных
    пропустит уже обработанные семплы. Можно прерывать и возобновлять.

---

## Шаг 3 — Нормализовать

```python
from plantain2asr import SimpleNormalizer

# Создаёт новый вид датасета — crowd остаётся неизменным
norm = crowd >> SimpleNormalizer()
```

Нормализация выполняет: приведение к нижнему регистру, удаление пунктуации, `ё → е`.

!!! tip "Аннотации корпуса DaGRuS"
    Используйте `DagrusNormalizer`, чтобы также убрать `[laugh]`, `{word*}` и слова-паразиты:
    ```python
    from plantain2asr import DagrusNormalizer
    norm = ds >> DagrusNormalizer()
    ```

---

## Шаг 4 — Посчитать метрики

```python
from plantain2asr import Metrics

norm >> Metrics.composite()
# WER, CER, MER, WIL, WIP, Accuracy, IDR, LengthRatio — всё за один быстрый проход
```

---

## Шаг 5 — Исследовать результаты

=== "Pandas"
    ```python
    df = norm.to_pandas()
    print(df.groupby("model")[["WER", "CER", "Accuracy"]].mean().sort_values("WER"))
    ```

=== "Интерактивный отчёт"
    ```python
    from plantain2asr import ReportServer
    ReportServer(norm, audio_dir="data/golos").serve()
    ```
    Откройте **http://localhost:8765** — таблица метрик, частота ошибок с воспроизведением аудио, пословный diff.

=== "Ошибки по словам"
    ```python
    from plantain2asr import WordErrorAnalyzer
    norm >> WordErrorAnalyzer(model_name="GigaAM-v3-e2e-rnnt", top_n=20)
    ```

---

## Загрузка готовых результатов

Запустили инференс на GPU-машине, скопировали JSONL — загружаете здесь:

```python
ds = GolosDataset("data/golos")
ds.load_model_results("GigaAM-v3-rnnt", "results/GigaAM-v3-rnnt_results.jsonl")
```

Формат JSONL — одна строка на семпл:
```json
{"audio_path": "/любой/путь/file.wav", "hypothesis": "распознанный текст", "processing_time": 1.23}
```

!!! warning "Сопоставление по имени файла"
    Семплы сопоставляются по **basename** пути `audio_path`, а не по полному пути.
    Это позволяет использовать результаты, посчитанные на другой машине.
