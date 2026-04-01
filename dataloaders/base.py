from abc import ABC
import json
import os
import copy
import math
import random
from typing import Iterator, List, Dict, TYPE_CHECKING, Union, Optional, Callable, Any, Tuple
try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:  # minimal stub when torch is not installed
        pass
from pathlib import Path
from ..utils.logging import get_logger
from . import io as dataset_io
from .types import AudioSample

# Пробуем импортировать tqdm
try:
    from tqdm.auto import tqdm
except ImportError:
    get_logger(__name__).warning("tqdm is not installed; progress bars are disabled.")
    def tqdm(iterable, **kwargs): return iterable

if TYPE_CHECKING:
    from ..models.base import BaseASRModel
    from ..metrics.base import BaseMetric

logger = get_logger(__name__)

class BaseASRDataset(ABC, Dataset):
    """
    Абстрактный базовый класс для ASR датасетов.
    Поддерживает:
    1. Visitor Pattern (.apply)
    2. Functional Pipeline (.filter, .sort, .take, .split)
    3. Magic Operator (>>)
    """
    
    def __init__(self):
        self._samples: List[AudioSample] = []
        self._id_map = {}  # Для быстрого поиска по ID
        self.name = "BaseDataset"
        self.cache_dir = Path("cache")

    def _load_manifest(self):
        """Парсит JSONL файл один раз при инициализации"""
        seen_paths = set()
        self._id_map = {}
        
        if not hasattr(self, 'manifest_path') or not self.manifest_path.exists():
            return
            
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if self.limit and len(self._samples) >= self.limit:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    rel_path = data.get('audio_filepath', '')
                    
                    if not rel_path:
                        continue
                        
                    # Дедупликация
                    if rel_path in seen_paths:
                        continue
                    seen_paths.add(rel_path)

                    full_path = self.root_dir / rel_path
                    
                    text = data.pop('text', '')
                    duration = float(data.pop('duration', 0.0))
                    data.pop('audio_filepath', None)  
                    
                    sample = AudioSample(
                        id=full_path.name,
                        audio_path=str(full_path),
                        text=text,
                        duration=duration,
                        meta=data
                    )
                    self._samples.append(sample)
                    self._id_map[sample.id] = sample
                    
                except json.JSONDecodeError:
                    continue

        logger.info("[%s] Loaded %s samples from %s", self.name, len(self._samples), self.manifest_path)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> AudioSample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[AudioSample]:
        return iter(self._samples)

    def _ensure_not_empty(self, action: str) -> None:
        if self._samples:
            return
        raise ValueError(
            f"Dataset '{self.name}' is empty. Cannot {action}. "
            "Load data first or check your manifest/root_dir configuration."
        )
    
    # ===== Functional API (Transformations) =====

    def clone(self) -> 'BaseASRDataset':
        """
        Создает поверхностную копию датасета (View).
        Список сэмплов копируется, но сами объекты AudioSample остаются общими.
        Это позволяет фильтровать датасет, обрабатывать часть, и видеть результаты в оригинале.
        """
        new_ds = copy.copy(self)
        new_ds._samples = list(self._samples) # Копия списка ссылок
        return new_ds

    def filter(self, predicate: Callable[[AudioSample], bool]) -> 'BaseASRDataset':
        """
        Фильтрует датасет по условию. Возвращает новый объект датасета.
        Пример: ds.filter(lambda s: s.duration < 10)
        """
        new_ds = self.clone()
        new_ds._samples = [s for s in self._samples if predicate(s)]
        logger.info("Filtered dataset: %s -> %s samples", len(self), len(new_ds))
        return new_ds

    def sort(self, key: Callable[[AudioSample], Any], reverse: bool = False) -> 'BaseASRDataset':
        """
        Сортирует датасет. Возвращает новый объект датасета.
        Пример: ds.sort(key=lambda s: s.duration)
        """
        new_ds = self.clone()
        new_ds._samples.sort(key=key, reverse=reverse)
        return new_ds

    def take(self, n: int) -> 'BaseASRDataset':
        """
        Берет первые N элементов.
        """
        new_ds = self.clone()
        new_ds._samples = self._samples[:n]
        logger.info("Taking first %s samples", len(new_ds))
        return new_ds

    def random_split(self, ratio: float = 0.8, seed: int = 42) -> tuple['BaseASRDataset', 'BaseASRDataset']:
        """
        Разделяет датасет на два (train/test) случайным образом.
        """
        rng = random.Random(seed)
        indices = list(range(len(self._samples)))
        rng.shuffle(indices)
        
        split_idx = int(len(self._samples) * ratio)
        
        train_ds = self.clone()
        train_ds.name = f"{self.name}_train"
        train_ds._samples = [self._samples[i] for i in indices[:split_idx]]
        
        test_ds = self.clone()
        test_ds.name = f"{self.name}_test"
        test_ds._samples = [self._samples[i] for i in indices[split_idx:]]
        
        logger.info(
            "Random split: %s -> %s train + %s test",
            len(self),
            len(train_ds),
            len(test_ds),
        )
        return train_ds, test_ds

    def stratified_split(
        self, 
        ratio: float = 0.8, 
        by: Union[str, Callable] = 'duration', 
        buckets: int = 10,
        seed: int = 42
    ) -> tuple['BaseASRDataset', 'BaseASRDataset']:
        """
        Стратифицированное разбиение. Гарантирует сохранение распределения признака (например, длительности).
        
        Args:
            ratio: Доля train.
            by: Имя поля ('duration') или функция для извлечения значения.
            buckets: Количество корзин для непрерывных значений.
            seed: Seed.
        """
        rng = random.Random(seed)
        
        # 1. Извлекаем значения для стратификации
        values = []
        for s in self._samples:
            if isinstance(by, str):
                val = getattr(s, by, None)
            else:
                val = by(s)
            values.append(val)
            
        # 2. Определяем группы (bins)
        # Если значения числовые (float/int), делаем биннинг
        if all(isinstance(v, (int, float)) for v in values if v is not None):
            valid_values = [v for v in values if v is not None]
            if not valid_values:
                logger.warning("Stratification failed: no valid values. Falling back to random split.")
                return self.random_split(ratio, seed)
                
            min_v, max_v = min(valid_values), max(valid_values)
            # Защита от деления на ноль, если все значения одинаковые
            if max_v == min_v:
                groups = [0] * len(values)
            else:
                step = (max_v - min_v) / buckets
                groups = []
                for v in values:
                    if v is None:
                        groups.append(-1)
                    else:
                        bin_idx = min(int((v - min_v) / step), buckets - 1)
                        groups.append(bin_idx)
        else:
            # Категориальные значения (строки и т.д.) - используем как есть
            groups = values

        # 3. Группируем индексы
        from collections import defaultdict
        indices_by_group = defaultdict(list)
        for i, g in enumerate(groups):
            indices_by_group[g].append(i)
            
        # 4. Сплитим внутри каждой группы
        train_indices = []
        test_indices = []
        
        for g, indices in indices_by_group.items():
            rng.shuffle(indices)
            split_idx = int(len(indices) * ratio)
            
            # Гарантируем хотя бы 1 элемент в трейне, если группа не пуста (опционально)
            # if split_idx == 0 and len(indices) > 0: split_idx = 1 
            
            train_indices.extend(indices[:split_idx])
            test_indices.extend(indices[split_idx:])
            
        # Перемешиваем финальные списки, чтобы не шли блоками
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
        
        train_ds = self.clone()
        train_ds.name = f"{self.name}_train_stratified"
        train_ds._samples = [self._samples[i] for i in train_indices]
        
        test_ds = self.clone()
        test_ds.name = f"{self.name}_test_stratified"
        test_ds._samples = [self._samples[i] for i in test_indices]
        
        logger.info(
            "Stratified split by %s: %s -> %s train + %s test",
            by,
            len(self),
            len(train_ds),
            len(test_ds),
        )
        return train_ds, test_ds

    def __rshift__(self, other) -> Union['BaseASRDataset', tuple['BaseASRDataset', 'BaseASRDataset']]:
        """
        Магический оператор >> для построения пайплайнов.
        
        dataset >> Filter(...)    -> filter(...)
        dataset >> Sort(...)      -> sort(...)
        dataset >> Split(...)     -> stratified_split(...) -> (train, test)
        dataset >> model          -> apply(model)
        dataset >> "file.jsonl"   -> save_results("file.jsonl")
        """
        if isinstance(other, str):
            # Строка -> путь для сохранения
            self.save_results(other)
            return self
        
        # Поддержка функциональных глаголов (Filter, Sort, Take, Split)
        if hasattr(other, 'apply_to'):
            return other.apply_to(self)
            
        # Иначе -> процессор (модель или метрика)
        return self.apply(other)
    
    # ===== Интеграция с моделями и Кэширование (Private) =====

    def _get_cache_path(self, model_name: str) -> Path:
        """Возвращает путь к файлу кэша для конкретной модели"""
        # Структура: cache/{dataset_name}/{model_name}.jsonl
        d = self.cache_dir / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{model_name}.jsonl"

    def _load_cache_for_model(self, model_name: str):
        """Загружает уже существующие результаты из кэша"""
        cache_path = self._get_cache_path(model_name)
        if not cache_path.exists():
            return 0
        
        loaded = 0
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    sid = data.get('id')
                    if sid in self._id_map:
                        # Ищем результаты этой модели в загруженном объекте
                        # В кэше мы сохраняем полный объект AudioSample
                        cached_results = data.get('asr_results', {})
                        if model_name in cached_results:
                            res = cached_results[model_name]
                            self._id_map[sid].add_result(
                                model_name,
                                res.get('hypothesis', ''),
                                res.get('processing_time', 0.0),
                                res.get('error'),
                                metrics=res.get('metrics') # Загружаем метрики если есть
                            )
                            loaded += 1
                except json.JSONDecodeError:
                    continue
        return loaded

    # ===== Public API =====

    def apply(self, processors: Union[object, List[object]], **kwargs) -> 'BaseASRDataset':
        """
        Применяет один или несколько процессоров к датасету.

        Каждый процессор должен реализовывать интерфейс Processor (метод apply_to).
        Для передачи параметров (batch_size, save_step и т.д.) передавайте их
        непосредственно конструктору процессора или используйте >> оператор.

        Args:
            processors: Один процессор или список процессоров.
            **kwargs:   Устарело. Передавайте параметры в конструктор процессора.
        """
        if not isinstance(processors, list):
            processors = [processors]

        dataset = self
        for p in processors:
            if hasattr(p, 'apply_to'):
                result = p.apply_to(dataset)
                if result is not None:
                    dataset = result
            else:
                raise TypeError(
                    f"Processor {type(p).__name__} не реализует метод apply_to(). "
                    "Унаследуйтесь от plantain2asr.core.Processor."
                )

        return dataset

    def run_model(
        self,
        model: 'BaseASRModel',
        batch_size: Optional[int] = None,
        save_step: Optional[int] = None,
        force_process: bool = False,
    ) -> 'BaseASRDataset':
        """
        Публичный orchestration-метод для инференса одной модели.

        Нужен для верхнеуровневых experiment/scenario API, чтобы не обращаться
        к private `_apply_model()` напрямую.
        """
        self._ensure_not_empty(f"run model '{model.name}'")
        resolved_batch_size = batch_size if batch_size is not None else getattr(model, "batch_size", 32)
        resolved_save_step = save_step if save_step is not None else getattr(model, "save_step", 32)
        self._apply_model(
            model,
            batch_size=resolved_batch_size,
            save_step=resolved_save_step,
            force_process=force_process,
        )
        return self

    def evaluate_metric(
        self,
        evaluator: 'BaseMetric',
        force: bool = False,
    ) -> 'BaseASRDataset':
        """
        Публичный orchestration-метод для расчёта метрик.
        """
        self._ensure_not_empty(f"evaluate metric '{getattr(evaluator, 'name', type(evaluator).__name__)}'")
        self._apply_metric(evaluator, force=force)
        return self

    # ===== Internal Logic =====

    def _apply_model(
        self,
        model: 'BaseASRModel',
        batch_size: int = 32,
        save_step: int = 32,
        force_process: bool = False,
    ):
        """Применяет ОДНУ модель к датасету с кэшированием.

        Args:
            model:         ASR-модель.
            batch_size:    Количество сэмплов на один вызов model.process_samples.
            save_step:     Записывать в кэш каждые N сэмплов.
            force_process: Если True — удаляет старый кэш и пересчитывает всё заново.
                           Если False (по умолчанию) — дозаписывает только то, что не обработано.
        """
        self._ensure_not_empty(f"run model '{model.name}'")
        logger.info("Evaluating model %s", model.name)

        # 1. Опционально: чистим старые результаты
        if force_process:
            logger.info("Force mode enabled; clearing previous results for %s", model.name)
            cache_path = self._get_cache_path(model.name)
            if cache_path.exists():
                os.remove(cache_path)
                logger.info("Deleted cache file %s", cache_path)

            cleared = sum(
                1 for s in self._samples
                if s.asr_results.pop(model.name, None) is not None
            )
            if cleared:
                logger.info("Cleared %s in-memory results for %s", cleared, model.name)

        # 2. Всегда: подгружаем то, что уже есть в кэше
        loaded_count = self._load_cache_for_model(model.name)
        if loaded_count > 0:
            logger.info("Resumed %s samples from cache for %s", loaded_count, model.name)

        # 3. Определяем, что осталось обработать
        todo_samples = [s for s in self._samples if model.name not in s.asr_results]

        if not todo_samples:
            logger.info("All samples already processed for %s", model.name)
            return

        logger.info("Processing %s remaining samples for %s", len(todo_samples), model.name)

        cache_path = self._get_cache_path(model.name)

        # 4. Обрабатываем чанками, дозаписываем в кэш
        with open(cache_path, 'a', encoding='utf-8') as f_cache:
            for i in tqdm(range(0, len(todo_samples), save_step), desc=model.name):
                chunk = todo_samples[i : i + save_step]

                processed_chunk = []
                for j in range(0, len(chunk), batch_size):
                    sub_batch = chunk[j : j + batch_size]
                    model.process_samples(sub_batch, inplace=True)
                    processed_chunk.extend(sub_batch)

                for sample in processed_chunk:
                    f_cache.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
                f_cache.flush()

    def _apply_metric(self, evaluator: 'BaseMetric', force: bool = False):
        """Применяет метрику ко всем результатам.

        Если evaluator поддерживает calculate_batch_per_sample (CompositeMetric
        со стандартным набором) — использует батчевый режим jiwer:
        2 вызова на модель вместо N_samples × N_metrics.
        Иначе — стандартный per-sample режим.
        """
        found_models: set = set()
        for s in self._samples:
            found_models.update(s.asr_results.keys())
        model_names = list(found_models)

        metric_name = getattr(evaluator, 'name', 'Metric')
        use_batch   = (
            hasattr(evaluator, 'calculate_batch_per_sample')
            and getattr(evaluator, '_use_fast_path', False)
        )

        mode = "batch (fast)" if use_batch else "per-sample"
        self._ensure_not_empty(f"evaluate metric '{metric_name}'")
        logger.info("Calculating %s in %s mode for %s models", metric_name, mode, len(model_names))

        if use_batch:
            count = self._apply_metric_batch(evaluator, model_names, metric_name, force)
        else:
            count = self._apply_metric_per_sample(evaluator, model_names, metric_name, force)

        logger.info("Updated %s for %s entries", metric_name, count)

    def _apply_metric_batch(self, evaluator, model_names, metric_name, force):
        """Батчевый путь: 2 jiwer-вызова на модель."""
        count = 0
        for m_name in model_names:
            # Собираем сэмплы, для которых нужен расчёт
            todo = []
            for s in self._samples:
                if not s.text or m_name not in s.asr_results:
                    continue
                res = s.asr_results[m_name]
                if res.get('error'):
                    continue
                existing_metrics = res.get('metrics') or {}
                if not force and metric_name in existing_metrics:
                    continue
                todo.append(s)

            if not todo:
                continue

            refs = [s.text for s in todo]
            hyps = [s.asr_results[m_name].get('hypothesis', '') for s in todo]

            try:
                batch_results = evaluator.calculate_batch_per_sample(refs, hyps)
            except Exception as e:
                logger.warning("Batch metric calculation failed for %s: %s. Falling back to per-sample mode.", m_name, e)
                batch_results = None

            if batch_results is None:
                # Fallback: per-sample
                for s in tqdm(todo, desc=m_name):
                    res = s.asr_results[m_name]
                    if 'metrics' not in res:
                        res['metrics'] = {}
                    try:
                        r = evaluator.calculate(s.text, res.get('hypothesis', ''))
                        res['metrics'].update(r if isinstance(r, dict) else {metric_name: r})
                        count += 1
                    except Exception as e:
                        logger.warning("Metric calculation failed for sample %s: %s", s.id, e)
                continue

            for s, metrics_dict in zip(tqdm(todo, desc=m_name), batch_results):
                res = s.asr_results[m_name]
                if 'metrics' not in res:
                    res['metrics'] = {}
                res['metrics'].update(metrics_dict)
                count += 1

        return count

    def _apply_metric_per_sample(self, evaluator, model_names, metric_name, force):
        """Стандартный per-sample путь."""
        count = 0
        for sample in tqdm(self._samples, desc=f"Evaluating {metric_name}"):
            if not sample.text:
                continue
            for m_name in model_names:
                if m_name not in sample.asr_results:
                    continue
                res = sample.asr_results[m_name]
                if res.get('error'):
                    continue
                existing_metrics = res.get('metrics') or {}
                if not force and metric_name in existing_metrics:
                    continue
                if 'metrics' not in res:
                    res['metrics'] = {}
                try:
                    result = evaluator.calculate(sample.text, res.get('hypothesis', ''))
                    if isinstance(result, dict):
                        res['metrics'].update(result)
                    else:
                        res['metrics'][metric_name] = result
                    count += 1
                except Exception as e:
                    logger.warning("Metric calculation failed for sample %s: %s", sample.id, e)
        return count

    
    # ===== Сохранение, Загрузка и Экспорт =====
    
    def save_results(self, output_path: str):
        """Сохраняет текущее состояние (Unified format)"""
        self.save_unified_results(output_path)

    def save_unified_results(self, output_path: str):
        """Сохраняет полную историю результатов в один файл."""
        dataset_io.save_unified_results(self, output_path)
                
    def save_legacy_results(self, output_path: str, model_name: str):
        """
        Сохраняет результаты КОНКРЕТНОЙ модели в старом плоском формате (для обратной совместимости).
        """
        dataset_io.save_legacy_results(self, output_path, model_name)

    def load_model_results(self, model_name: str, jsonl_path: str) -> int:
        """
        Загружает предрасчитанные результаты инференса из JSONL-файла.

        Используется для импорта результатов, полученных на другой машине
        (скриптом ``scripts/run_inference.py`` или любым другим инструментом).

        Каждая строка файла должна содержать минимум::

            {"audio_path": "/any/path/file.wav", "hypothesis": "текст"}

        Опционально: ``processing_time`` или ``time`` (секунды инференса).

        Сэмплы сопоставляются по **имени файла** (basename), поэтому
        абсолютный путь может отличаться от записанного в JSONL.

        Args:
            model_name: Логическое имя модели (ключ в ``sample.asr_results``).
            jsonl_path: Путь к JSONL-файлу с результатами.

        Returns:
            Число сэмплов, для которых найдено совпадение.

        Example::

            ds = GolosDataset("data/golos", subset="crowd")
            ds.load_model_results(
                "GigaAM-v3-rnnt",
                "plantain2asr/asr_data/golos/crowd/GigaAM-v3-rnnt_results.jsonl",
            )
        """
        return dataset_io.load_model_results(self, model_name, jsonl_path)

    def load_results(self, paths: Union[str, List[str]]):
        """
        Загружает результаты из JSONL файлов (поддерживает оба формата).
        """
        dataset_io.load_results(self, paths)

    def to_pandas(self):
        """
        Экспортирует датасет в pandas DataFrame для анализа.
        """
        return dataset_io.to_pandas(self)

    def iter_results_rows(self) -> List[Dict[str, Any]]:
        """
        Возвращает плоские записи вида one-row-per-(sample, model).

        Это базовый экспортный формат для CSV/Excel/summary и исследовательских
        надстроек.
        """
        return dataset_io.iter_results_rows(self)

    def save_csv(self, output_path: str) -> str:
        """
        Сохраняет плоский исследовательский экспорт в CSV без зависимости от pandas.
        """
        return dataset_io.save_csv(self, output_path)

    def save_excel(self, output_path: str) -> str:
        """
        Сохраняет плоский исследовательский экспорт в Excel.
        Требует pandas и один из Excel backends.
        """
        return dataset_io.save_excel(self, output_path)

    def summarize_by_model(self, metrics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Агрегирует результаты по моделям в лёгкий tabular-friendly вид.
        """
        return dataset_io.summarize_by_model(self, metrics=metrics)
