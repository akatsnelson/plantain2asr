from abc import ABC
import json
import os
import copy
import logging
import math
import random
from typing import Iterator, List, TYPE_CHECKING, Union, Optional, Callable, Any, Tuple
from torch.utils.data import Dataset
from pathlib import Path
from .types import AudioSample

# Пробуем импортировать tqdm
try:
    from tqdm.auto import tqdm
except ImportError:
    print("⚠️ tqdm not installed. Install for progress bars: pip install tqdm")
    def tqdm(iterable, **kwargs): return iterable

if TYPE_CHECKING:
    from ..models.base import BaseASRModel
    from ..metrics.base import BaseMetric

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

        print(f"[{self.name}] Loaded {len(self._samples)} samples from {self.manifest_path}")
        
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> AudioSample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[AudioSample]:
        return iter(self._samples)
    
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
        print(f"✂️ Filtered: {len(self)} -> {len(new_ds)} samples")
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
        print(f"✂️ Take: {len(new_ds)} samples")
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
        
        print(f"✂️ Random Split: {len(self)} -> {len(train_ds)} (train) + {len(test_ds)} (test)")
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
                print("⚠️ Stratification failed: no valid values. Fallback to random split.")
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
        
        print(f"✂️ Stratified Split (by {by}): {len(self)} -> {len(train_ds)} (train) + {len(test_ds)} (test)")
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
        Универсальный метод для применения Моделей (ASR) или Метрик.
        Паттерн Visitor / Pipeline.
        
        Args:
            processors: Одна модель/метрика или список.
            **kwargs: Аргументы (batch_size, save_step, force, force_process).
        """
        if not isinstance(processors, list):
            processors = [processors]
            
        print(f"🚀 Applying {len(processors)} processors...")

        for p in processors:
            # Duck typing check
            is_model = hasattr(p, 'transcribe') or hasattr(p, 'process_samples')
            is_metric = hasattr(p, 'calculate') and hasattr(p, 'name') and not is_model
            
            if is_model:
                valid_args = ['batch_size', 'save_step', 'force_process']
                model_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
                self._apply_model(p, **model_kwargs)
                
            elif is_metric:
                force = kwargs.get('force', True)
                self._apply_metric(p, force=force)
            else:
                print(f"⚠️ Unknown processor type: {type(p)}. Skipping.")
                
        return self # Fluent API support

    # ===== Internal Logic =====

    def _apply_model(
        self, 
        model: 'BaseASRModel', 
        batch_size: int = 32,
        save_step: int = 32,
        force_process: bool = False
    ):
        """Применяет ОДНУ модель к датасету."""
        print(f"\nEvaluating {model.name}...")
        
        # Если форсируем - чистим всё старое
        if force_process:
            print(f"   🔄 Force processing enabled: clearing previous results for {model.name}")
            cache_path = self._get_cache_path(model.name)
            if cache_path.exists():
                os.remove(cache_path)
                print(f"      Deleted cache file: {cache_path}")
            
            # Чистим результаты в памяти
            cleared_count = 0
            for s in self._samples:
                if model.name in s.asr_results:
                    del s.asr_results[model.name]
                    cleared_count += 1
            if cleared_count:
                print(f"      Cleared {cleared_count} in-memory results")
            
            # 1. Загружаем кэш (восстановление)
            loaded_count = self._load_cache_for_model(model.name)
            if loaded_count > 0:
                print(f"   ⏩ Resumed {loaded_count} samples from cache")
                
            # 2. Определяем, что осталось обработать
            todo_samples = [s for s in self._samples if model.name not in s.asr_results]
            
            if not todo_samples:
                print(f"   ✅ All samples processed for {model.name}")
            return
                
            print(f"   🔥 Processing {len(todo_samples)} remaining samples...")
            
            cache_path = self._get_cache_path(model.name)
            
            # 3. Обрабатываем чанками
            with open(cache_path, 'a', encoding='utf-8') as f_cache:
                
                for i in tqdm(range(0, len(todo_samples), save_step), desc=f"{model.name}"):
                    chunk = todo_samples[i : i + save_step]
                    
                    # Саб-батчинг для модели
                    processed_chunk = []
                    for j in range(0, len(chunk), batch_size):
                        sub_batch = chunk[j : j + batch_size]
                        model.process_samples(sub_batch, inplace=True)
                        processed_chunk.extend(sub_batch)
                    
                # Сохраняем полный сэмпл
                    for sample in processed_chunk:
                        f_cache.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
                    f_cache.flush()

    def _apply_metric(self, evaluator: 'BaseMetric', force: bool = False):
        """Применяет ОДНУ метрику ко всем результатам."""
        
        # Ищем все доступные модели в сэмплах
        found_models = set()
        for s in self._samples:
            found_models.update(s.asr_results.keys())
        model_names = list(found_models)
            
        metric_name = getattr(evaluator, 'name', 'Metric')
        print(f"📊 Calculating {metric_name} for {len(model_names)} models (force={force})...")
        
        count = 0
        for sample in tqdm(self._samples, desc=f"Evaluating {metric_name}"):
            if not sample.text:
                continue
                
            for m_name in model_names:
                if m_name in sample.asr_results:
                    res = sample.asr_results[m_name]
                    
                    # Если метрик нет ИЛИ мы форсируем пересчет
                    if force or 'metrics' not in res or metric_name not in res.get('metrics', {}):
                        hyp = res.get('hypothesis', '')
                        if hyp and not res.get('error'):
                            if 'metrics' not in res:
                                res['metrics'] = {}
                                
                            try:
                                result = evaluator.calculate(sample.text, hyp)
                                
                                # Корректная обработка результатов метрик
                                if isinstance(result, dict):
                                    new_metrics = result
                                else:
                                    new_metrics = {metric_name: result}
                                    
                                res['metrics'].update(new_metrics)
                                count += 1
                            except Exception as e:
                                print(f"Error computing metrics for {sample.id}: {e}")
                            
        print(f"✅ Updated {metric_name} for {count} entries")

    
    # ===== Сохранение, Загрузка и Экспорт =====
    
    def save_results(self, output_path: str):
        """Сохраняет текущее состояние (Unified format)"""
        self.save_unified_results(output_path)

    def save_unified_results(self, output_path: str):
        """Сохраняет полную историю результатов в один файл."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving unified results to {output_path}...")
        with open(path, 'w', encoding='utf-8') as f:
            for sample in self._samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
                
    def save_legacy_results(self, output_path: str, model_name: str):
        """
        Сохраняет результаты КОНКРЕТНОЙ модели в старом плоском формате (для обратной совместимости).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving legacy results ({model_name}) to {output_path}...")
        
        saved_count = 0
        with open(path, 'w', encoding='utf-8') as f:
            for sample in self._samples:
                if model_name not in sample.asr_results:
                    continue
                    
                res = sample.asr_results[model_name]
                
                data = {
                    "id": sample.id,
                    "audio_path": sample.audio_path,
                    "reference": sample.text,
                    "duration": sample.duration,
                    **sample.meta,
                    "hypothesis": res.get("hypothesis", ""),
                    "time": res.get("processing_time", 0.0),
                    "model": model_name
                }
                
                if res.get("error"):
                    data["error"] = res["error"]
                    
                # Добавляем основные метрики в плоский вид, если есть
                if 'metrics' in res:
                    for k, v in res['metrics'].items():
                        data[k] = v
                        
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                saved_count += 1
        print(f"   Saved {saved_count} lines")

    def load_results(self, paths: Union[str, List[str]]):
        """
        Загружает результаты из JSONL файлов (поддерживает оба формата).
        """
        if isinstance(paths, str):
            paths = [paths]
            
        for path in paths:
            path = Path(path)
            if not path.exists():
                print(f"⚠️ File not found: {path}")
                continue
                
            print(f"📖 Loading results from {path.name}...")
            loaded_count = 0
            
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        sample_id = data.get('id')
                        
                        if sample_id in self._id_map:
                            sample = self._id_map[sample_id]
                            
                            # 1. Проверяем новый формат (asr_results)
                            if 'asr_results' in data:
                                sample.asr_results.update(data['asr_results'])
                                loaded_count += 1
                                continue
                                
                            # 2. Проверяем старый unified формат (results)
                            if 'results' in data:
                                sample.asr_results.update(data['results'])
                                loaded_count += 1
                                continue
                                
                            # 3. Проверяем легаси (плоский)
                            model_name = data.get('name', data.get('model'))
                            if not model_name and 'model_info' in data:
                                model_name = data['model_info'].get('name')
                            
                            if model_name:
                                hyp = data.get('hypothesis', '')
                                time = data.get('processing_time', 0.0)
                                err = data.get('error')
                                # Пытаемся вытащить метрики из плоского файла
                                metrics = {}
                                for k in ['wer', 'cer', 'mer', 'wil', 'wip', 'accuracy']:
                                    if k in data:
                                        metrics[k] = data[k]
                                
                                sample.add_result(model_name, hyp, time, err, metrics=metrics if metrics else None)
                                loaded_count += 1
                            
                    except json.JSONDecodeError:
                        continue
            
            print(f"   Matched {loaded_count}/{len(self)} samples")

    def to_pandas(self):
        """
        Экспортирует датасет в pandas DataFrame для анализа.
        """
        try:
            import pandas as pd
        except ImportError:
            print("⚠️ Pandas not installed. Install it via 'pip install pandas'")
            return None
        
        flat_data = []
        for s in self._samples:
            # Базовая инфа о сэмпле
            base_info = {
                "id": s.id,
                "duration": s.duration,
                "reference": s.text,
                **s.meta
            }
            
            # Разворачиваем результаты моделей
            for model_name, res in s.asr_results.items():
                row = base_info.copy()
                row["model"] = model_name
                row["hypothesis"] = res.get("hypothesis")
                row["processing_time"] = res.get("processing_time")
                row["error"] = res.get("error")
                
                # Добавляем метрики как отдельные колонки
                if "metrics" in res and res["metrics"]:
                    for m_name, m_val in res["metrics"].items():
                        row[f"{m_name}"] = m_val
                
                flat_data.append(row)
        
        return pd.DataFrame(flat_data)
