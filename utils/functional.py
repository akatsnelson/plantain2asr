import random
from typing import Callable, Any, TYPE_CHECKING, Tuple, Union, Optional

if TYPE_CHECKING:
    from .dataloaders.base import BaseASRDataset

class Filter:
    """
    Функциональный объект для фильтрации датасета в пайплайне.
    Использование: dataset >> Filter(lambda s: s.duration < 10)
    """
    def __init__(self, predicate: Callable[[Any], bool]):
        self.predicate = predicate

    def apply_to(self, dataset: 'BaseASRDataset') -> 'BaseASRDataset':
        return dataset.filter(self.predicate)

class Sort:
    """
    Функциональный объект для сортировки датасета в пайплайне.
    Использование: dataset >> Sort(key=lambda s: s.duration)
    """
    def __init__(self, key: Callable[[Any], Any], reverse: bool = False):
        self.key = key
        self.reverse = reverse

    def apply_to(self, dataset: 'BaseASRDataset') -> 'BaseASRDataset':
        return dataset.sort(self.key, self.reverse)

class Take:
    """
    Функциональный объект для взятия первых N элементов.
    Использование: dataset >> Take(100)
    """
    def __init__(self, n: int):
        self.n = n

    def apply_to(self, dataset: 'BaseASRDataset') -> 'BaseASRDataset':
        return dataset.take(self.n)

class Split:
    """
    Функциональный объект для разделения датасета на train/test.
    Возвращает КОРТЕЖ из двух датасетов!
    
    Args:
        ratio: Доля первого датасета (train). Default: 0.8
        seed: Random seed. Default: 42
        stratify_by: Поле для стратификации. 
                     Может быть строкой ('duration') или функцией (lambda s: s.meta['gender']).
                     Если None - обычный random split.
        buckets: Количество корзин для стратификации непрерывных величин (duration). Default: 10
        
    Использование: 
        train, test = dataset >> Split(0.8)
        train, test = dataset >> Split(0.8, stratify_by='duration')
    """
    def __init__(
        self, 
        ratio: float = 0.8, 
        seed: int = 42, 
        stratify_by: Optional[Union[str, Callable]] = None,
        buckets: int = 10
    ):
        self.ratio = ratio
        self.seed = seed
        self.stratify_by = stratify_by
        self.buckets = buckets

    def apply_to(self, dataset: 'BaseASRDataset') -> Tuple['BaseASRDataset', 'BaseASRDataset']:
        if self.stratify_by:
            return dataset.stratified_split(self.ratio, self.stratify_by, self.buckets, self.seed)
        else:
            return dataset.random_split(self.ratio, self.seed)
