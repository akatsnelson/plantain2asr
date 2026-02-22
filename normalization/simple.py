"""
SimpleNormalizer — базовая нормализация текста.

Применяется к большинству ASR-моделей и датасетов:
lowercase + е=ё + удаление технических маркеров + удаление пунктуации.

Это то, что раньше делал do_clean=True в BaseMetric.
"""

import re
from .base import BaseNormalizer


class SimpleNormalizer(BaseNormalizer):
    """
    Универсальная нормализация:
        1. Lowercase
        2. ё → е (большинство ASR-моделей не различают)
        3. Удаление технических маркеров (#нрзб#, *говорит на другом языке* и т.д.)
        4. Удаление пунктуации (оставляем буквы, цифры, пробелы)
        5. Схлопывание пробелов
    """

    def normalize_ref(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = text.replace('ё', 'е')
        text = re.sub(r'#.*?#', '', text)
        text = re.sub(r'\*.*?\*', '', text)
        text = re.sub(r'[^\wа-яa-z\d\s]', ' ', text, flags=re.UNICODE)
        return re.sub(r'\s+', ' ', text).strip()
