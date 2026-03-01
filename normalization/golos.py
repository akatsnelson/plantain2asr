"""
GolosNormalizer — нормализатор для датасета GOLOS (и аналогичных).

Расширяет SimpleNormalizer одним важным шагом в normalize_hyp:
конвертирует арабские цифры в русские слова перед стандартной очисткой.

Проблема:
    GOLOS reference всегда написан словами:
        "эпизод тридцать один сезона тринадцать"
    Whisper-large-v3-ru-podlodka и аналогичные генеративные модели
    часто возвращают цифры:
        "Эпизод 31 сезона 13 фильма «Люди»."
    Без конвертации WER для Whisper искусственно завышается ~на 15.9% сэмплов.

Решение:
    normalize_hyp:
        1. Заменяем цифры → русские слова (num2words, lang='ru')
        2. Затем применяем стандартный SimpleNormalizer pipeline

    normalize_ref:
        Стандартный SimpleNormalizer (reference цифр не содержит).

Зависимость:
    pip install num2words
"""

import re
from .simple import SimpleNormalizer

try:
    from num2words import num2words as _num2words
    _NUM2WORDS_AVAILABLE = True
except ImportError:
    _NUM2WORDS_AVAILABLE = False


def _digits_to_words(text: str) -> str:
    """
    Заменяет все числовые последовательности в тексте на русские слова.

    Порядок замен важен:
    1. Числа с разделителями (телефоны, диапазоны): 866-66-300 → разбиваем по частям
    2. Числа с % или руб.: "70%" → "семьдесят процентов" (только токен, знак чистится позже)
    3. Обычные целые числа: "31" → "тридцать один"
    4. Десятичные дроби: "21.00" → обрабатываем как целое (часы)
    """
    if not _NUM2WORDS_AVAILABLE:
        return text

    def _convert(n: int) -> str:
        try:
            return _num2words(n, lang='ru')
        except Exception:
            return str(n)

    def _replace_number(m: re.Match) -> str:
        raw = m.group(0)
        # Убираем возможные знаки препинания вокруг (%, руб. и т.д. почистятся позже)
        digits = re.sub(r'[^\d]', '', raw)
        if not digits:
            return raw
        try:
            return _convert(int(digits))
        except (ValueError, OverflowError):
            return raw

    # 1. Числа-через-дефис (телефоны, индексы): заменяем каждую часть отдельно
    #    866-66-300-15-17  →  восемьсот шестьдесят шесть шестьдесят шесть...
    def _replace_phone(m: re.Match) -> str:
        parts = re.split(r'[-–]', m.group(0))
        words = []
        for p in parts:
            p = p.strip()
            if p.isdigit():
                try:
                    words.append(_convert(int(p)))
                except (ValueError, OverflowError):
                    words.append(p)
            else:
                words.append(p)
        return ' '.join(words)

    text = re.sub(r'\d[\d\-–]+\d', _replace_phone, text)

    # 2. Числа с точкой / двоеточием (время: 21.00, 21:00) → берём только часы
    text = re.sub(r'\b(\d{1,2})[.:]\d{2}\b', lambda m: _convert(int(m.group(1))), text)

    # 3. Все оставшиеся числа (включая с % и т.д.)
    text = re.sub(r'\d+', _replace_number, text)

    return text


class GolosNormalizer(SimpleNormalizer):
    """
    Нормализатор для GOLOS и аналогичных датасетов с чистым русским.

    Отличается от SimpleNormalizer только normalize_hyp:
    перед стандартной очисткой конвертирует цифры → русские слова,
    чтобы выровнять формат с reference (который всегда пишется словами).

    Если библиотека num2words не установлена — работает как SimpleNormalizer
    и выводит предупреждение.

    Пример использования:
        from plantain2asr.normalization.golos import GolosNormalizer

        norm = GolosNormalizer()
        golos_crowd_n = golos_crowd >> norm
        golos_crowd_n >> Metrics.composite()
    """

    def __init__(self):
        if not _NUM2WORDS_AVAILABLE:
            print(
                "⚠️  GolosNormalizer: num2words не установлен.\n"
                "   Конвертация цифр→слова отключена, WER для Whisper может быть завышен.\n"
                "   Установите: pip install num2words"
            )

    def normalize_hyp(self, text: str) -> str:
        """
        Гипотеза: сначала цифры → слова, потом стандартная очистка.
        """
        if not text:
            return ""
        text = _digits_to_words(text)
        return super().normalize_ref(text)  # lowercase + ё→е + пунктуация
