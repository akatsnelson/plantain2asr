"""
BaseSection — единый контракт для всех секций отчёта.

Каждая секция знает три вещи:
    1. Как собрать данные из датасета  → compute(dataset) → dict
    2. Как отрисовать себя в браузере  → js_function() → str (JS)
    3. Какой дополнительный HTML нужен → extra_html() → str

Добавить новую вкладку = написать один класс-наследник.
Трогать server.py, builder.py или template.py не нужно.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...dataloaders.base import BaseASRDataset


class BaseSection(ABC):
    """
    Базовый класс секции (вкладки) отчёта.

    Контракт:
        name        str   — уникальный id ("metrics", "errors", …)
        title       str   — заголовок вкладки
        icon        str   — эмодзи-иконка

        compute(dataset) → dict
            Вызывается один раз при построении отчёта.
            Результат сериализуется в JSON и отдаётся через /api/{name}.

        js_function() → str
            Строка JavaScript. ОБЯЗАНА определять функцию render_{name}().
            Функция имеет доступ к глобальному состоянию S и утилитам.

        extra_html() → str          (опционально)
            HTML-элементы вне основного <div id="{name}-panel">,
            например, сайдбары и модалки.

        css() → str                 (опционально)
            Дополнительный CSS специфичный для этой секции.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """URL-safe идентификатор секции (lowercase, без пробелов)."""

    @property
    @abstractmethod
    def title(self) -> str:
        """Заголовок вкладки в навигации."""

    @property
    @abstractmethod
    def icon(self) -> str:
        """Emoji-иконка вкладки."""

    @abstractmethod
    def compute(self, dataset: 'BaseASRDataset') -> dict:
        """
        Собирает данные секции из датасета.
        Возвращает JSON-сериализуемый словарь.
        Вызывается в фоне при старте сервера.
        """

    @abstractmethod
    def js_function(self) -> str:
        """
        Возвращает строку JavaScript, которая ОБЯЗАНА включать:
            function render_{self.name}() { ... }

        Функция имеет доступ к:
            S.data['{self.name}']  — данные от compute()
            S.activeModel          — текущая модель из селектора
            esc(s), fmtNum(v)      — утилиты из base template
        """

    def panel_html(self) -> str:
        """
        Начальный HTML внутри <div id="{name}-panel">.
        По умолчанию — спиннер, который заменяется при первой загрузке.
        Переопределите, если секции нужна особая внутренняя структура
        (например, двух-колоночный flex-лейаут с сайдбаром).
        """
        return '<div class="spinner-wrap"><div class="spinner"></div></div>'

    def css(self) -> str:
        """Дополнительный CSS для этой секции."""
        return ""
