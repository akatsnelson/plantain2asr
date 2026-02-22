"""
ReportBuilder — собирает данные отчёта, делегируя каждой секции compute().

До рефакторинга: монолитный класс с private-методами (_build_metrics, _build_errors, _build_diff).
После:           тонкий координатор — просто итерирует по секциям.

Добавить новую секцию = написать класс секции. ReportBuilder менять не нужно.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..dataloaders.base import BaseASRDataset
    from .sections.base import BaseSection


class ReportBuilder:
    """
    Собирает данные для всех секций и сохраняет их в JSON-файлы.

    Args:
        dataset:  Датасет с ASR-результатами (уже нормализованный, если нужно).
        sections: Список секций. Если не передан — MetricsSection + ErrorFrequencySection + DiffSection.

    Нормализация применяется к датасету ДО создания builder:
        norm_ds = dataset >> DagrusNormalizer()
        builder = ReportBuilder(norm_ds)
    """

    def __init__(
        self,
        dataset: 'BaseASRDataset',
        sections: Optional[List['BaseSection']] = None,
    ):
        self.dataset   = dataset
        self.sections  = sections or _default_sections()
        self._cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> dict:
        """
        Вычисляет данные всех секций.
        Результат кешируется; повторный вызов возвращает кеш.
        """
        if self._cache is not None:
            return self._cache

        result = {}
        for section in self.sections:
            print(f"  ⠿ [{section.icon}] Building {section.title}…", flush=True)
            try:
                result[section.name] = section.compute(self.dataset)
            except Exception as e:
                result[section.name] = {"_error": str(e)}

        self._cache = result
        return result

    def save(self, output_dir: str) -> None:
        """
        Сохраняет данные каждой секции в отдельный JSON-файл
        в папке output_dir/{name}.json.
        """
        data = self.build()
        os.makedirs(output_dir, exist_ok=True)
        for name, section_data in data.items():
            path = os.path.join(output_dir, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(section_data, f, ensure_ascii=False, indent=2)
            print(f"  → Saved {path}")


def _default_sections() -> List['BaseSection']:
    from .sections import MetricsSection, ErrorFrequencySection, DiffSection
    return [MetricsSection(), ErrorFrequencySection(), DiffSection()]
