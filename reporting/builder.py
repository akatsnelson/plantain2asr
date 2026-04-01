"""
ReportBuilder — собирает данные отчёта, делегируя каждой секции compute().

До рефакторинга: монолитный класс с private-методами (_build_metrics, _build_errors, _build_diff).
После:           тонкий координатор — просто итерирует по секциям.

Добавить новую секцию = написать класс секции. ReportBuilder менять не нужно.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from ..utils.logging import get_logger
from .template import build_html

if TYPE_CHECKING:
    from ..dataloaders.base import BaseASRDataset
    from .sections.base import BaseSection

logger = get_logger(__name__)


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
            logger.info("Building report section [%s] %s", section.icon, section.title)
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
            logger.info("Saved report section to %s", path)

    def save_static_html(self, output_path: str) -> str:
        """
        Save a self-contained static HTML report with precomputed section data.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = build_html(self.sections, initial_data=self.build())
        path.write_text(html, encoding="utf-8")
        logger.info("Saved static report HTML to %s", path)
        return str(path)


def _default_sections() -> List['BaseSection']:
    from .sections import default_sections

    return default_sections()
