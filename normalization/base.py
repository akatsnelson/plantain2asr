"""
Базовый класс нормализации текста.

Нормализатор — полноправный участник пайплайна.
Его можно применять к данным через >> так же как модели и метрики:

    dataset >> DagrusNormalizer() >> Metrics.create_composite() >> DiffVisualizer(...)

При применении к датасету нормализатор создаёт нормализованный VIEW:
    - sample.text            → normalize_ref(text)
    - asr_results[m][hypothesis] → normalize_hyp(hypothesis)

Оригинальный датасет не мутируется.
Разделение ref/hyp нужно потому, что референс может содержать аннотации
(DagRus: щас{сейчас*}), а гипотеза — нет.
"""

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

from ..core.processor import Processor

if TYPE_CHECKING:
    from ..dataloaders.base import BaseASRDataset


class BaseNormalizer(Processor):
    """
    Абстрактный нормализатор текста.

    Подклассы реализуют normalize_ref и normalize_hyp.
    По умолчанию normalize_hyp делегирует в normalize_ref (симметричный случай).
    """

    @abstractmethod
    def normalize_ref(self, text: str) -> str:
        """Нормализует референсный (эталонный) текст."""
        pass

    def normalize_hyp(self, text: str) -> str:
        """
        Нормализует гипотезу ASR-модели.
        По умолчанию — то же, что normalize_ref (симметричная нормализация).
        Переопределите, если ref и hyp требуют разной обработки.
        """
        return self.normalize_ref(text)

    def normalize_pair(self, ref: str, hyp: str) -> Tuple[str, str]:
        """Нормализует пару (ref, hyp) за один вызов."""
        return self.normalize_ref(ref), self.normalize_hyp(hyp)

    # ------------------------------------------------------------------
    # Pipeline integration: dataset >> normalizer
    # ------------------------------------------------------------------

    def apply_to(self, dataset: 'BaseASRDataset') -> 'BaseASRDataset':
        """
        Применяет нормализацию к датасету и возвращает нормализованный VIEW.

        Трогает два поля:
            sample.text                          → normalize_ref
            asr_results[model]['hypothesis']     → normalize_hyp

        Оригинальный датасет не изменяется.
        """
        new_ds = dataset.clone()
        new_ds._samples = []
        for s in dataset._samples:
            ns = copy.copy(s)
            ns.text = self.normalize_ref(s.text or "")
            ns.asr_results = {}
            for model_name, res in s.asr_results.items():
                # copy.copy даёт shallow dict, но вложенный "metrics" будет shared.
                # Копируем его отдельно, чтобы расчёт метрик на norm_ds
                # не мутировал оригинальный dataset.
                nr = dict(res)
                if "metrics" in nr:
                    nr["metrics"] = dict(nr["metrics"])   # независимая копия
                if "hypothesis" in nr:
                    nr["hypothesis"] = self.normalize_hyp(nr["hypothesis"] or "")
                ns.asr_results[model_name] = nr
            new_ds._samples.append(ns)

        new_ds._id_map = {s.id: s for s in new_ds._samples}
        print(f"✅ Normalized {len(new_ds)} samples ({type(self).__name__})")
        return new_ds
