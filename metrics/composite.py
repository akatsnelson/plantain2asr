from typing import List, Dict, Optional, TYPE_CHECKING
from .base import BaseMetric

if TYPE_CHECKING:
    from ..normalization.base import BaseNormalizer
    from ..dataloaders.base import BaseASRDataset

try:
    import jiwer as _jiwer
    _HAS_JIWER = True
except ImportError:
    _HAS_JIWER = False

# Стандартный набор классов (импортируем лениво, чтобы не тащить torch при любом импорте)
_DEFAULT_METRIC_CLASSES = None


def _default_metric_classes():
    global _DEFAULT_METRIC_CLASSES
    if _DEFAULT_METRIC_CLASSES is None:
        from .simple.wer import WER
        from .simple.cer import CER
        from .simple.mer import MER
        from .simple.wil import WIL
        from .simple.wip import WIP
        from .simple.accuracy import Accuracy
        from .simple.idr import IDR
        from .simple.length_ratio import LengthRatio
        _DEFAULT_METRIC_CLASSES = [WER, CER, MER, WIL, WIP, Accuracy, IDR, LengthRatio]
    return _DEFAULT_METRIC_CLASSES


# Имена метрик, которые вычисляются батчем из jiwer выравниваний.
# Если CompositeMetric содержит ТОЛЬКО эти метрики → используем быстрый путь.
_FAST_METRICS = frozenset({"WER", "CER", "MER", "WIL", "WIP", "Accuracy", "IDR", "LengthRatio"})


class CompositeMetric:
    """
    Агрегатор метрик: считает несколько метрик за один проход.

    Нормализация применяется на уровне датасета до расчёта:
        dataset >> DagrusNormalizer() >> Metrics.composite()

    Оптимизация: при стандартном наборе метрик использует батчевый режим jiwer
    (2 вызова на всю модель вместо N_samples × N_metrics вызовов).
    """

    name: str = "Composite"

    def __init__(
        self,
        metrics=None,
        normalizer: Optional['BaseNormalizer'] = None,
    ):
        """
        Args:
            metrics:    Список *классов* метрик. None → стандартный набор.
            normalizer: Нормализатор (предпочтительнее применять к данным через >>).
        """
        if metrics is None:
            metrics = _default_metric_classes()

        self.metrics: List[BaseMetric] = [m(normalizer=normalizer) for m in metrics]

        # Быстрый путь доступен, если все метрики из стандартного набора и jiwer есть
        metric_names = {m.name for m in self.metrics}
        self._use_fast_path = _HAS_JIWER and metric_names.issubset(_FAST_METRICS)

    # ------------------------------------------------------------------
    # Per-sample API (используется как fallback)
    # ------------------------------------------------------------------

    def calculate(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Считает все метрики для одной пары строк."""
        result = {}
        for m in self.metrics:
            val = m.calculate(reference, hypothesis)
            if isinstance(val, dict):
                result.update(val)
            else:
                result[m.name] = val
        return result

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """Считает усреднённые метрики для батча (для агрегации)."""
        result = {}
        for m in self.metrics:
            val = m.calculate_batch(references, hypotheses)
            if isinstance(val, dict):
                result.update(val)
            else:
                result[m.name] = val
        return result

    # ------------------------------------------------------------------
    # Батчевый per-sample API (быстрый путь)
    # ------------------------------------------------------------------

    def calculate_batch_per_sample(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> List[Dict[str, float]]:
        """
        Вычисляет метрики для каждой пары отдельно, используя батчевый jiwer.

        Вместо N×M вызовов jiwer — ровно 2:
            jiwer.process_words(refs, hyps)      → WER, MER, WIL, WIP, Accuracy, IDR, LR
            jiwer.process_characters(refs, hyps) → CER

        Returns:
            Список словарей метрик, по одному на каждую пару. Для пустого ref → пустой dict.
        """
        n = len(references)
        results: List[Dict] = [{} for _ in range(n)]

        # Индексы, где ref непустой
        valid_idx = [i for i, r in enumerate(references) if r.strip()]
        if not valid_idx:
            return results

        valid_refs = [references[i] for i in valid_idx]
        valid_hyps = [hypotheses[i] for i in valid_idx]

        # ── Словесные метрики ──────────────────────────────────────────
        need_words = self._metric_names & {"WER", "MER", "WIL", "WIP", "Accuracy", "IDR", "LengthRatio"}
        if need_words:
            try:
                w_out = _jiwer.process_words(valid_refs, valid_hyps)
                for j, i in enumerate(valid_idx):
                    results[i].update(
                        _word_metrics_from_alignment(
                            w_out.references[j],
                            w_out.hypotheses[j],
                            w_out.alignments[j],
                            need_words,
                        )
                    )
            except Exception as e:
                pass  # fallback handled in _apply_metric

        # ── CER ────────────────────────────────────────────────────────
        if "CER" in self._metric_names:
            try:
                c_out = _jiwer.process_characters(valid_refs, valid_hyps)
                for j, i in enumerate(valid_idx):
                    results[i]["CER"] = _cer_from_char_alignment(
                        c_out.references[j],
                        c_out.hypotheses[j],
                        c_out.alignments[j],
                    )
            except Exception:
                pass

        # ── Пустые ref → WER=100 если hyp непустой ────────────────────
        for i in range(n):
            if i not in valid_idx:
                hyp_empty = not hypotheses[i].strip()
                v = 0.0 if hyp_empty else 100.0
                results[i] = {k: v for k in self._metric_names}

        return results

    @property
    def _metric_names(self):
        names = set()
        for m in self.metrics:
            names.add(m.name)
        return names

    # ------------------------------------------------------------------
    # Pipeline integration
    # ------------------------------------------------------------------

    def apply_to(self, dataset: 'BaseASRDataset') -> 'BaseASRDataset':
        """Интеграция с pipeline >>."""
        dataset._apply_metric(self)
        return dataset


# ── Вспомогательные функции для батч-вычисления ───────────────────────────────

def _word_metrics_from_alignment(ref_words, hyp_words, chunks, needed: frozenset) -> dict:
    """Вычисляет словарные метрики из одного jiwer-выравнивания."""
    n_ref = len(ref_words)
    n_hyp = len(hyp_words)
    n_hit = n_del = n_ins = n_sub = 0

    for chunk in chunks:
        t = chunk.type
        if t == "equal":
            n_hit += chunk.ref_end_idx - chunk.ref_start_idx
        elif t == "delete":
            n_del += chunk.ref_end_idx - chunk.ref_start_idx
        elif t == "insert":
            n_ins += chunk.hyp_end_idx - chunk.hyp_start_idx
        elif t == "substitute":
            n_sub += chunk.ref_end_idx - chunk.ref_start_idx

    out = {}

    if "WER" in needed:
        out["WER"] = (n_del + n_ins + n_sub) / n_ref * 100 if n_ref else (100.0 if n_hyp else 0.0)

    denom_mer = n_ref + n_ins
    mer = (n_del + n_ins + n_sub) / denom_mer * 100 if denom_mer else 0.0
    if "MER" in needed:
        out["MER"] = mer
    if "Accuracy" in needed:
        out["Accuracy"] = 100.0 - mer

    if "WIP" in needed or "WIL" in needed:
        wip = (n_hit / n_ref) * (n_hit / n_hyp) * 100 if (n_ref and n_hyp) else 0.0
        if "WIP" in needed:
            out["WIP"] = wip
        if "WIL" in needed:
            out["WIL"] = 100.0 - wip

    if "IDR" in needed:
        out["Insertion"]    = n_ins / n_ref * 100 if n_ref else 0.0
        out["Deletion"]     = n_del / n_ref * 100 if n_ref else 0.0
        out["Substitution"] = n_sub / n_ref * 100 if n_ref else 0.0

    if "LengthRatio" in needed:
        out["LengthRatio"] = n_hyp / n_ref if n_ref else 0.0

    return out


def _cer_from_char_alignment(ref_chars, hyp_chars, chunks) -> float:
    """Вычисляет CER из символьного jiwer-выравнивания."""
    n_ref = len(ref_chars)
    n_hyp = len(hyp_chars)
    n_del = n_ins = n_sub = 0

    for chunk in chunks:
        t = chunk.type
        if t == "delete":
            n_del += chunk.ref_end_idx - chunk.ref_start_idx
        elif t == "insert":
            n_ins += chunk.hyp_end_idx - chunk.hyp_start_idx
        elif t == "substitute":
            n_sub += chunk.ref_end_idx - chunk.ref_start_idx

    return (n_del + n_ins + n_sub) / n_ref * 100 if n_ref else (100.0 if n_hyp else 0.0)
