"""
Анализатор структуры и богатства корпуса.

Считает по reference-текстам датасета:

Объём:
    total_tokens        — общее число слов
    total_utterances    — число высказываний (сэмплов с текстом)
    vocab_forms         — уникальных словоформ
    vocab_lemmas        — уникальных лемм

Структура высказываний:
    mean_utt_len        — средняя длина высказывания (слов)
    std_utt_len         — СКО длины высказывания
    mean_word_len       — средняя длина слова (букв)

Лексическое богатство:
    ttr                 — Type-Token Ratio (V/N); зависит от N, только для справки
    msttr               — Mean Segmental TTR; нейтрализует зависимость от N
    hapax_ratio         — доля hapax legomena от словаря
    hapax_token_ratio   — hapax / N
    dis_legomena_ratio  — доля dis legomena
    yule_k              — Yule's K (↓ = богаче)
    herdan_c            — Herdan's C = log(V)/log(N) (↑ = богаче)
    inflection_ratio    — словоформ / лемм (морфологическая сложность)
    gini                — коэффициент Джини частот слов (↑ = концентрированнее)
    zipf_slope          — наклон прямой Ципфа в log-log пространстве

Кривая покрытия:
    coverage_at         — {50: k, 80: k, 90: k, 95: k} — сколько лемм покрывают X% токенов

Лингвистические маркеры:
    ne_density          — доля именованных сущностей (собств. имена, топонимы, фамилии)
    colloquial_ratio    — доля разговорных/диалектных форм из COLLOQUIAL_MAP

POS-распределение:
    pos_counts          — Counter по тегам pymorphy3

Пример:
    from plantain2asr import CorpusStatsAnalyzer

    comp = CorpusStatsAnalyzer.compare([golos_crowd, rudevices, dagrus])
    comp.print_comparison()
    df = comp.to_pandas()
    comp.coverage_df()     # кривая покрытия
    comp.zipf_plot()       # график Ципфа
"""

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

try:
    import pymorphy3 as _morphy
except ImportError:
    try:
        import pymorphy2 as _morphy
    except ImportError:
        _morphy = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

from ..dataloaders.base import BaseASRDataset
from ..core.processor import Processor

# ── POS справочники ──────────────────────────────────────────────────────────

POS_NAMES = {
    'NOUN': 'Существительное', 'ADJF': 'Прилагательное',
    'ADJS': 'Прилагательное (кр.)', 'COMP': 'Компаратив',
    'VERB': 'Глагол', 'INFN': 'Инфинитив',
    'PRTF': 'Причастие', 'PRTS': 'Причастие (кр.)', 'GRND': 'Деепричастие',
    'NUMR': 'Числительное', 'ADVB': 'Наречие', 'NPRO': 'Местоимение',
    'PRED': 'Предикатив', 'PREP': 'Предлог', 'CONJ': 'Союз',
    'PRCL': 'Частица', 'INTJ': 'Междометие',
}

POS_GROUPS = {
    'Именная группа':    {'NOUN', 'ADJF', 'ADJS', 'PRTF', 'PRTS', 'NPRO'},
    'Глагольная группа': {'VERB', 'INFN', 'GRND'},
    'Служебные слова':   {'PREP', 'CONJ', 'PRCL'},
    'Наречия':           {'ADVB', 'COMP', 'PRED'},
    'Числительные':      {'NUMR'},
    'Междометия':        {'INTJ'},
}

# Теги именованных сущностей в pymorphy3
NE_TAGS: Set[str] = {'Name', 'Patr', 'Surn', 'Geox', 'Orgn'}


def _clean_token(word: str) -> str:
    """Нижний регистр + ё→е + убираем знаки препинания."""
    return re.sub(r'[^\w]', '', word.lower().replace('ё', 'е').strip())


# ── Вычисление отдельных метрик ──────────────────────────────────────────────

def _msttr(tokens: List[str], seg_size: int = 100) -> float:
    """Mean Segmental TTR."""
    if len(tokens) < seg_size:
        return len(set(tokens)) / len(tokens) if tokens else 0.0
    ttrs = [
        len(set(tokens[i: i + seg_size])) / seg_size
        for i in range(0, len(tokens) - seg_size + 1, seg_size)
    ]
    return sum(ttrs) / len(ttrs) if ttrs else 0.0


def _yule_k(form_counts: Counter) -> float:
    """Yule's K = 10^4 * (Σ r²·V(r) − N) / N²."""
    n = sum(form_counts.values())
    if n == 0:
        return 0.0
    freq_of_freq: Counter = Counter(form_counts.values())
    sigma = sum(r * r * v for r, v in freq_of_freq.items())
    return 1e4 * (sigma - n) / (n * n) if n > 1 else 0.0


def _herdan_c(v: int, n: int) -> float:
    if n <= 1 or v <= 1:
        return 0.0
    return math.log(v) / math.log(n)


def _gini(counts: Counter) -> float:
    """Коэффициент Джини распределения частот слов."""
    vals = sorted(counts.values())
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total == 0:
        return 0.0
    cumsum = 0.0
    for i, v in enumerate(vals, 1):
        cumsum += v * (2 * i - n - 1)
    return cumsum / (n * total)


def _zipf_slope(form_counts: Counter) -> float:
    """
    Наклон log(частота) ~ log(ранг) методом МНК.
    Для идеального закона Ципфа ≈ −1.
    Крутой наклон (< −1) → «длинный хвост», богатый словарь.
    """
    if not NUMPY_AVAILABLE or len(form_counts) < 5:
        return 0.0
    freqs = sorted(form_counts.values(), reverse=True)
    log_ranks = np.log(np.arange(1, len(freqs) + 1, dtype=float))
    log_freqs = np.log(np.array(freqs, dtype=float) + 1e-9)
    slope = np.polyfit(log_ranks, log_freqs, 1)[0]
    return float(slope)


def _coverage_at(lemma_counts: Counter, thresholds=(50, 80, 90, 95)) -> Dict[int, int]:
    """
    Сколько лемм (по убыванию частоты) нужно, чтобы покрыть X% токенов.
    Возвращает {50: k, 80: k, 90: k, 95: k}.
    """
    total = sum(lemma_counts.values())
    if total == 0:
        return {t: 0 for t in thresholds}
    result: Dict[int, int] = {}
    cumulative = 0
    for i, (_, cnt) in enumerate(lemma_counts.most_common(), 1):
        cumulative += cnt
        pct = cumulative / total * 100
        for t in list(thresholds):
            if t not in result and pct >= t:
                result[t] = i
    for t in thresholds:
        result.setdefault(t, len(lemma_counts))
    return result


# ── Dataclass отчёта ─────────────────────────────────────────────────────────

@dataclass
class CorpusReport:
    """Полный отчёт о структуре одного корпуса."""
    dataset_name: str

    # Объём
    total_tokens: int = 0
    total_utterances: int = 0
    vocab_forms: int = 0
    vocab_lemmas: int = 0

    # Структура высказываний
    mean_utt_len: float = 0.0
    std_utt_len: float = 0.0
    median_utt_len: float = 0.0
    mean_word_len: float = 0.0

    # Лексическое богатство
    ttr: float = 0.0
    msttr: float = 0.0
    hapax_count: int = 0
    hapax_ratio: float = 0.0
    hapax_token_ratio: float = 0.0
    dis_legomena_count: int = 0
    dis_legomena_ratio: float = 0.0
    yule_k: float = 0.0
    herdan_c: float = 0.0
    inflection_ratio: float = 0.0   # vocab_forms / vocab_lemmas
    gini: float = 0.0
    zipf_slope: float = 0.0

    # Кривая покрытия
    coverage_at: Dict[int, int] = field(default_factory=dict)

    # Лингвистические маркеры
    ne_density: float = 0.0
    colloquial_ratio: float = 0.0

    # POS
    pos_counts: Dict[str, int] = field(default_factory=dict)

    # Топ слов
    _top_forms: List[Tuple[str, int]] = field(default_factory=list, repr=False)
    _top_lemmas: List[Tuple[str, int]] = field(default_factory=list, repr=False)
    # Полные счётчики для Zipf-графика
    _form_counts: Dict[str, int] = field(default_factory=dict, repr=False)

    def print(self):
        w = 56
        sep = "=" * w
        print(f"\n{sep}")
        print(f"  Корпус: {self.dataset_name}")
        print(sep)
        print(f"  {'Высказываний':<32} {self.total_utterances:>10,}")
        print(f"  {'Токенов (N)':<32} {self.total_tokens:>10,}")
        print(f"  {'Уникальных словоформ':<32} {self.vocab_forms:>10,}")
        print(f"  {'Уникальных лемм':<32} {self.vocab_lemmas:>10,}")
        print(f"  {'Флективность (формы/леммы)':<32} {self.inflection_ratio:>10.2f}")
        print()
        print(f"  {'Ср. длина высказывания':<32} {self.mean_utt_len:>10.1f}  слов  (σ={self.std_utt_len:.1f})")
        print(f"  {'Ср. длина слова':<32} {self.mean_word_len:>10.2f}  букв")
        print()
        print(f"  {'MSTTR (богатство)':<32} {self.msttr:>10.4f}  ↑ выше = богаче")
        print(f"  {'TTR (V/N, зависит от N)':<32} {self.ttr:>10.4f}")
        yule_label = "Yule's K"
        herdan_label = "Herdan's C"
        print(f"  {yule_label:<32} {self.yule_k:>10.2f}  ↓ ниже = богаче")
        print(f"  {herdan_label:<32} {self.herdan_c:>10.4f}  ↑ выше = богаче")
        print(f"  {'Gini частот':<32} {self.gini:>10.4f}  ↑ выше = концентрированнее")
        print(f"  {'Наклон Ципфа':<32} {self.zipf_slope:>10.3f}")
        print()
        print(f"  {'Hapax legomena':<32} {self.hapax_count:>10,}  ({self.hapax_ratio*100:.1f}% словаря)")
        print(f"  {'Hapax / N':<32} {self.hapax_token_ratio:>10.4f}")
        print(f"  {'Dis legomena':<32} {self.dis_legomena_count:>10,}  ({self.dis_legomena_ratio*100:.1f}% словаря)")
        print()
        if self.coverage_at:
            print(f"  {'Кривая покрытия (лемм для X% токенов)':}")
            for pct, k in sorted(self.coverage_at.items()):
                share = k / self.vocab_lemmas * 100 if self.vocab_lemmas else 0
                print(f"    {pct:>3}% токенов → {k:>7,} лемм  ({share:.1f}% словаря)")
        print()
        if self.ne_density:
            print(f"  {'Именованные сущности':<32} {self.ne_density*100:>9.1f}%")
        if self.colloquial_ratio:
            print(f"  {'Разговорные формы':<32} {self.colloquial_ratio*100:>9.1f}%")
        if self.pos_counts:
            total = self.total_tokens or 1
            print()
            print("  Распределение по частям речи:")
            for pos, cnt in sorted(self.pos_counts.items(), key=lambda x: -x[1]):
                name = POS_NAMES.get(pos, pos)
                pct = cnt / total * 100
                bar = '█' * int(pct / 2)
                print(f"    {name:<26} {cnt:>8,}  {pct:5.1f}%  {bar}")
        print(sep + "\n")


# ── Класс сравнения ──────────────────────────────────────────────────────────

class CorpusComparison:
    """Результат сравнительного анализа нескольких корпусов."""

    def __init__(self, reports: List[CorpusReport]):
        self.reports = reports

    def print_comparison(self):
        names = [r.dataset_name for r in self.reports]
        col = max(len(n) for n in names)

        def _row(label, vals, fmt='{:.4f}'):
            parts = '  '.join(fmt.format(v).rjust(col) for v in vals)
            print(f"  {label:<34} {parts}")

        header = '  '.join(n.rjust(col) for n in names)
        w = 36 + (col + 2) * len(names)
        print("\n" + "=" * w)
        print("  Сравнение корпусов")
        print("=" * w)
        print(f"  {'Метрика':<34} {header}")
        print("-" * w)
        _row("Высказываний",               [r.total_utterances for r in self.reports], '{:,}')
        _row("Токенов (N)",                [r.total_tokens for r in self.reports], '{:,}')
        _row("Уникальных словоформ",       [r.vocab_forms for r in self.reports], '{:,}')
        _row("Уникальных лемм",            [r.vocab_lemmas for r in self.reports], '{:,}')
        _row("Флективность (формы/леммы)", [r.inflection_ratio for r in self.reports], '{:.2f}')
        print()
        _row("Ср. длина высказывания",     [r.mean_utt_len for r in self.reports], '{:.1f}')
        _row("Ср. длина слова (букв)",     [r.mean_word_len for r in self.reports], '{:.2f}')
        print()
        _row("MSTTR ↑ богаче",             [r.msttr for r in self.reports])
        _row("TTR (V/N)",                  [r.ttr for r in self.reports])
        _row("Yule's K ↓ богаче",          [r.yule_k for r in self.reports], '{:.1f}')
        _row("Herdan's C ↑ богаче",        [r.herdan_c for r in self.reports])
        _row("Gini частот ↑ концентр.",    [r.gini for r in self.reports])
        _row("Наклон Ципфа",               [r.zipf_slope for r in self.reports], '{:.3f}')
        print()
        _row("Hapax / словарь (%)",        [r.hapax_ratio * 100 for r in self.reports], '{:.1f}')
        _row("Hapax / N",                  [r.hapax_token_ratio for r in self.reports])
        _row("Dis legomena / словарь (%)", [r.dis_legomena_ratio * 100 for r in self.reports], '{:.1f}')
        print()
        _row("Именованные сущности (%)",   [r.ne_density * 100 for r in self.reports], '{:.1f}')
        _row("Разговорные формы (%)",       [r.colloquial_ratio * 100 for r in self.reports], '{:.1f}')
        print("=" * w + "\n")

    def to_pandas(self) -> 'pd.DataFrame':
        """DataFrame: строки — метрики, столбцы — датасеты."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas не установлен")
        metrics = {
            "Высказываний":                  lambda r: r.total_utterances,
            "Токенов (N)":                   lambda r: r.total_tokens,
            "Уникальных словоформ":          lambda r: r.vocab_forms,
            "Уникальных лемм":               lambda r: r.vocab_lemmas,
            "Флективность (формы/леммы)":    lambda r: round(r.inflection_ratio, 2),
            "Ср. длина высказыв. (слов)":    lambda r: round(r.mean_utt_len, 2),
            "Ср. длина слова (букв)":        lambda r: round(r.mean_word_len, 2),
            "TTR":                           lambda r: round(r.ttr, 4),
            "MSTTR":                         lambda r: round(r.msttr, 4),
            "Yule's K":                      lambda r: round(r.yule_k, 2),
            "Herdan's C":                    lambda r: round(r.herdan_c, 4),
            "Gini частот":                   lambda r: round(r.gini, 4),
            "Наклон Ципфа":                  lambda r: round(r.zipf_slope, 3),
            "Hapax legomena":                lambda r: r.hapax_count,
            "Hapax / словарь (%)":           lambda r: round(r.hapax_ratio * 100, 2),
            "Hapax / N":                     lambda r: round(r.hapax_token_ratio, 4),
            "Dis legomena":                  lambda r: r.dis_legomena_count,
            "Dis legomena / словарь (%)":    lambda r: round(r.dis_legomena_ratio * 100, 2),
            "Именованные сущности (%)":      lambda r: round(r.ne_density * 100, 2),
            "Разговорные формы (%)":          lambda r: round(r.colloquial_ratio * 100, 2),
        }
        # Добавляем кривую покрытия
        all_thresholds = sorted({t for r in self.reports for t in r.coverage_at})
        for t in all_thresholds:
            metrics[f"Покрытие {t}% (лемм)"] = lambda r, _t=t: r.coverage_at.get(_t, 0)

        data = {m: [fn(r) for r in self.reports] for m, fn in metrics.items()}
        return pd.DataFrame(data, index=[r.dataset_name for r in self.reports]).T

    def pos_df(self) -> 'pd.DataFrame':
        """POS-распределение: строки — POS, столбцы — датасеты (доля %)."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas не установлен")
        all_pos = sorted({pos for r in self.reports for pos in r.pos_counts})
        rows = []
        for pos in all_pos:
            row = {"POS": pos, "Название": POS_NAMES.get(pos, pos)}
            for r in self.reports:
                total = r.total_tokens or 1
                row[r.dataset_name] = round(r.pos_counts.get(pos, 0) / total * 100, 2)
            rows.append(row)
        df = pd.DataFrame(rows).set_index("POS")
        df = df.sort_values(self.reports[0].dataset_name, ascending=False)
        return df

    def coverage_df(self) -> 'pd.DataFrame':
        """
        DataFrame с кривой покрытия.

        Строки: пороги покрытия (50, 80, 90, 95%).
        Значения: сколько лемм нужно.
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas не установлен")
        all_thresholds = sorted({t for r in self.reports for t in r.coverage_at})
        rows = []
        for t in all_thresholds:
            row = {"Покрытие (%)": t}
            for r in self.reports:
                k = r.coverage_at.get(t, 0)
                row[r.dataset_name] = k
                row[f"{r.dataset_name} (% словаря)"] = round(
                    k / r.vocab_lemmas * 100, 1
                ) if r.vocab_lemmas else 0
            rows.append(row)
        return pd.DataFrame(rows).set_index("Покрытие (%)")

    def top_words_df(self, kind: str = "lemmas", n: int = 30) -> 'pd.DataFrame':
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas не установлен")
        dfs = []
        for r in self.reports:
            src = r._top_lemmas if kind == "lemmas" else r._top_forms
            total = r.total_tokens or 1
            df = pd.DataFrame(src[:n], columns=["слово", "кол-во"])
            df["%"] = (df["кол-во"] / total * 100).round(3)
            df.index = range(1, len(df) + 1)
            df.columns = pd.MultiIndex.from_tuples(
                [(r.dataset_name, c) for c in df.columns]
            )
            dfs.append(df)
        return pd.concat(dfs, axis=1)

    def zipf_plot(self, ax=None, top_n: int = 5000):
        """
        График Ципфа: log(ранг) × log(частота) для каждого корпуса.

        Args:
            ax:    matplotlib Axes. Если None — создаётся новая фигура.
            top_n: Сколько самых частых слов включить.
        """
        if not MPL_AVAILABLE:
            raise ImportError("matplotlib не установлен")
        show = ax is None
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))
        for r in self.reports:
            freqs = sorted(r._form_counts.values(), reverse=True)[:top_n]
            if not freqs:
                continue
            if NUMPY_AVAILABLE:
                ranks = np.arange(1, len(freqs) + 1)
                # freqs из Counter: все >= 1, log10 безопасен
                ax.plot(np.log10(ranks), np.log10(np.array(freqs, dtype=float)),
                        label=r.dataset_name, linewidth=1.5)
            else:
                ax.plot(
                    [math.log10(i + 1) for i in range(len(freqs))],
                    [math.log10(f) for f in freqs],
                    label=r.dataset_name, linewidth=1.5,
                )
        ax.set_xlabel("log₁₀(ранг)")
        ax.set_ylabel("log₁₀(частота)")
        ax.set_title("Закон Ципфа по корпусам")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if show:
            plt.tight_layout()
            plt.show()
        return ax


# ── Основной класс ───────────────────────────────────────────────────────────

class CorpusStatsAnalyzer(Processor):
    """
    Анализирует структуру и лексическое богатство корпуса(ов).

    Одиночный датасет через pipeline:
        ds >> CorpusStatsAnalyzer()

    Сравнение нескольких корпусов (рекомендуется):
        comp = CorpusStatsAnalyzer.compare([golos_crowd, rudevices, dagrus])
        comp.print_comparison()
        df   = comp.to_pandas()
        cov  = comp.coverage_df()
        comp.zipf_plot()
    """

    def __init__(
        self,
        top_n: int = 50,
        msttr_seg: int = 100,
        coverage_thresholds: Tuple = (50, 80, 90, 95),
        colloquial_words: Optional[Set[str]] = None,
    ):
        """
        Args:
            top_n:               Число топ-слов в отчёте.
            msttr_seg:           Размер сегмента для MSTTR.
            coverage_thresholds: Пороги кривой покрытия (%).
            colloquial_words:    Множество разговорных/диалектных форм.
                                 По умолчанию загружается из DagrusNormalizer.COLLOQUIAL_MAP.
        """
        if _morphy is None:
            raise ImportError("Для CorpusStatsAnalyzer нужен pymorphy3: pip install pymorphy3")
        self.morph = _morphy.MorphAnalyzer()
        self.top_n = top_n
        self.msttr_seg = msttr_seg
        self.coverage_thresholds = coverage_thresholds

        if colloquial_words is None:
            try:
                from ..normalization.dagrus import COLLOQUIAL_MAP
                # Берём ключи (разговорные формы), исключаем пустые (нрзб)
                colloquial_words = {k for k, v in COLLOQUIAL_MAP.items() if v}
            except ImportError:
                colloquial_words = set()
        self.colloquial_words: Set[str] = colloquial_words

        self.report: Optional[CorpusReport] = None

    @classmethod
    def compare(
        cls,
        datasets: List[BaseASRDataset],
        top_n: int = 50,
        msttr_seg: int = 100,
        coverage_thresholds: Tuple = (50, 80, 90, 95),
        colloquial_words: Optional[Set[str]] = None,
    ) -> 'CorpusComparison':
        """
        Анализирует каждый датасет отдельно и возвращает объект сравнения.

        Пример:
            comp = CorpusStatsAnalyzer.compare([golos_crowd, rudevices, dagrus])
            comp.print_comparison()
            comp.to_pandas()
        """
        analyzer = cls(
            top_n=top_n,
            msttr_seg=msttr_seg,
            coverage_thresholds=coverage_thresholds,
            colloquial_words=colloquial_words,
        )
        reports = []
        for ds in datasets:
            report = analyzer._build_report_for(ds)
            report.print()
            reports.append(report)
        return CorpusComparison(reports)

    def apply_to(self, dataset: BaseASRDataset) -> BaseASRDataset:
        self.report = self._build_report_for(dataset)
        self.report.print()
        return dataset

    def analyze(self, dataset: BaseASRDataset) -> 'CorpusStatsAnalyzer':
        self.report = self._build_report_for(dataset)
        return self

    def to_pandas(self) -> 'pd.DataFrame':
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas не установлен")
        if self.report is None:
            raise RuntimeError("Сначала запустите analyze() или примените через >>")
        total = self.report.total_tokens or 1
        rows = []
        for pos, cnt in sorted(self.report.pos_counts.items(), key=lambda x: -x[1]):
            rows.append({
                "POS": pos, "Название": POS_NAMES.get(pos, pos),
                "Кол-во": cnt, "Доля (%)": round(cnt / total * 100, 2),
            })
        return pd.DataFrame(rows)

    def top_words_df(self, kind: str = "lemmas", n: int = 50) -> 'pd.DataFrame':
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas не установлен")
        if self.report is None:
            raise RuntimeError("Сначала запустите analyze()")
        source = self.report._top_lemmas if kind == "lemmas" else self.report._top_forms
        total = self.report.total_tokens or 1
        return pd.DataFrame([
            {"Слово": w, "Кол-во": c, "Доля (%)": round(c / total * 100, 3)}
            for w, c in source[:n]
        ])

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_report_for(self, dataset: BaseASRDataset) -> CorpusReport:
        print(f"📚 Corpus stats: {dataset.name} ({len(dataset):,} samples)...")

        form_counts: Counter = Counter()
        lemma_counts: Counter = Counter()
        pos_counts: Counter = Counter()
        all_tokens: List[str] = []
        utt_lengths: List[int] = []
        word_lengths: List[int] = []
        ne_count = 0
        colloquial_count = 0

        for sample in tqdm(dataset, desc=dataset.name, leave=False):
            if not sample.text:
                continue
            words = sample.text.split()
            utt_len = 0
            for raw_word in words:
                word = _clean_token(raw_word)
                if not word:
                    continue

                parsed = self.morph.parse(word)
                if not parsed:
                    continue
                best = parsed[0]
                lemma = best.normal_form
                pos = best.tag.POS or "UNKN"
                grammemes = best.tag.grammemes

                form_counts[word] += 1
                lemma_counts[lemma] += 1
                pos_counts[pos] += 1
                all_tokens.append(word)
                word_lengths.append(len(word))
                utt_len += 1

                if grammemes & NE_TAGS:
                    ne_count += 1
                if word in self.colloquial_words:
                    colloquial_count += 1

            if utt_len > 0:
                utt_lengths.append(utt_len)

        n = len(all_tokens)
        v = len(form_counts)
        v_lemma = len(lemma_counts)

        hapax = sum(1 for c in form_counts.values() if c == 1)
        dis   = sum(1 for c in form_counts.values() if c == 2)

        return CorpusReport(
            dataset_name=dataset.name,
            total_tokens=n,
            total_utterances=len(utt_lengths),
            vocab_forms=v,
            vocab_lemmas=v_lemma,
            mean_utt_len=statistics.mean(utt_lengths) if utt_lengths else 0.0,
            std_utt_len=statistics.stdev(utt_lengths) if len(utt_lengths) > 1 else 0.0,
            median_utt_len=statistics.median(utt_lengths) if utt_lengths else 0.0,
            mean_word_len=statistics.mean(word_lengths) if word_lengths else 0.0,
            ttr=v / n if n else 0.0,
            msttr=_msttr(all_tokens, self.msttr_seg),
            hapax_count=hapax,
            hapax_ratio=hapax / v if v else 0.0,
            hapax_token_ratio=hapax / n if n else 0.0,
            dis_legomena_count=dis,
            dis_legomena_ratio=dis / v if v else 0.0,
            yule_k=_yule_k(form_counts),
            herdan_c=_herdan_c(v, n),
            inflection_ratio=v / v_lemma if v_lemma else 0.0,
            gini=_gini(form_counts),
            zipf_slope=_zipf_slope(form_counts),
            coverage_at=_coverage_at(lemma_counts, self.coverage_thresholds),
            ne_density=ne_count / n if n else 0.0,
            colloquial_ratio=colloquial_count / n if n else 0.0,
            pos_counts=dict(pos_counts),
            _top_forms=form_counts.most_common(self.top_n),
            _top_lemmas=lemma_counts.most_common(self.top_n),
            _form_counts=dict(form_counts),
        )
