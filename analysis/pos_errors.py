"""
Анализ ошибок ASR по частям речи (POS Error Analysis).

Для каждого слова из reference определяет POS-тег (pymorphy3) и фиксирует тип события:
- correct      — слово распознано верно
- deletion     — слово пропущено моделью
- substitution — слово заменено другим

Insertion'ы не привязаны к конкретному слову reference, поэтому учитываются
как отдельная агрегированная метрика.

Основной вопрос для статьи:
    Какие части речи модели ошибаются чаще всего на DaGRuS,
    и отличается ли это от стандартного русского (GOLOS)?

Гипотезы:
    - NOUN (существительные) и собственные имена (Name) — главные жертвы:
      диалектные существительные уходят в OOV или заменяются.
    - VERB (глаголы) — устойчивее: фонетика менее диалектно-специфична.
    - PREP/CONJ (служебные) — почти не страдают: короткие частотные слова.

Пример:
    from plantain2asr import PosErrorAnalyzer

    analyzer = PosErrorAnalyzer(model_name='Whisper-ru')
    dagrus_n >> analyzer
    analyzer.report.print()

    df = analyzer.report.to_pandas()   # строки — POS, столбцы — метрики
    comp = PosErrorAnalyzer.compare([golos_crowd_n, dagrus_n], model_name='Whisper-ru')
"""

import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import pymorphy3 as _morphy
except ImportError:
    try:
        import pymorphy2 as _morphy
    except ImportError:
        _morphy = None

try:
    import jiwer as _jiwer
except ImportError:
    _jiwer = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kwargs): return it

from ..dataloaders.base import BaseASRDataset
from ..core.processor import Processor

# ── POS справочник (совпадает с corpus_stats) ────────────────────────────────

POS_NAMES = {
    'NOUN': 'Сущ.',        'ADJF': 'Прил.',         'ADJS': 'Прил.(кр.)',
    'COMP': 'Компар.',     'VERB': 'Глагол',         'INFN': 'Инфинитив',
    'PRTF': 'Причастие',   'PRTS': 'Прич.(кр.)',     'GRND': 'Деепр.',
    'NUMR': 'Числит.',     'ADVB': 'Наречие',         'NPRO': 'Местоим.',
    'PRED': 'Предикатив',  'PREP': 'Предлог',         'CONJ': 'Союз',
    'PRCL': 'Частица',     'INTJ': 'Межд.',           'UNKN': 'Неизв.',
}

POS_GROUPS = {
    'Именная':    {'NOUN', 'ADJF', 'ADJS', 'PRTF', 'PRTS', 'NPRO'},
    'Глагольная': {'VERB', 'INFN', 'GRND'},
    'Служебная':  {'PREP', 'CONJ', 'PRCL'},
    'Наречная':   {'ADVB', 'COMP', 'PRED'},
    'Числит.':    {'NUMR'},
    'Прочее':     {'INTJ', 'UNKN'},
}


def _clean(word: str) -> str:
    return re.sub(r'[^\w]', '', word.lower().replace('ё', 'е').strip())


# ── Отчёт ────────────────────────────────────────────────────────────────────

@dataclass
class PosErrorReport:
    """Результаты POS-анализа ошибок для одного датасета × одной модели."""
    dataset_name: str
    model_name:   str

    # Счётчики событий по POS
    # {pos: {'total': N, 'correct': N, 'deletion': N, 'substitution': N}}
    pos_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Топ-слов по каждому типу ошибки внутри POS
    # {pos: Counter(word → count)}
    deletion_words:      Dict[str, Counter] = field(default_factory=dict)
    substitution_words:  Dict[str, Counter] = field(default_factory=dict)

    def error_rate(self, pos: str) -> float:
        s = self.pos_stats.get(pos, {})
        total = s.get('total', 0)
        if total == 0:
            return 0.0
        return (s.get('deletion', 0) + s.get('substitution', 0)) / total

    def to_pandas(self) -> 'pd.DataFrame':
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas не установлен")
        rows = []
        for pos, stats in sorted(self.pos_stats.items(),
                                  key=lambda x: -x[1].get('total', 0)):
            total = stats.get('total', 0)
            if total == 0:
                continue
            corr = stats.get('correct', 0)
            dele = stats.get('deletion', 0)
            sub  = stats.get('substitution', 0)
            err  = dele + sub
            rows.append({
                'POS':             pos,
                'Название':        POS_NAMES.get(pos, pos),
                'Слов':            total,
                'Верно':           corr,
                'Удаления':        dele,
                'Замены':          sub,
                'Ошибок':          err,
                'Верно %':         round(corr / total * 100, 1),
                'Удаления %':      round(dele / total * 100, 1),
                'Замены %':        round(sub  / total * 100, 1),
                'Error rate %':    round(err  / total * 100, 1),
            })
        return pd.DataFrame(rows)

    def print(self, top_n: int = 8):
        w = 70
        print(f"\n{'='*w}")
        print(f"  POS Error Analysis | {self.dataset_name} | {self.model_name}")
        print(f"{'='*w}")
        print(f"  {'POS':<6} {'Название':<16} {'Слов':>7} {'Верно%':>7} {'Del%':>6} {'Sub%':>6} {'Err%':>6}")
        print(f"  {'-'*60}")
        for pos, stats in sorted(
            self.pos_stats.items(),
            key=lambda x: -(x[1].get('deletion', 0) + x[1].get('substitution', 0))
        ):
            total = stats.get('total', 0)
            if total < 5:
                continue
            corr = stats.get('correct', 0)
            dele = stats.get('deletion', 0)
            sub  = stats.get('substitution', 0)
            err  = dele + sub
            print(
                f"  {pos:<6} {POS_NAMES.get(pos, pos):<16} "
                f"{total:>7,} {corr/total*100:>7.1f} "
                f"{dele/total*100:>6.1f} {sub/total*100:>6.1f} "
                f"{err/total*100:>6.1f}"
            )
        print(f"{'='*w}\n")


# ── Сравнение нескольких датасетов ───────────────────────────────────────────

class PosErrorComparison:
    """Сравнительный анализ POS-ошибок по нескольким датасетам."""

    def __init__(self, reports: List[PosErrorReport]):
        self.reports = reports

    def to_pandas(self) -> 'pd.DataFrame':
        """
        DataFrame: строки — POS, столбцы — (датасет, метрика).
        Метрики: Error rate %, Удаления %, Замены %.
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas не установлен")
        all_pos = sorted({pos for r in self.reports for pos in r.pos_stats})
        rows = []
        for pos in all_pos:
            row = {'POS': pos, 'Название': POS_NAMES.get(pos, pos)}
            for r in self.reports:
                stats = r.pos_stats.get(pos, {})
                total = stats.get('total', 0)
                dele  = stats.get('deletion', 0)
                sub   = stats.get('substitution', 0)
                err   = dele + sub
                prefix = r.dataset_name
                row[f'{prefix}_total']   = total
                row[f'{prefix}_err_%']   = round(err  / total * 100, 1) if total else 0.0
                row[f'{prefix}_del_%']   = round(dele / total * 100, 1) if total else 0.0
                row[f'{prefix}_sub_%']   = round(sub  / total * 100, 1) if total else 0.0
            rows.append(row)

        df = pd.DataFrame(rows).set_index('POS')
        # Убираем POS с нулевыми данными во всех датасетах
        err_cols = [c for c in df.columns if c.endswith('_err_%')]
        df = df[df[err_cols].sum(axis=1) > 0]
        return df

    def error_rate_df(self) -> 'pd.DataFrame':
        """
        Компактный DataFrame: строки — POS, столбцы — датасеты.
        Значение: Error rate %.
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas не установлен")
        all_pos = sorted({pos for r in self.reports for pos in r.pos_stats})
        data = {}
        for r in self.reports:
            col = {}
            for pos in all_pos:
                stats = r.pos_stats.get(pos, {})
                total = stats.get('total', 0)
                err   = stats.get('deletion', 0) + stats.get('substitution', 0)
                col[pos] = round(err / total * 100, 1) if total else 0.0
            data[r.dataset_name] = col
        df = pd.DataFrame(data, index=all_pos)
        # Сортируем по максимальному error rate
        df = df.loc[df.max(axis=1).sort_values(ascending=False).index]
        return df.loc[df.sum(axis=1) > 0]

    def delta_df(self, base_idx: int = 0) -> 'pd.DataFrame':
        """
        Δ error rate: каждый датасет минус базовый (index 0 по умолчанию).
        Положительное → хуже, отрицательное → лучше.
        """
        df = self.error_rate_df()
        base_col = df.columns[base_idx]
        for col in df.columns:
            if col != base_col:
                df[f'Δ {col}'] = (df[col] - df[base_col]).round(1)
        return df


# ── Основной анализатор ──────────────────────────────────────────────────────

class PosErrorAnalyzer(Processor):
    """
    Анализирует ошибки ASR по частям речи.

    Использует jiwer для выравнивания (ref ↔ hyp) и pymorphy3 для POS-тегирования.

    Пример:
        analyzer = PosErrorAnalyzer(model_name='GigaAM-v3-e2e_rnnt')
        dagrus_n >> analyzer
        analyzer.report.print()
        df = analyzer.report.to_pandas()

    Сравнение датасетов (одна строка):
        comp = PosErrorAnalyzer.compare(
            [golos_crowd_n, dagrus_n],
            model_name='GigaAM-v3-e2e_rnnt',
        )
        comp.error_rate_df()   # error rate % по POS × датасет
        comp.delta_df()        # прирост DaGRuS − GOLOS
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        min_pos_count: int = 10,
    ):
        """
        Args:
            model_name:    Имя модели из asr_results. None → первая найденная.
            min_pos_count: Минимальное число слов данного POS для включения в отчёт.
        """
        if _morphy is None:
            raise ImportError("Нужен pymorphy3: pip install pymorphy3")
        if _jiwer is None:
            raise ImportError("Нужен jiwer: pip install jiwer")

        self.morph         = _morphy.MorphAnalyzer()
        self.model_name    = model_name
        self.min_pos_count = min_pos_count
        self.report: Optional[PosErrorReport] = None

    @classmethod
    def compare(
        cls,
        datasets: List[BaseASRDataset],
        model_name: Optional[str] = None,
        min_pos_count: int = 10,
    ) -> PosErrorComparison:
        """
        Анализирует каждый датасет и возвращает объект сравнения.

        Пример:
            comp = PosErrorAnalyzer.compare([golos_crowd_n, dagrus_n], model_name='Whisper-ru')
            comp.error_rate_df()
            comp.delta_df()
        """
        analyzer = cls(model_name=model_name, min_pos_count=min_pos_count)
        reports = []
        for ds in datasets:
            analyzer._run(ds)
            reports.append(analyzer.report)
            analyzer.report.print()
        return PosErrorComparison(reports)

    def apply_to(self, dataset: BaseASRDataset) -> BaseASRDataset:
        self._run(dataset)
        self.report.print()
        return dataset

    def _get_pos(self, word: str) -> str:
        word = _clean(word)
        if not word:
            return 'UNKN'
        parsed = self.morph.parse(word)
        if not parsed:
            return 'UNKN'
        pos = parsed[0].tag.POS
        return pos if pos else 'UNKN'

    def _resolve_model(self, dataset: BaseASRDataset) -> Optional[str]:
        if self.model_name:
            return self.model_name
        for s in dataset:
            if s.asr_results:
                return next(iter(s.asr_results))
        return None

    def _run(self, dataset: BaseASRDataset):
        model = self._resolve_model(dataset)
        if not model:
            raise RuntimeError("В датасете нет результатов ASR")

        print(f"🏷  POS error analysis: [{dataset.name}] model={model}")

        pos_stats:           Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        deletion_words:      Dict[str, Counter]         = defaultdict(Counter)
        substitution_words:  Dict[str, Counter]         = defaultdict(Counter)

        for sample in tqdm(dataset, desc=dataset.name, leave=False):
            if not sample.text or model not in sample.asr_results:
                continue
            ref = sample.text
            hyp = sample.asr_results[model].get('hypothesis', '')
            if not ref:
                continue

            try:
                out = _jiwer.process_words(ref, hyp)
            except Exception:
                continue

            ref_tokens = out.references[0]
            hyp_tokens = out.hypotheses[0]

            for chunk in out.alignments[0]:
                t  = chunk.type
                ri = chunk.ref_start_idx
                hi = chunk.hyp_start_idx

                if t == 'equal':
                    ref_word = ref_tokens[ri]
                    pos = self._get_pos(ref_word)
                    pos_stats[pos]['total']   += 1
                    pos_stats[pos]['correct'] += 1

                elif t == 'delete':
                    ref_word = ref_tokens[ri]
                    pos = self._get_pos(ref_word)
                    pos_stats[pos]['total']    += 1
                    pos_stats[pos]['deletion'] += 1
                    deletion_words[pos][_clean(ref_word)] += 1

                elif t == 'substitute':
                    ref_word = ref_tokens[ri]
                    hyp_word = hyp_tokens[hi]
                    pos = self._get_pos(ref_word)
                    pos_stats[pos]['total']         += 1
                    pos_stats[pos]['substitution']  += 1
                    substitution_words[pos][_clean(ref_word)] += 1

        # Фильтруем редкие POS
        filtered_stats = {
            pos: dict(cnt)
            for pos, cnt in pos_stats.items()
            if cnt.get('total', 0) >= self.min_pos_count
        }

        self.report = PosErrorReport(
            dataset_name=dataset.name,
            model_name=model,
            pos_stats=filtered_stats,
            deletion_words={p: dict(c) for p, c in deletion_words.items()},
            substitution_words={p: dict(c) for p, c in substitution_words.items()},
        )
