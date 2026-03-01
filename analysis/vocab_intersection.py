"""
Анализатор пересечения лексики между датасетами.

Сравнивает два (или более) корпуса по словарю:
- Общая лексика (intersection)
- Слова, уникальные для каждого корпуса
- OOV (Out-of-Vocabulary) rate: какая доля слов диалектного корпуса
  отсутствует в базовом
- Топ OOV-слов (кандидаты на специфичную диалектную лексику)
- Опционально: WER отдельно на OOV vs In-Vocabulary сэмплах
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

try:
    import pymorphy3 as _morphy
except ImportError:
    try:
        import pymorphy2 as _morphy
    except ImportError:
        _morphy = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

from ..dataloaders.base import BaseASRDataset


def _normalize_token(word: str) -> str:
    """Нижний регистр + ё→е + убираем знаки препинания."""
    return re.sub(r'[^\w]', '', word.lower().replace('ё', 'е').strip())


def _extract_vocab(
    dataset: BaseASRDataset,
    use_lemmas: bool = True,
    morph=None,
) -> Tuple[Counter, Set[str]]:
    """
    Извлекает словарь из reference-текстов датасета.

    Returns:
        (counter, vocab_set) — счётчик частот и множество форм/лемм.
    """
    counter: Counter = Counter()
    for sample in tqdm(dataset, desc=f"Vocab: {dataset.name}", leave=False):
        if not sample.text:
            continue
        for raw in sample.text.split():
            word = _normalize_token(raw)
            if not word:
                continue
            if use_lemmas and morph:
                parsed = morph.parse(word)
                token = parsed[0].normal_form if parsed else word
            else:
                token = word
            counter[token] += 1
    return counter, set(counter.keys())


@dataclass
class VocabIntersectionReport:
    """Результаты сравнения лексики двух корпусов."""
    base_name: str
    target_name: str
    use_lemmas: bool

    base_size: int
    target_size: int
    intersection_size: int
    base_only_size: int
    target_only_size: int

    target_oov_rate: float        # доля слов target, отсутствующих в base
    base_coverage: float          # какую долю target_vocab покрывает base

    top_target_oov: List[Tuple[str, int]] = field(default_factory=list)
    top_shared: List[Tuple[str, int]] = field(default_factory=list)

    # WER раздельно — заполняется опционально
    wer_oov: Optional[float] = None
    wer_iv: Optional[float] = None

    def print(self):
        kind = "лемм" if self.use_lemmas else "словоформ"
        print("\n" + "=" * 60)
        print("🔍 Vocabulary Intersection Report")
        print("=" * 60)
        print(f"Базовый корпус   : {self.base_name}   ({self.base_size:,} {kind})")
        print(f"Целевой корпус   : {self.target_name} ({self.target_size:,} {kind})")
        print("-" * 60)
        print(f"Общая лексика    : {self.intersection_size:,}")
        print(f"Только в базовом : {self.base_only_size:,}")
        print(f"Только в целевом : {self.target_only_size:,}  ← OOV")
        print(f"OOV rate         : {self.target_oov_rate:.1%}  "
              f"(слов целевого, не встречающихся в базовом)")
        print(f"Coverage         : {self.base_coverage:.1%}  "
              f"(слов целевого, покрытых базовым)")

        if self.wer_oov is not None and self.wer_iv is not None:
            print(f"\nWER (In-Vocabulary сэмплы) : {self.wer_iv:.1f}%")
            print(f"WER (OOV сэмплы)           : {self.wer_oov:.1f}%")

        if self.top_target_oov:
            print(f"\nТоп OOV-слов (уникальных для {self.target_name}):")
            for w, c in self.top_target_oov[:15]:
                print(f"  {w:<25} {c:>6,}")

        if self.top_shared:
            print(f"\nТоп общих слов:")
            for w, c in self.top_shared[:10]:
                print(f"  {w:<25} {c:>6,}")
        print("=" * 60 + "\n")


class VocabIntersectionAnalyzer:
    """
    Анализирует пересечение лексики между двумя датасетами.

    Пример использования:
        analyzer = VocabIntersectionAnalyzer()
        analyzer.analyze(golos_ds, dagrus_ds)
        analyzer.report.print()

        df_oov = analyzer.oov_df()  # DataFrame с OOV-словами
    """

    def __init__(self, use_lemmas: bool = True, top_n: int = 50):
        """
        Args:
            use_lemmas: Если True — сравниваем по леммам (рекомендуется),
                        иначе по словоформам.
            top_n:      Сколько топ-слов включать в отчёт/DataFrame.
        """
        if _morphy is None and use_lemmas:
            raise ImportError(
                "Для лемматизации нужен pymorphy3: pip install pymorphy3\n"
                "Либо используйте VocabIntersectionAnalyzer(use_lemmas=False)"
            )
        self.use_lemmas = use_lemmas
        self.top_n = top_n
        self.morph = _morphy.MorphAnalyzer() if _morphy and use_lemmas else None

        self.report: Optional[VocabIntersectionReport] = None

        # Сырые данные — для построения DataFrame
        self._base_counter: Counter = Counter()
        self._target_counter: Counter = Counter()
        self._base_vocab: Set[str] = set()
        self._target_vocab: Set[str] = set()

    def analyze(
        self,
        base_ds: BaseASRDataset,
        target_ds: BaseASRDataset,
        compute_wer: bool = True,
    ) -> 'VocabIntersectionAnalyzer':
        """
        Проводит анализ пересечения лексики.

        Args:
            base_ds:     Базовый датасет (например, GOLOS — эталонный русский).
            target_ds:   Целевой датасет (например, DaGRuS — диалект).
            compute_wer: Если True и у target_ds есть метрики, считает WER
                         раздельно на OOV vs In-Vocabulary сэмплах.
        """
        print(f"🔍 Vocab intersection: '{base_ds.name}' vs '{target_ds.name}'")

        self._base_counter, self._base_vocab = _extract_vocab(
            base_ds, self.use_lemmas, self.morph
        )
        self._target_counter, self._target_vocab = _extract_vocab(
            target_ds, self.use_lemmas, self.morph
        )

        intersection = self._base_vocab & self._target_vocab
        base_only = self._base_vocab - self._target_vocab
        target_only = self._target_vocab - self._base_vocab

        target_oov_rate = len(target_only) / len(self._target_vocab) if self._target_vocab else 0.0
        base_coverage = 1.0 - target_oov_rate

        # Топ OOV-слов по частоте в target
        top_oov = sorted(
            [(w, self._target_counter[w]) for w in target_only],
            key=lambda x: -x[1],
        )[:self.top_n]

        # Топ общих слов (по частоте в target)
        top_shared = sorted(
            [(w, self._target_counter[w]) for w in intersection],
            key=lambda x: -x[1],
        )[:self.top_n]

        # Опциональный расчёт WER по OOV vs IV сэмплам
        wer_oov, wer_iv = None, None
        if compute_wer:
            wer_oov, wer_iv = self._split_wer_by_oov(target_ds, target_only)

        self.report = VocabIntersectionReport(
            base_name=base_ds.name,
            target_name=target_ds.name,
            use_lemmas=self.use_lemmas,
            base_size=len(self._base_vocab),
            target_size=len(self._target_vocab),
            intersection_size=len(intersection),
            base_only_size=len(base_only),
            target_only_size=len(target_only),
            target_oov_rate=target_oov_rate,
            base_coverage=base_coverage,
            top_target_oov=top_oov,
            top_shared=top_shared,
            wer_oov=wer_oov,
            wer_iv=wer_iv,
        )
        self.report.print()
        return self

    def _split_wer_by_oov(
        self, dataset: BaseASRDataset, oov_set: Set[str]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Считает средний WER отдельно для сэмплов, содержащих OOV-слова,
        и для сэмплов, где все слова есть в базовом словаре.
        """
        oov_wers: List[float] = []
        iv_wers: List[float] = []

        for sample in dataset:
            if not sample.text or not sample.asr_results:
                continue

            words = {
                _normalize_token(w)
                for w in sample.text.split()
                if _normalize_token(w)
            }
            if self.use_lemmas and self.morph:
                lemmas = set()
                for w in words:
                    parsed = self.morph.parse(w)
                    lemmas.add(parsed[0].normal_form if parsed else w)
                words = lemmas

            has_oov = bool(words & oov_set)

            for model_name, res in sample.asr_results.items():
                wer = res.get('metrics', {}).get('WER')
                if wer is None:
                    wer = res.get('metrics', {}).get('wer')
                if wer is None:
                    continue
                if has_oov:
                    oov_wers.append(wer)
                else:
                    iv_wers.append(wer)

        wer_oov = (sum(oov_wers) / len(oov_wers)) if oov_wers else None
        wer_iv = (sum(iv_wers) / len(iv_wers)) if iv_wers else None
        return wer_oov, wer_iv

    def oov_df(self) -> 'pd.DataFrame':
        """
        DataFrame с OOV-словами (уникальными для целевого корпуса).

        Столбцы: Слово | Частота_target | Статус
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas не установлен: pip install pandas")
        if self.report is None:
            raise RuntimeError("Сначала запустите analyze()")

        rows = [
            {"Слово": w, "Частота": c, "Статус": "OOV"}
            for w, c in self.report.top_target_oov
        ]
        return pd.DataFrame(rows)

    def full_df(self) -> 'pd.DataFrame':
        """
        DataFrame по всей лексике целевого корпуса с пометкой статуса.

        Столбцы: Слово | Частота_target | Частота_base | Статус
        Статус: "shared" | "oov" | "base_only"
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas не установлен: pip install pandas")
        if self.report is None:
            raise RuntimeError("Сначала запустите analyze()")

        rows = []
        all_words = self._base_vocab | self._target_vocab
        for word in sorted(all_words):
            freq_t = self._target_counter.get(word, 0)
            freq_b = self._base_counter.get(word, 0)
            if freq_t > 0 and freq_b > 0:
                status = "shared"
            elif freq_t > 0:
                status = "oov"
            else:
                status = "base_only"
            rows.append({
                "Слово": word,
                "Частота_target": freq_t,
                "Частота_base": freq_b,
                "Статус": status,
            })

        df = pd.DataFrame(rows)
        return df.sort_values(["Статус", "Частота_target"], ascending=[True, False]).reset_index(drop=True)


# Публичный псевдоним для удобства импорта в ноутбуках
extract_vocab = _extract_vocab
