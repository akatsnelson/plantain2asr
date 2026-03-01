from collections import Counter
from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass, field

try:
    import jiwer
except ImportError:
    jiwer = None

from ..dataloaders.base import BaseASRDataset
from ..core.processor import Processor


@dataclass
class ErrorReport:
    """Контейнер для отчета об ошибках."""
    top_deletions: pd.DataFrame
    top_insertions: pd.DataFrame
    top_substitutions: pd.DataFrame
    substitution_details: Dict[str, Counter]
    total_words: int
    total_errors: int
    # Суммарные счётчики для сравнения датасетов
    total_deletions: int = 0
    total_insertions: int = 0
    total_substitutions: int = 0

    @property
    def del_rate(self) -> float:
        """Доля удалений от всех токенов reference."""
        return self.total_deletions / self.total_words if self.total_words else 0.0

    @property
    def ins_rate(self) -> float:
        """Доля вставок от всех токенов reference."""
        return self.total_insertions / self.total_words if self.total_words else 0.0

    @property
    def sub_rate(self) -> float:
        """Доля замен от всех токенов reference."""
        return self.total_substitutions / self.total_words if self.total_words else 0.0

    def to_dict(self) -> dict:
        """Сводная строка для pandas DataFrame."""
        return {
            'total_words':     self.total_words,
            'total_errors':    self.total_errors,
            'deletions':       self.total_deletions,
            'insertions':      self.total_insertions,
            'substitutions':   self.total_substitutions,
            'del_rate_%':      round(self.del_rate * 100, 2),
            'ins_rate_%':      round(self.ins_rate * 100, 2),
            'sub_rate_%':      round(self.sub_rate * 100, 2),
            'wer_%':           round(self.total_errors / self.total_words * 100, 2) if self.total_words else 0.0,
        }

    def print(self):
        w = self.total_words or 1
        print(f"\n📊 Word Error Analysis Report")
        print(f"  Слов reference : {self.total_words:,}")
        print(f"  Ошибок всего   : {self.total_errors:,}  ({self.total_errors/w*100:.1f}% WER)")
        print(f"  Удаления       : {self.total_deletions:,}  ({self.del_rate*100:.1f}%)")
        print(f"  Вставки        : {self.total_insertions:,}  ({self.ins_rate*100:.1f}%)")
        print(f"  Замены         : {self.total_substitutions:,}  ({self.sub_rate*100:.1f}%)")
        print("-" * 40)

        print(f"\n📉 Top Missed Words (Deletions):")
        print(self.top_deletions.to_markdown(index=False))

        print(f"\n📈 Top Hallucinations (Insertions):")
        print(self.top_insertions.to_markdown(index=False))

        print(f"\n🔄 Top Confusions (Substitutions):")
        subs_df = self.top_substitutions.copy()
        subs_df['Examples (->)'] = subs_df['Reference'].apply(
            lambda w: ", ".join(
                f"{k} ({v})" for k, v in self.substitution_details.get(w, Counter()).most_common(3)
            )
        )
        print(subs_df[['Reference', 'Count', 'Examples (->)']].to_markdown(index=False))


class WordErrorAnalyzer(Processor):
    """
    Анализатор ошибок на уровне слов.

    Строит рейтинг самых частых ошибок (пропуски, вставки, замены).

    Нормализация применяется к датасету до анализа:
        dataset >> DagrusNormalizer() >> WordErrorAnalyzer()

    Пример:
        # Без нормализации (данные уже нормализованы)
        norm_ds = dataset >> DagrusNormalizer()
        norm_ds >> WordErrorAnalyzer(model_name="GigaAM-rnnt")

        # Или в одну строку:
        dataset >> DagrusNormalizer() >> WordErrorAnalyzer()
    """

    def __init__(self, model_name: Optional[str] = None, top_n: int = 15):
        """
        Args:
            model_name: Имя модели из asr_results. None → берётся первая найденная.
            top_n:      Количество топ-слов в отчёте.
        """
        self.model_name = model_name
        self.top_n = top_n

        if jiwer is None:
            print("⚠️ 'jiwer' not installed: pip install jiwer")

    def apply_to(self, dataset: BaseASRDataset) -> BaseASRDataset:
        if jiwer is None:
            print("⚠️ 'jiwer' not installed: pip install jiwer")
            return dataset

        target_model = self.model_name
        if not target_model:
            for s in dataset:
                if s.asr_results:
                    target_model = next(iter(s.asr_results.keys()))
                    break

        if not target_model:
            print("⚠️ No model results found in dataset.")
            return dataset

        print(f"🔍 Analyzing word errors for model: {target_model}...")

        deletions:        Counter = Counter()
        insertions:       Counter = Counter()
        substitutions:    Counter = Counter()
        substitution_map: dict    = {}

        total_words  = 0
        total_errors = 0
        total_del = total_ins = total_sub = 0

        for sample in dataset:
            if not sample.text or target_model not in sample.asr_results:
                continue

            ref = sample.text
            hyp = sample.asr_results[target_model].get('hypothesis', '')

            if not ref:
                continue

            try:
                output = jiwer.process_words(ref, hyp)

                for chunk in output.alignments[0]:
                    t = chunk.type
                    ri = chunk.ref_start_idx
                    hi = chunk.hyp_start_idx

                    if t == 'delete':
                        word = output.references[0][ri]
                        deletions[word] += 1
                        total_errors += 1
                        total_del += 1
                    elif t == 'insert':
                        word = output.hypotheses[0][hi]
                        insertions[word] += 1
                        total_errors += 1
                        total_ins += 1
                    elif t == 'substitute':
                        ref_word = output.references[0][ri]
                        hyp_word = output.hypotheses[0][hi]
                        substitutions[ref_word] += 1
                        substitution_map.setdefault(ref_word, Counter())[hyp_word] += 1
                        total_errors += 1
                        total_sub += 1

                total_words += len(output.references[0])

            except Exception:
                continue

        del_df = pd.DataFrame(deletions.most_common(self.top_n), columns=['Word', 'Count'])
        if not del_df.empty:
            del_df['Rate (%)'] = (del_df['Count'] / total_words * 100).round(2)

        ins_df = pd.DataFrame(insertions.most_common(self.top_n), columns=['Word', 'Count'])
        if not ins_df.empty:
            ins_df['Rate (%)'] = (ins_df['Count'] / total_words * 100).round(2)

        sub_df = pd.DataFrame(substitutions.most_common(self.top_n), columns=['Reference', 'Count'])
        if not sub_df.empty:
            sub_df['Rate (%)'] = (sub_df['Count'] / total_words * 100).round(2)

        report = ErrorReport(
            top_deletions=del_df,
            top_insertions=ins_df,
            top_substitutions=sub_df,
            substitution_details=substitution_map,
            total_words=total_words,
            total_errors=total_errors,
            total_deletions=total_del,
            total_insertions=total_ins,
            total_substitutions=total_sub,
        )
        self.report = report
        report.print()
        return dataset
