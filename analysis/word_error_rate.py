from collections import Counter
from typing import Dict, Optional
import pandas as pd
from dataclasses import dataclass

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

    def print(self):
        print(f"\n📊 Word Error Analysis Report")
        print(f"Total words: {self.total_words}, Total errors: {self.total_errors}")
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
                    elif t == 'insert':
                        word = output.hypotheses[0][hi]
                        insertions[word] += 1
                        total_errors += 1
                    elif t == 'substitute':
                        ref_word = output.references[0][ri]
                        hyp_word = output.hypotheses[0][hi]
                        substitutions[ref_word] += 1
                        substitution_map.setdefault(ref_word, Counter())[hyp_word] += 1
                        total_errors += 1

                total_words += len(output.references[0])

            except Exception:
                continue

        del_df = pd.DataFrame(deletions.most_common(self.top_n), columns=['Word', 'Count'])
        if not del_df.empty:
            del_df['Rate (%)'] = (del_df['Count'] / total_errors * 100).round(1)

        ins_df = pd.DataFrame(insertions.most_common(self.top_n), columns=['Word', 'Count'])
        sub_df = pd.DataFrame(substitutions.most_common(self.top_n), columns=['Reference', 'Count'])

        report = ErrorReport(
            top_deletions=del_df,
            top_insertions=ins_df,
            top_substitutions=sub_df,
            substitution_details=substitution_map,
            total_words=total_words,
            total_errors=total_errors,
        )
        self.report = report
        report.print()
        return dataset
