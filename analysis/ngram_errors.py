from collections import Counter
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass
from ..dataloaders.base import BaseASRDataset
from ..core.processor import Processor


@dataclass
class NgramReport:
    n: int
    model_name: str
    top_missed: pd.DataFrame
    top_inserted: pd.DataFrame

    def print(self):
        print(f"\n🔗 N-gram Error Analysis (N={self.n}, model={self.model_name})")
        print("-" * 60)
        print(f"\n📉 Top Missed Phrases (Deletions):")
        print(self.top_missed.to_markdown(index=False))
        print(f"\n📈 Top Hallucinated Phrases (Insertions):")
        print(self.top_inserted.to_markdown(index=False))


class NgramErrorAnalyzer(Processor):
    """
    Анализирует ошибки на уровне фраз (n-грамм).

    Missed N-grams   — устойчивые словосочетания из reference, которых нет в hypothesis.
    Inserted N-grams — устойчивые словосочетания в hypothesis, которых нет в reference.

    Пример:
        analyzer = NgramErrorAnalyzer(n=2, top_k=20)
        dagrus_n >> analyzer
        analyzer.report.print()
        df_missed = analyzer.report.top_missed
    """

    def __init__(
        self,
        n: int = 2,
        top_k: int = 20,
        model_name: Optional[str] = None,
    ):
        """
        Args:
            n:          Длина n-граммы (2 = биграмма, 3 = триграмма).
            top_k:      Сколько самых частых n-грамм выводить.
            model_name: Имя модели из asr_results. None → агрегация по всем моделям.
        """
        self.n = n
        self.top_k = top_k
        self.model_name = model_name
        self.report: Optional[NgramReport] = None

    def _get_ngrams(self, words: List[str]) -> List[str]:
        if len(words) < self.n:
            return []
        return [" ".join(words[i: i + self.n]) for i in range(len(words) - self.n + 1)]

    def apply_to(self, dataset: BaseASRDataset) -> BaseASRDataset:
        label = self.model_name or "all models"
        print(f"🔗 Analyzing {self.n}-gram errors [{label}] on {dataset.name}...")

        missed_ngrams:   Counter = Counter()
        inserted_ngrams: Counter = Counter()

        for sample in dataset:
            if not sample.text:
                continue

            ref_words = sample.text.split()

            # Определяем, по каким моделям итерировать
            if self.model_name:
                items = [(self.model_name, sample.asr_results.get(self.model_name, {}))]
            else:
                items = list(sample.asr_results.items())

            for _, res in items:
                hyp = res.get('hypothesis', '')
                if not hyp:
                    continue
                hyp_words = hyp.split()

                ref_ngrams = self._get_ngrams(ref_words)
                hyp_ngrams = self._get_ngrams(hyp_words)

                hyp_ngram_set = set(hyp_ngrams)
                ref_ngram_set = set(ref_ngrams)

                for ng in ref_ngrams:
                    if ng not in hyp_ngram_set:
                        missed_ngrams[ng] += 1

                for ng in hyp_ngrams:
                    if ng not in ref_ngram_set:
                        inserted_ngrams[ng] += 1

        missed_df = pd.DataFrame(
            missed_ngrams.most_common(self.top_k), columns=['Phrase', 'Count']
        )
        inserted_df = pd.DataFrame(
            inserted_ngrams.most_common(self.top_k), columns=['Phrase', 'Count']
        )

        self.report = NgramReport(
            n=self.n,
            model_name=label,
            top_missed=missed_df,
            top_inserted=inserted_df,
        )
        self.report.print()
        return dataset
