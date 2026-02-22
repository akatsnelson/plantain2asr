from collections import Counter
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass
from ..dataloaders.base import BaseASRDataset
from ..core.processor import Processor

try:
    import jiwer
except ImportError:
    jiwer = None

@dataclass
class NgramReport:
    n: int
    top_missed: pd.DataFrame
    top_inserted: pd.DataFrame
    
    def print(self):
        print(f"\n🔗 N-gram Error Analysis (N={self.n})")
        print("-" * 60)
        
        print(f"\n📉 Top Missed Phrases (Deletions):")
        print(self.top_missed.to_markdown(index=False))
        
        print(f"\n📈 Top Hallucinated Phrases (Insertions):")
        print(self.top_inserted.to_markdown(index=False))

class NgramErrorAnalyzer(Processor):
    """
    Анализирует ошибки на уровне фраз (n-грамм).
    Помогает найти устойчивые словосочетания, которые модель:
    - Постоянно пропускает (Missed N-grams)
    - Постоянно выдумывает (Hallucinated N-grams)
    """
    def __init__(self, n: int = 2, top_k: int = 10):
        self.n = n
        self.top_k = top_k
        if jiwer is None:
            print("⚠️ 'jiwer' is required for alignment.")

    def _get_ngrams(self, words: List[str]) -> List[str]:
        if len(words) < self.n: return []
        return [" ".join(words[i:i+self.n]) for i in range(len(words)-self.n+1)]

    def apply_to(self, dataset: BaseASRDataset) -> BaseASRDataset:
        if jiwer is None:
            return dataset
        print(f"🔗 Analyzing {self.n}-gram errors...")
        
        missed_ngrams = Counter()
        inserted_ngrams = Counter()
        
        models = set()
        for s in dataset: models.update(s.asr_results.keys())
        # Для простоты берем первую попавшуюся модель или агрегируем по всем?
        # Лучше агрегировать по всем, чтобы найти общие проблемы.
        
        for sample in dataset:
            if not sample.text: continue
            
            ref_words = sample.text.lower().split()
            
            for model_name, res in sample.asr_results.items():
                hyp = res.get('hypothesis', '')
                if not hyp: continue
                hyp_words = hyp.lower().split()
                
                # Используем jiwer для выравнивания, чтобы понять, какие слова удалены/вставлены
                try:
                    out = jiwer.process_words(ref_words, hyp_words)
                    
                    # Восстанавливаем последовательности удалений/вставок
                    # Это сложная задача через чанки, упростим:
                    # Просто берем n-граммы из Ref, которых нет в Hyp (как множества) - грубо, но быстро
                    # Более точно: идем по чанкам.
                    
                    # Вариант 2: "Мягкий" поиск
                    # Missed = n-граммы Ref, которые не встречаются в Hyp
                    # Inserted = n-граммы Hyp, которые не встречаются в Ref
                    
                    ref_ngrams = self._get_ngrams(ref_words)
                    hyp_ngrams = self._get_ngrams(hyp_words)
                    
                    ref_ngram_set = set(ref_ngrams)
                    hyp_ngram_set = set(hyp_ngrams)
                    
                    for ng in ref_ngrams:
                        if ng not in hyp_ngram_set:
                            missed_ngrams[ng] += 1
                            
                    for ng in hyp_ngrams:
                        if ng not in ref_ngram_set:
                            inserted_ngrams[ng] += 1
                            
                except:
                    continue
                    
        # Формируем отчет
        missed_df = pd.DataFrame(missed_ngrams.most_common(self.top_k), columns=['Phrase', 'Count'])
        inserted_df = pd.DataFrame(inserted_ngrams.most_common(self.top_k), columns=['Phrase', 'Count'])
        
        report = NgramReport(self.n, missed_df, inserted_df)
        report.print()
        self.report = report
        return dataset
