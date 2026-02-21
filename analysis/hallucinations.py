import re
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Set
from ..dataloaders.base import BaseASRDataset

@dataclass
class HallucinationReport:
    summary_df: pd.DataFrame
    examples: Dict[str, List[Dict]] # model -> list of {ref, hyp, reason}

    def print(self):
        print("\n👻 Hallucination Analysis Report")
        print("-" * 60)
        print(self.summary_df.to_markdown(index=False, floatfmt=".2f"))
        
        print("\n👀 Examples of Hallucinations:")
        for model, exs in self.examples.items():
            if not exs: continue
            print(f"\nModel: {model}")
            for i, ex in enumerate(exs[:3], 1):
                print(f"  {i}. Reason: {ex['reason']}")
                print(f"     Ref: '{ex['ref']}'")
                print(f"     Hyp: '{ex['hyp']}'")

class HallucinationAnalyzer:
    """
    Выявляет "галлюцинации" моделей:
    1. Гипотеза намного длиннее референса.
    2. Гипотеза содержит типичные слова-паразиты (субтитры, лайк, подписка).
    3. Гипотеза есть, а референс пустой.
    """
    def __init__(self, length_ratio_threshold: float = 3.0, min_len: int = 5):
        self.length_ratio_threshold = length_ratio_threshold
        self.min_len = min_len
        self.triggers = {
            'субтитры', 'продолжение', 'следует', 'видео', 'канал', 
            'подписывайтесь', 'лайк', 'колокольчик', 'просмотр'
        }

    def _is_hallucination(self, ref: str, hyp: str) -> str:
        ref_words = ref.lower().split()
        hyp_words = hyp.lower().split()
        
        # 1. Пустой референс, но длинная гипотеза
        if len(ref_words) == 0 and len(hyp_words) >= self.min_len:
            return "Empty Ref, Long Hyp"
            
        # 2. Триггеры (YouTube мусор)
        if any(w in self.triggers for w in hyp_words):
            return "Trigger Word Found"
            
        # 3. Аномальная длина
        if len(ref_words) > 0 and len(hyp_words) > len(ref_words) * self.length_ratio_threshold and len(hyp_words) > self.min_len:
            return f"Length Ratio > {self.length_ratio_threshold}"
            
        return None

    def apply_to(self, dataset: BaseASRDataset) -> HallucinationReport:
        print("👻 Detecting hallucinations...")
        
        models = set()
        for s in dataset: models.update(s.asr_results.keys())
        
        stats = []
        examples = {}
        
        for model in models:
            total = 0
            hallucinations = 0
            model_examples = []
            
            for s in dataset:
                if model not in s.asr_results: continue
                
                hyp = s.asr_results[model].get('hypothesis', '')
                ref = s.text or ""
                
                reason = self._is_hallucination(ref, hyp)
                if reason:
                    hallucinations += 1
                    if len(model_examples) < 5:
                        model_examples.append({'ref': ref, 'hyp': hyp, 'reason': reason})
                
                total += 1
            
            if total > 0:
                stats.append({
                    'Model': model,
                    'Samples': total,
                    'Hallucinations': hallucinations,
                    'Rate (%)': (hallucinations / total * 100)
                })
                examples[model] = model_examples
                
        df = pd.DataFrame(stats)
        if 'Rate (%)' in df.columns:
            df = df.sort_values('Rate (%)', ascending=False)
            
        report = HallucinationReport(df, examples)
        report.print()
        return report
