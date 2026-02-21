from collections import Counter
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass, field

try:
    import jiwer
except ImportError:
    jiwer = None

from ..dataloaders.base import BaseASRDataset

@dataclass
class ErrorReport:
    """Контейнер для отчета об ошибках"""
    top_deletions: pd.DataFrame
    top_insertions: pd.DataFrame
    top_substitutions: pd.DataFrame
    substitution_details: Dict[str, Counter] # word -> Counter of what it was replaced with
    total_words: int
    total_errors: int

    def print(self):
        print(f"\n📊 Word Error Analysis Report (Sections 8 & 9)")
        print(f"Total words: {self.total_words}, Total errors: {self.total_errors}")
        print("-" * 40)
        
        print(f"\n📉 Top Missed Words (Deletions):")
        print(self.top_deletions.to_markdown(index=False))
            
        print(f"\n📈 Top Hallucinations (Insertions):")
        print(self.top_insertions.to_markdown(index=False))
            
        print(f"\n🔄 Top Confusions (Substitutions):")
        # Добавляем примеры замен в таблицу вывода
        subs_df = self.top_substitutions.copy()
        subs_df['Examples (->)'] = subs_df['Reference'].apply(
            lambda w: ", ".join([f"{k} ({v})" for k,v in self.substitution_details.get(w, Counter()).most_common(3)])
        )
        print(subs_df[['Reference', 'Count', 'Examples (->)']].to_markdown(index=False))

class WordErrorAnalyzer:
    """
    Анализатор ошибок на уровне слов.
    Строит рейтинг самых частых ошибок (пропуски, вставки, замены) и примеры путаницы.
    """
    def __init__(self, model_name: Optional[str] = None, top_n: int = 15):
        self.model_name = model_name
        self.top_n = top_n
        
        if jiwer is None:
            print("⚠️ 'jiwer' not installed. Please install it for detailed analysis: pip install jiwer")

    def apply_to(self, dataset: BaseASRDataset) -> ErrorReport:
        if jiwer is None:
            return None
            
        # Определяем имя модели, если не задано
        target_model = self.model_name
        if not target_model:
            for s in dataset:
                if s.asr_results:
                    target_model = next(iter(s.asr_results.keys()))
                    break
        
        print(f"🔍 Analyzing word errors for model: {target_model or 'Unknown'}...")
        
        deletions = Counter()
        insertions = Counter()
        substitutions = Counter() # (ref, hyp) -> count
        substitution_map = {} # ref -> Counter(hyp)
        
        total_words = 0
        total_errors = 0
        
        if not target_model:
            print("⚠️ No model results found.")
            return None

        for sample in dataset:
            if not sample.text or target_model not in sample.asr_results:
                continue
                
            ref = sample.text.lower()
            hyp = sample.asr_results[target_model].get('hypothesis', '').lower()
            
            try:
                output = jiwer.process_words(ref, hyp)
                
                # Alignments structure depends on jiwer version, but generally available
                for align_chunk in output.alignments[0]:
                    type_ = align_chunk.type
                    ref_idx = align_chunk.ref_start_idx
                    hyp_idx = align_chunk.hyp_start_idx
                    
                    if type_ == 'delete':
                        word = output.references[ref_idx]
                        deletions[word] += 1
                        total_errors += 1
                    elif type_ == 'insert':
                        word = output.hypothesis[hyp_idx]
                        insertions[word] += 1
                        total_errors += 1
                    elif type_ == 'substitute':
                        ref_word = output.references[ref_idx]
                        hyp_word = output.hypothesis[hyp_idx]
                        substitutions[ref_word] += 1
                        
                        if ref_word not in substitution_map:
                            substitution_map[ref_word] = Counter()
                        substitution_map[ref_word][hyp_word] += 1
                        
                        total_errors += 1
                
                total_words += len(output.references)
                
            except Exception:
                continue

        # Prepare DataFrames
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
            total_errors=total_errors
        )
        
        report.print()
        return report
