import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict
from ..dataloaders.base import BaseASRDataset

@dataclass
class DurationReport:
    wer_by_duration: pd.DataFrame

    def print(self):
        print("\n⏳ Duration Robustness Analysis (WER by Audio Length)")
        print("-" * 60)
        print(self.wer_by_duration.to_markdown(index=False, floatfmt=".1f"))
        print("\n💡 Insight: Models often fail on very short (<3s) due to lack of context")
        print("           or very long (>20s) due to attention limits.")

class DurationAnalyzer:
    """
    Анализирует качество (WER) в зависимости от длительности аудио.
    Разбивает на группы: Short, Medium, Long, Extra Long.
    """
    def __init__(self, bins: List[float] = [0, 3, 10, 20, float('inf')], labels: List[str] = ["Short (<3s)", "Medium (3-10s)", "Long (10-20s)", "Extra (>20s)"]):
        self.bins = bins
        self.labels = labels

    def apply_to(self, dataset: BaseASRDataset) -> DurationReport:
        print("⏳ Analyzing WER by duration...")
        
        models = set()
        for s in dataset: models.update(s.asr_results.keys())
        
        # Structure: bucket_idx -> model -> list of WERs
        bucket_wers = {i: {m: [] for m in models} for i in range(len(self.labels))}
        bucket_counts = {i: 0 for i in range(len(self.labels))}
        
        for s in dataset:
            dur = getattr(s, 'duration', 0)
            
            # Find bucket
            b_idx = -1
            for i in range(len(self.bins) - 1):
                if self.bins[i] <= dur < self.bins[i+1]:
                    b_idx = i
                    break
            
            if b_idx == -1: continue
            
            bucket_counts[b_idx] += 1
            
            for m, res in s.asr_results.items():
                metrics = res.get('metrics', {})
                if 'wer' in metrics:
                    bucket_wers[b_idx][m].append(metrics['wer'])
                    
        # Build DataFrame
        data = []
        for i, label in enumerate(self.labels):
            row = {'Duration Group': label, 'Count': bucket_counts[i]}
            for m in sorted(models):
                wers = bucket_wers[i][m]
                row[m] = np.mean(wers) * 100 if wers else None
            data.append(row)
            
        df = pd.DataFrame(data)
        report = DurationReport(df)
        report.print()
        return report
