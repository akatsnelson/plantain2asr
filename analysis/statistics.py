import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..dataloaders.base import BaseASRDataset

@dataclass
class BootstrapReport:
    ci_df: pd.DataFrame
    
    def print(self):
        print("\n📊 Statistical Significance (Bootstrap 95% CI) (Section 4)")
        print("-" * 80)
        print(self.ci_df.to_markdown(index=False, floatfmt=".2f"))

class BootstrapAnalyzer:
    """
    Рассчитывает доверительные интервалы (Confidence Intervals) для WER
    используя метод Bootstrap. Позволяет оценить статистическую значимость разницы.
    """
    def __init__(self, n_iterations: int = 1000, ci_level: float = 0.95):
        self.n_iterations = n_iterations
        self.ci_level = ci_level

    def apply_to(self, dataset: BaseASRDataset) -> BootstrapReport:
        models = set()
        for s in dataset:
            models.update(s.asr_results.keys())
            
        results = []
        alpha = (1 - self.ci_level) / 2
        
        print(f"🎲 Running Bootstrap Analysis ({self.n_iterations} iterations)...")
        
        for model_name in models:
            wers = []
            for s in dataset:
                if model_name in s.asr_results:
                    val = s.asr_results[model_name].get('metrics', {}).get('wer')
                    if val is not None:
                        wers.append(val)
            
            if not wers:
                continue
                
            wers_arr = np.array(wers)
            means = []
            
            # Bootstrap resampling
            # Оптимизированная версия через numpy
            # Генерируем индексы (n_iter, n_samples)
            n = len(wers_arr)
            # Делаем чанками, чтобы память не взорвалась если выборка большая
            chunk_size = 100
            for _ in range(0, self.n_iterations, chunk_size):
                current_chunk = min(chunk_size, self.n_iterations - _)
                indices = np.random.randint(0, n, (current_chunk, n))
                sample_means = wers_arr[indices].mean(axis=1)
                means.extend(sample_means)
            
            means = np.array(means)
            lower = np.percentile(means, alpha * 100)
            upper = np.percentile(means, (1 - alpha) * 100)
            mean_val = np.mean(wers_arr)
            
            results.append({
                'Model': model_name,
                'Mean WER (%)': mean_val * 100,
                'CI Lower (%)': lower * 100,
                'CI Upper (%)': upper * 100,
                'CI Width (%)': (upper - lower) * 100
            })
            
        df = pd.DataFrame(results)
        if 'Mean WER (%)' in df.columns:
            df = df.sort_values('Mean WER (%)')
            
        return BootstrapReport(df)
