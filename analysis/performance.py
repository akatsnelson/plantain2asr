import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

try:
    import jiwer
except ImportError:
    jiwer = None

from ..dataloaders.base import BaseASRDataset

@dataclass
class PerformanceReport:
    summary_df: pd.DataFrame
    
    def print(self):
        print("\n📊 Model Performance Summary (Sections 1, 2, 3, 6, 7)")
        print("-" * 80)
        # Форматируем для красивого вывода
        print(self.summary_df.to_markdown(index=False, floatfmt=".2f"))

class PerformanceAnalyzer:
    """
    Анализирует общую производительность моделей:
    - Quality: WER, CER
    - Efficiency: RTF (Real Time Factor)
    - Reliability: Perfect Match %, Catastrophic Failure %
    - Error Components: Insertions, Deletions, Substitutions (IDR)
    """
    def __init__(self, catastrophic_threshold: float = 1.0):
        self.catastrophic_threshold = catastrophic_threshold # WER > 100% считается катастрофой

    def apply_to(self, dataset: BaseASRDataset) -> PerformanceReport:
        if len(dataset) == 0:
            print("⚠️ Dataset is empty.")
            return None
        
        # Собираем все модели, которые есть в результатах
        models = set()
        for s in dataset:
            models.update(s.asr_results.keys())
        
        results_data = []

        for model_name in models:
            metrics_acc = {
                'wer': [], 'cer': [], 
                'insertions': [], 'deletions': [], 'substitutions': [],
                'rtf': [], 'perfect': 0, 'catastrophic': 0,
                'count': 0
            }
            
            total_audio_duration = 0.0
            total_proc_time = 0.0

            for sample in dataset:
                if model_name not in sample.asr_results:
                    continue
                
                res = sample.asr_results[model_name]
                metrics = res.get('metrics', {})
                
                # WER/CER
                wer = metrics.get('wer', None)
                cer = metrics.get('cer', None)
                
                if wer is not None:
                    metrics_acc['wer'].append(wer)
                    if wer == 0.0:
                        metrics_acc['perfect'] += 1
                    if wer >= self.catastrophic_threshold:
                        metrics_acc['catastrophic'] += 1
                
                if cer is not None:
                    metrics_acc['cer'].append(cer)

                # RTF calculation
                proc_time = res.get('processing_time', 0.0)
                # Если длительность есть в сэмпле, используем её
                duration = getattr(sample, 'duration', 0.0) or getattr(sample, 'audio_duration', 0.0)
                
                if duration > 0 and proc_time > 0:
                    total_audio_duration += duration
                    total_proc_time += proc_time
                    metrics_acc['rtf'].append(proc_time / duration)

                # IDR Analysis (требует jiwer для пересчета, если метрики не сохранены детально)
                # Если мы уже сохраняли ins/del/sub в метриках - берем оттуда, иначе считаем на лету
                # Для скорости здесь предположим, что мы считаем на лету, если есть текст
                if jiwer and sample.text and 'hypothesis' in res:
                    try:
                        out = jiwer.process_words(sample.text, res['hypothesis'])
                        # Normalize counts by reference length for rates, or raw counts?
                        # Usually IDR is average counts or rates. Let's store raw counts and normalize later.
                        # Но jiwer возвращает абсолютные числа.
                        # Rate = Count / Len(Ref). 
                        ref_len = len(out.references)
                        if ref_len > 0:
                            metrics_acc['insertions'].append(out.insertions / ref_len)
                            metrics_acc['deletions'].append(out.deletions / ref_len)
                            metrics_acc['substitutions'].append(out.substitutions / ref_len)
                    except:
                        pass
                
                metrics_acc['count'] += 1

            if metrics_acc['count'] == 0:
                continue

            # Aggregating
            row = {
                'Model': model_name,
                'WER (%)': np.mean(metrics_acc['wer']) * 100 if metrics_acc['wer'] else 0,
                'CER (%)': np.mean(metrics_acc['cer']) * 100 if metrics_acc['cer'] else 0,
                'RTF': (total_proc_time / total_audio_duration) if total_audio_duration > 0 else np.mean(metrics_acc['rtf']) if metrics_acc['rtf'] else 0,
                'Perfect (%)': (metrics_acc['perfect'] / metrics_acc['count']) * 100,
                'Catastrophic (%)': (metrics_acc['catastrophic'] / metrics_acc['count']) * 100,
            }
            
            # Add IDR if available
            if metrics_acc['insertions']:
                row['Ins Rate'] = np.mean(metrics_acc['insertions'])
                row['Del Rate'] = np.mean(metrics_acc['deletions'])
                row['Sub Rate'] = np.mean(metrics_acc['substitutions'])
                
                # Ratio string I:D:S
                total_err = row['Ins Rate'] + row['Del Rate'] + row['Sub Rate']
                if total_err > 0:
                    i_r = row['Ins Rate'] / total_err
                    d_r = row['Del Rate'] / total_err
                    s_r = row['Sub Rate'] / total_err
                    row['IDR Ratio'] = f"{i_r:.2f}:{d_r:.2f}:{s_r:.2f}"
                else:
                    row['IDR Ratio'] = "0:0:0"

            results_data.append(row)

        df = pd.DataFrame(results_data)
        # Сортировка по WER
        if 'WER (%)' in df.columns:
            df = df.sort_values('WER (%)')
            
        return PerformanceReport(df)
