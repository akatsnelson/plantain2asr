import numpy as np
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass
from ..dataloaders.base import BaseASRDataset
from ..core.processor import Processor

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

@dataclass
class CalibrationReport:
    plot_path: Optional[str]
    data: pd.DataFrame
    
    def print(self):
        print("\n📈 Calibration Analysis (Length vs WER)")
        print("-" * 60)
        if self.plot_path:
            print(f"✅ Plot saved to: {self.plot_path}")
        else:
            print("⚠️ Matplotlib not installed, plot skipped.")
            
        print("\nData Summary:")
        print(self.data.to_markdown(index=False, floatfmt=".1f"))

class CalibrationAnalyzer(Processor):
    """
    Строит график зависимости качества (WER) от длительности аудио.
    Помогает понять 'зону комфорта' модели (например, 5-15 секунд)
    и где она ломается (слишком короткие или слишком длинные).
    """
    def __init__(self, output_plot: str = "calibration_plot.png", bins: int = 10):
        self.output_plot = output_plot
        self.bins = bins

    def apply_to(self, dataset: BaseASRDataset) -> BaseASRDataset:
        print("📈 Generating Calibration Plot...")
        
        data_points = []
        for s in dataset:
            dur = getattr(s, 'duration', 0)
            if dur <= 0: continue
            
            for model_name, res in s.asr_results.items():
                metrics = res.get('metrics', {})
                if 'wer' in metrics:
                    data_points.append({
                        'Model': model_name,
                        'Duration': dur,
                        'WER': metrics['wer'] * 100
                    })
                    
        df = pd.DataFrame(data_points)
        if df.empty:
            print("⚠️ No data for calibration plot.")
            self.report = None
            return dataset
            
        # Биннинг по длительности
        df['DurationBin'] = pd.cut(df['Duration'], bins=self.bins)
        
        # Агрегация
        agg_df = df.groupby(['Model', 'DurationBin'], observed=True)['WER'].mean().reset_index()
        # Преобразуем интервалы в строки для красоты
        agg_df['DurationBin'] = agg_df['DurationBin'].apply(lambda x: f"{x.left:.0f}-{x.right:.0f}s")
        
        pivot_df = agg_df.pivot(index='DurationBin', columns='Model', values='WER').reset_index()

        # Рисуем график
        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(10, 6))
            
            # Получаем средние значения середин интервалов для оси X
            # (немного костыльно, но работает для cut)
            intervals = df.groupby('DurationBin', observed=True)['Duration'].mean()
            
            for model_name in df['Model'].unique():
                model_data = agg_df[agg_df['Model'] == model_name]
                # Используем индекс бинов для X, чтобы было равномерно
                plt.plot(range(len(model_data)), model_data['WER'], marker='o', label=model_name)
            
            plt.xticks(range(len(intervals)), [f"{i.left:.0f}-{i.right:.0f}s" for i in intervals.index], rotation=45)
            plt.title('Model Calibration: WER vs Audio Duration')
            plt.xlabel('Audio Duration')
            plt.ylabel('WER (%)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_plot)
            plt.close()
        
        report = CalibrationReport(
            plot_path=self.output_plot if MATPLOTLIB_AVAILABLE else None,
            data=pivot_df,
        )
        report.print()
        self.report = report
        return dataset
