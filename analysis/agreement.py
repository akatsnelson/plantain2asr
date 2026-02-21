from collections import defaultdict
from typing import List, Dict, Set
import pandas as pd
from dataclasses import dataclass
from ..dataloaders.base import BaseASRDataset

@dataclass
class AgreementReport:
    total_samples: int
    all_agree: int
    none_agree: int
    partial_agree: int
    disagreements: List[Dict] # Топ расхождений

    def print(self):
        print("\n🤝 Agreement Analysis Report (Section 3)")
        print("-" * 60)
        
        df = pd.DataFrame([
            {"Category": "Full Agreement (All models match)", "Count": self.all_agree, "%": self.all_agree/self.total_samples*100},
            {"Category": "Total Disagreement (All unique)", "Count": self.none_agree, "%": self.none_agree/self.total_samples*100},
            {"Category": "Partial Agreement", "Count": self.partial_agree, "%": self.partial_agree/self.total_samples*100}
        ])
        print(df.to_markdown(index=False, floatfmt=".2f"))
        
        print(f"\n💡 Interpretation:")
        print(f"   - {self.all_agree} 'Easy' samples (can be auto-labeled)")
        print(f"   - {self.none_agree} 'Hard' samples (require manual review)")
        
        if self.disagreements:
            print("\n🔍 Examples of Total Disagreement:")
            for i, item in enumerate(self.disagreements[:3], 1):
                print(f"\nSample {i}: {item['audio']}")
                for model, hyp in list(item['hyps'].items())[:4]: # Показываем первые 4
                    print(f"   - {model}: {hyp}")

class AgreementAnalyzer:
    """
    Анализирует согласованность между моделями.
    Помогает найти сложные (где модели расходятся) и легкие (где согласны) примеры.
    """
    def __init__(self, min_models: int = 2):
        self.min_models = min_models

    def apply_to(self, dataset: BaseASRDataset) -> AgreementReport:
        print(f"🤝 Analyzing model agreement...")
        
        all_agree = 0
        none_agree = 0
        partial_agree = 0
        
        disagreements = []
        valid_samples = 0
        
        for sample in dataset:
            hyps = {}
            for model_name, res in sample.asr_results.items():
                if 'hypothesis' in res and res['hypothesis']:
                    hyps[model_name] = res['hypothesis'].strip().lower()
            
            if len(hyps) < self.min_models:
                continue
                
            valid_samples += 1
            unique_hyps = set(hyps.values())
            n_unique = len(unique_hyps)
            n_models = len(hyps)
            
            if n_unique == 1:
                all_agree += 1
            elif n_unique == n_models:
                none_agree += 1
                if len(disagreements) < 10:
                    disagreements.append({
                        'audio': sample.audio_path,
                        'hyps': hyps
                    })
            else:
                partial_agree += 1
                
        if valid_samples == 0:
            print("⚠️ Not enough samples with multiple model results.")
            return None
            
        report = AgreementReport(
            total_samples=valid_samples,
            all_agree=all_agree,
            none_agree=none_agree,
            partial_agree=partial_agree,
            disagreements=disagreements
        )
        
        report.print()
        return report
