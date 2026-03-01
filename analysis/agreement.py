from collections import defaultdict
from typing import List, Dict, Optional
import pandas as pd
from dataclasses import dataclass, field
from ..dataloaders.base import BaseASRDataset
from ..core.processor import Processor


@dataclass
class AgreementReport:
    total_samples: int
    all_agree: int
    none_agree: int
    partial_agree: int
    disagreements: List[Dict]  # топ полных расхождений по гипотезам
    # Сэмплы, где все модели имеют WER > threshold ("слепые зоны")
    blind_spots: List[Dict] = field(default_factory=list)

    def print(self):
        print("\n🤝 Agreement Analysis Report")
        print("-" * 60)

        total = self.total_samples or 1
        df = pd.DataFrame([
            {"Категория": "Все модели согласны",
             "Кол-во": self.all_agree,   "%": self.all_agree   / total * 100},
            {"Категория": "Все модели расходятся",
             "Кол-во": self.none_agree,  "%": self.none_agree  / total * 100},
            {"Категория": "Частичное согласие",
             "Кол-во": self.partial_agree, "%": self.partial_agree / total * 100},
        ])
        print(df.to_markdown(index=False, floatfmt=".1f"))

        print(f"\n💡 Интерпретация:")
        print(f"   {self.all_agree} 'лёгких' сэмплов (автоматическая разметка)")
        print(f"   {self.none_agree} сэмплов с полным расхождением")
        if self.blind_spots:
            print(f"   {len(self.blind_spots)} 'слепых зон' (все модели WER > порога)")

        if self.disagreements:
            print("\n🔍 Примеры полного расхождения гипотез:")
            for i, item in enumerate(self.disagreements[:3], 1):
                print(f"\n  Сэмпл {i}: {item.get('ref', item.get('audio',''))[:80]}")
                for model, hyp in list(item['hyps'].items())[:4]:
                    print(f"    {model:<30}: {hyp[:70]}")

    def blind_spots_df(self) -> pd.DataFrame:
        """DataFrame со слепыми зонами: ref, wer_mean, wer по каждой модели."""
        if not self.blind_spots:
            return pd.DataFrame()
        return pd.DataFrame(self.blind_spots).sort_values('wer_mean', ascending=False)


class AgreementAnalyzer(Processor):
    """
    Анализирует согласованность между моделями.

    Находит:
    - лёгкие сэмплы (все модели согласны)
    - полные расхождения (каждая модель вывела что-то уникальное)
    - «слепые зоны» — сэмплы, где все модели имеют WER выше порога

    Пример:
        analyzer = AgreementAnalyzer(wer_blind_threshold=0.5)
        dagrus_n >> analyzer
        df_blind = analyzer.report.blind_spots_df()
    """

    def __init__(self, min_models: int = 2, wer_blind_threshold: float = 0.5):
        """
        Args:
            min_models:           Минимум моделей у сэмпла для учёта.
            wer_blind_threshold:  WER порог для «слепых зон» (0–1).
                                  Сэмпл попадает в blind_spots, если ВСЕ модели
                                  имеют WER выше этого значения.
        """
        self.min_models = min_models
        self.wer_blind_threshold = wer_blind_threshold
        self.report: Optional[AgreementReport] = None

    def apply_to(self, dataset: BaseASRDataset) -> BaseASRDataset:
        print(f"🤝 Analyzing model agreement on {dataset.name}...")

        all_agree = none_agree = partial_agree = valid_samples = 0
        disagreements: List[Dict] = []
        blind_spots:   List[Dict] = []

        for sample in dataset:
            hyps: Dict[str, str] = {}
            wers: Dict[str, float] = {}

            for model_name, res in sample.asr_results.items():
                hyp = res.get('hypothesis', '')
                if hyp:
                    hyps[model_name] = hyp.strip().lower()
                metrics = res.get('metrics', {})
                wer_val = metrics.get('WER', metrics.get('wer'))
                if wer_val is not None:
                    wers[model_name] = float(wer_val)

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
                if len(disagreements) < 20:
                    disagreements.append({
                        'audio': sample.audio_path,
                        'ref':   sample.text or '',
                        'hyps':  hyps,
                    })
            else:
                partial_agree += 1

            # Слепые зоны: у всех моделей с WER-данными WER > threshold
            if wers and all(w > self.wer_blind_threshold for w in wers.values()):
                wer_mean = sum(wers.values()) / len(wers)
                row = {
                    'ref':      sample.text or '',
                    'audio':    sample.audio_path,
                    'wer_mean': round(wer_mean, 3),
                    'n_models': len(wers),
                }
                row.update({f'wer_{m}': round(v, 3) for m, v in wers.items()})
                blind_spots.append(row)

        if valid_samples == 0:
            print("⚠️ Недостаточно сэмплов с несколькими моделями.")
            self.report = None
            return dataset

        self.report = AgreementReport(
            total_samples=valid_samples,
            all_agree=all_agree,
            none_agree=none_agree,
            partial_agree=partial_agree,
            disagreements=disagreements,
            blind_spots=blind_spots,
        )
        self.report.print()
        return dataset
