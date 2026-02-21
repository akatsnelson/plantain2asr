from typing import List
import logging
import warnings
import os
from torchmetrics.text.bert import BERTScore as TorchBERTScore
from ..base import BaseMetric

class BERTScore(BaseMetric):
    """
    BERTScore (Semantic Similarity).
    Использует эмбеддинги BERT для оценки смысловой близости.
    
    Внимание: Требует загрузки модели (~500MB-1GB) и работает медленно на CPU.
    """
    
    def __init__(self, do_clean: bool = True, model_name: str = "bert-base-multilingual-cased"):
        super().__init__(do_clean)
        self.model_name = model_name
        # Инициализируем лениво, чтобы не грузить модель при импорте
        self._metric = None

    @property
    def metric(self):
        if self._metric is None:
            print(f"⏳ Loading BERTScore model: {self.model_name}...")
            
            # Подавляем шумные ворнинги от transformers/torchmetrics при загрузке
            logging.getLogger("transformers").setLevel(logging.ERROR)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # num_layers=9 - обычно оптимальный слой для семантики
            self._metric = TorchBERTScore(model_name_or_path=self.model_name, num_layers=9)
        return self._metric

    @property
    def name(self) -> str:
        return "BERTScore"

    def calculate(self, reference: str, hypothesis: str) -> float:
        if self.do_clean:
            reference = self.normalize(reference)
            hypothesis = self.normalize(hypothesis)

        if not reference.strip():
            return 0.0 if hypothesis.strip() else 100.0 
            
        # Подавляем ворнинги "The following layers were not sharded" при инференсе
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*The following layers were not sharded.*")
        # BERTScore возвращает словарь {f1, precision, recall}
        score = self.metric([hypothesis], [reference])
        
        # Извлекаем F1. Обработка 0-d и 1-d тензоров.
        f1 = score["f1"]
        if f1.ndim == 0:
            return float(f1.item()) * 100.0
        else:
            return float(f1.mean().item()) * 100.0

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> float:
        if self.do_clean:
            references = [self.normalize(r) for r in references]
            hypotheses = [self.normalize(h) for h in hypotheses]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*The following layers were not sharded.*")
        score = self.metric(hypotheses, references)
            
        return float(score["f1"].mean().item()) * 100.0
