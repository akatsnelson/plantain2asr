import os
import torch
import logging
import warnings
import sys
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from ..models.base import BaseASRModel
from ..dataloaders.base import BaseASRDataset
from ..metrics.base import BaseMetric
from .config import TrainingConfig

# Suppress verbose warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
warnings.filterwarnings("ignore", message=".*Triton.*") # Suppress torchao triton warning
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

try:
    import wandb
except ImportError:
    wandb = None

class BaseTrainer(ABC):
    """
    Базовый класс для обучения ASR моделей.
    Обеспечивает интеграцию с пайплайном, логирование и цикл обучения.
    """
    def __init__(
        self, 
        model: BaseASRModel, 
        config: TrainingConfig, 
        eval_dataset: Optional[BaseASRDataset] = None,
        metrics: Optional[List[BaseMetric]] = None # Список метрик (WER, CER)
    ):
        self.model_wrapper = model
        self.config = config
        self.eval_dataset = eval_dataset
        self.metrics = metrics or []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Init vars
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.processor = None
        self._wandb_active = False

    @abstractmethod
    def prepare_training(self, train_dataset: BaseASRDataset):
        """Подготовка к обучению (DataLoader, Optimizer)."""
        pass

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Один шаг обучения (forward + backward)."""
        pass
    
    @abstractmethod
    def eval_step(self, batch: Any) -> Dict[str, Any]:
        """
        Один шаг валидации. 
        Должен возвращать {'loss': float, 'preds': List[str], 'refs': List[str]} 
        для расчета метрик.
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str):
        """Сохранение модели."""
        pass

    def _compute_metrics(self, preds: List[str], refs: List[str]) -> Dict[str, float]:
        """Расчет метрик на всем валидационном сете."""
        results = {}
        for metric in self.metrics:
            try:
                # Если метрика поддерживает батчевый расчет
                if hasattr(metric, 'calculate_batch'):
                    score = metric.calculate_batch(refs, preds)
                else:
                    # Иначе считаем по одному (медленнее)
                    scores = [metric.calculate(r, p) for r, p in zip(refs, preds)]
                    score = sum(scores) / len(scores) if scores else 0.0
                
                results[metric.name] = score
            except Exception as e:
                print(f"⚠️ Metric calculation failed for {metric.name}: {e}")
        return results

    def apply_to(self, train_dataset: BaseASRDataset) -> BaseASRModel:
        print(f"🚀 Starting Custom Training for {self.model_wrapper.name} on {self.device}...")
        
        # 1. Setup WandB
        if self.config.project_name and wandb:
            try:
                wandb.init(
                    project=self.config.project_name, 
                    name=self.config.run_name,
                    config=self.config.__dict__
                )
                self._wandb_active = True
                print("   ✅ WandB initialized.")
            except Exception as e:
                print(f"   ⚠️ WandB init failed: {e}. Continuing offline.")
                self._wandb_active = False
        
        # 2. Prepare
        self.prepare_training(train_dataset)
        
        # 3. Loop
        global_step = 0
        best_loss = float('inf')
        
        self.model_wrapper.model.to(self.device)
        
        for epoch in range(self.config.num_train_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_train_epochs}")
            
            # --- Training ---
            self.model_wrapper.model.train()
            train_pbar = tqdm(self.train_loader, desc="Training")
            epoch_loss = 0.0
            steps = 0
            
            for batch in train_pbar:
                step_res = self.train_step(batch)
                loss = step_res.get('loss', 0.0)
                epoch_loss += loss
                steps += 1
                global_step += 1
                
                train_pbar.set_postfix(loss=f"{loss:.4f}")
                if self._wandb_active and global_step % self.config.logging_steps == 0:
                    wandb.log({f"train/{k}": v for k, v in step_res.items()}, step=global_step)
            
            avg_train_loss = epoch_loss / steps if steps > 0 else 0
            print(f"  Train Loss: {avg_train_loss:.4f}")
            
            # --- Validation ---
            if self.val_loader:
                self.model_wrapper.model.eval()
                val_loss = 0.0
                val_steps = 0
                all_preds = []
                all_refs = []
                
                with torch.no_grad():
                    for batch in tqdm(self.val_loader, desc="Validation"):
                        step_res = self.eval_step(batch)
                        val_loss += step_res.get('loss', 0.0)
                        
                        # Собираем тексты для метрик
                        if 'preds' in step_res and 'refs' in step_res:
                            all_preds.extend(step_res['preds'])
                            all_refs.extend(step_res['refs'])
                            
                        val_steps += 1
                
                avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
                print(f"  Val Loss: {avg_val_loss:.4f}")
                
                log_data = {"val/loss": avg_val_loss, "epoch": epoch+1}
                
                # Считаем метрики (WER, CER и т.д.)
                if all_preds and self.metrics:
                    print("  Computing metrics...", end="\r")
                    metric_scores = self._compute_metrics(all_preds, all_refs)
                    for name, score in metric_scores.items():
                        print(f"  Val {name}: {score:.2f}%")
                        log_data[f"val/{name}"] = score
                        
                    # Show sample predictions
                    print("\n  Sample Predictions:")
                    for i in range(min(2, len(all_preds))):
                        print(f"    Ref: {all_refs[i]}")
                        print(f"    Hyp: {all_preds[i]}")
                
                if self._wandb_active:
                    wandb.log(log_data, step=global_step)
                
                # Checkpointing
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    save_path = os.path.join(self.config.output_dir, "best_model")
                    self.save_checkpoint(save_path)
                    print(f"  ⭐ New best model saved to {save_path}")
            
            # Epoch Checkpoint
            save_path = os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch+1}")
            self.save_checkpoint(save_path)
        
        print("✅ Training complete.")
        if self._wandb_active: wandb.finish()
        
        return self.model_wrapper
