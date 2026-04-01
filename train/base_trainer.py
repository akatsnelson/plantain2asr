import os
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from tqdm.auto import tqdm

try:
    import torch
except ImportError:
    torch = None

from ..models.base import BaseASRModel
from ..dataloaders.base import BaseASRDataset
from ..metrics.base import BaseMetric
from ..utils.logging import get_logger
from .config import TrainingConfig

# Suppress verbose warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
warnings.filterwarnings("ignore", message=".*Triton.*")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

try:
    import wandb
except ImportError:
    wandb = None

logger = get_logger(__name__)


class BaseTrainer(ABC):
    """
    Базовый класс для обучения ASR моделей.

    Главное правило: trainer остается обычным Processor, поэтому
    `dataset >> trainer` всегда возвращает dataset, а обученная модель
    доступна через `trainer.trained_model`.
    """

    def __init__(
        self,
        model: BaseASRModel,
        config: TrainingConfig,
        eval_dataset: Optional[BaseASRDataset] = None,
        metrics: Optional[List[BaseMetric]] = None,
    ):
        if torch is None:
            raise ImportError(
                "Training requires PyTorch. Install plantain2asr with the appropriate "
                "train/model extras before using the train API."
            )

        self.model_wrapper = model
        self.config = config
        self.eval_dataset = eval_dataset
        self.metrics = metrics or []
        self.device = self._resolve_device()

        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.processor = None
        self.train_model = None
        self.trained_model: Optional[BaseASRModel] = None
        self.last_train_dataset: Optional[BaseASRDataset] = None
        self.training_summary: Dict[str, Any] = {}
        self.last_checkpoint_path: Optional[str] = None
        self._wandb_active = False

    @staticmethod
    def _resolve_device() -> str:
        from ..utils.device import auto_select_device

        return auto_select_device()

    def _warn_config_mismatches(self) -> None:
        ignored = []
        if self.config.gradient_accumulation_steps != 1:
            ignored.append("gradient_accumulation_steps")
        if self.config.fp16:
            ignored.append("fp16")
        if self.config.warmup_steps:
            ignored.append("warmup_steps")
        if self.config.group_by_length:
            ignored.append("group_by_length")
        if self.config.gradient_checkpointing:
            ignored.append("gradient_checkpointing")
        if self.config.save_steps:
            ignored.append("save_steps")
        if self.config.eval_steps:
            ignored.append("eval_steps")
        if self.config.save_total_limit:
            ignored.append("save_total_limit")

        if ignored:
            logger.warning(
                "⚠️ TrainingConfig contains fields that are currently informational only "
                f"for this custom trainer: {', '.join(ignored)}"
            )

    @abstractmethod
    def prepare_training(self, train_dataset: BaseASRDataset):
        """Подготовка к обучению: dataloaders, optimizer и train-capable model."""

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Один шаг обучения. Должен вернуть хотя бы {'loss': float}."""

    @abstractmethod
    def eval_step(self, batch: Any) -> Dict[str, Any]:
        """
        Один шаг валидации.
        Должен возвращать {'loss': float, 'preds': List[str], 'refs': List[str]}.
        """

    @abstractmethod
    def save_checkpoint(self, path: str):
        """Сохранение текущего состояния обучаемой модели."""

    def _compute_metrics(self, preds: List[str], refs: List[str]) -> Dict[str, float]:
        results = {}
        for metric in self.metrics:
            try:
                if hasattr(metric, "calculate_batch"):
                    score = metric.calculate_batch(refs, preds)
                else:
                    scores = [metric.calculate(r, p) for r, p in zip(refs, preds)]
                    score = sum(scores) / len(scores) if scores else 0.0
                results[metric.name] = score
            except Exception as exc:
                logger.warning("Metric calculation failed for %s: %s", metric.name, exc)
        return results

    def _setup_wandb(self) -> None:
        if not self.config.project_name or wandb is None:
            return

        try:
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=self.config.__dict__,
            )
            self._wandb_active = True
            logger.info("WandB initialized")
        except Exception as exc:
            logger.warning("WandB init failed, continuing offline: %s", exc)
            self._wandb_active = False

    def _active_model(self):
        return self.train_model or getattr(self.model_wrapper, "model", None)

    def fit(self, train_dataset: BaseASRDataset) -> BaseASRModel:
        logger.info("Starting training for %s on %s", self.model_wrapper.name, self.device)
        self.last_train_dataset = train_dataset
        self._warn_config_mismatches()
        self._setup_wandb()

        best_loss = float("inf")
        best_model_path = None
        global_step = 0

        try:
            self.prepare_training(train_dataset)
            active_model = self._active_model()
            if active_model is None:
                raise RuntimeError(
                    "Trainer preparation did not provide a trainable model. "
                    "prepare_training() must populate either self.train_model or model_wrapper.model."
                )

            if hasattr(active_model, "to"):
                active_model.to(self.device)

            for epoch in range(self.config.num_train_epochs):
                logger.info("Epoch %s/%s", epoch + 1, self.config.num_train_epochs)

                if hasattr(active_model, "train"):
                    active_model.train()

                train_pbar = tqdm(self.train_loader, desc="Training")
                epoch_loss = 0.0
                steps = 0

                for batch in train_pbar:
                    step_res = self.train_step(batch)
                    loss = float(step_res.get("loss", 0.0))
                    epoch_loss += loss
                    steps += 1
                    global_step += 1

                    train_pbar.set_postfix(loss=f"{loss:.4f}")
                    if self._wandb_active and global_step % max(1, self.config.logging_steps) == 0:
                        wandb.log({f"train/{k}": v for k, v in step_res.items()}, step=global_step)

                avg_train_loss = epoch_loss / steps if steps else 0.0
                self.training_summary["last_train_loss"] = avg_train_loss
                logger.info("Train loss: %.4f", avg_train_loss)

                if self.val_loader:
                    if hasattr(active_model, "eval"):
                        active_model.eval()

                    val_loss = 0.0
                    val_steps = 0
                    all_preds = []
                    all_refs = []

                    with torch.no_grad():
                        for batch in tqdm(self.val_loader, desc="Validation"):
                            step_res = self.eval_step(batch)
                            val_loss += float(step_res.get("loss", 0.0))
                            all_preds.extend(step_res.get("preds", []))
                            all_refs.extend(step_res.get("refs", []))
                            val_steps += 1

                    avg_val_loss = val_loss / val_steps if val_steps else 0.0
                    self.training_summary["last_val_loss"] = avg_val_loss
                    logger.info("Validation loss: %.4f", avg_val_loss)

                    log_data = {"val/loss": avg_val_loss, "epoch": epoch + 1}
                    if all_preds and self.metrics:
                        metric_scores = self._compute_metrics(all_preds, all_refs)
                        for name, score in metric_scores.items():
                            logger.info("Validation %s: %.2f%%", name, score)
                            log_data[f"val/{name}"] = score

                    if self._wandb_active:
                        wandb.log(log_data, step=global_step)

                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        best_model_path = os.path.join(self.config.output_dir, "best_model")
                        self.save_checkpoint(best_model_path)
                        self.last_checkpoint_path = best_model_path
                        logger.info("New best model saved to %s", best_model_path)

                epoch_checkpoint = os.path.join(
                    self.config.output_dir,
                    f"checkpoint-epoch-{epoch + 1}",
                )
                self.save_checkpoint(epoch_checkpoint)
                self.last_checkpoint_path = epoch_checkpoint

            self.trained_model = self.model_wrapper
            self.training_summary.update(
                {
                    "epochs": self.config.num_train_epochs,
                    "global_steps": global_step,
                    "best_model_path": best_model_path,
                }
            )
            logger.info("Training complete")
            return self.trained_model
        finally:
            if self._wandb_active:
                wandb.finish()
                self._wandb_active = False

    def apply_to(self, train_dataset: BaseASRDataset) -> BaseASRDataset:
        """
        Processor API: запускает обучение как сайд-эффект и возвращает dataset,
        чтобы `dataset >> trainer >> ...` не ломал pipeline-контракт.
        """
        self.fit(train_dataset)
        return train_dataset
