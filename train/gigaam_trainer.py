import os
import re
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..dataloaders.base import BaseASRDataset
from .dataset import CTCCharTokenizer, CharCTCAudioDataset, ctc_audio_collate_fn
from ..utils.logging import get_logger

logger = get_logger(__name__)

class GigaAMTrainer(BaseTrainer):
    def _extract_ctc_vocab(self) -> Tuple[List[str], Optional[int]]:
        """
        Extract vocabulary and blank id from GigaAM config if available.
        In GigaAM-v3-ctc checkpoints, vocab length is often num_classes-1 (blank is extra class).
        """
        vocab: Optional[List[str]] = None
        num_classes: Optional[int] = None

        def deep_get(obj: Any, path: List[str]):
            cur = obj
            for key in path:
                if cur is None:
                    return None
                if isinstance(cur, dict):
                    cur = cur.get(key)
                else:
                    cur = getattr(cur, key, None)
            return cur

        cfg = getattr(self.model_wrapper.model, "config", None)
        cfg_root = getattr(cfg, "cfg", None) if cfg is not None else None

        # try dict/attr paths
        vocab = (
            deep_get(cfg_root, ["model", "cfg", "decoding", "vocabulary"])
            or deep_get(cfg_root, ["cfg", "model", "cfg", "decoding", "vocabulary"])
        )
        num_classes = (
            deep_get(cfg_root, ["model", "cfg", "head", "num_classes"])
            or deep_get(cfg_root, ["cfg", "model", "cfg", "head", "num_classes"])
        )

        # fallback: try inner model cfg
        inner = getattr(self.model_wrapper, "_inner_model", None)
        vocab = vocab or deep_get(inner, ["cfg", "decoding", "vocabulary"]) or deep_get(inner, ["cfg", "cfg", "decoding", "vocabulary"])
        num_classes = num_classes or deep_get(inner, ["cfg", "head", "num_classes"]) or deep_get(inner, ["cfg", "cfg", "head", "num_classes"])

        if not vocab:
            # safe fallback: space + russian alphabet (no ё)
            vocab = [" "] + list("абвгдежзийклмнопрстуфхцчшщъыьэюя")

        blank_id = None
        if isinstance(num_classes, int) and num_classes == len(vocab) + 1:
            blank_id = len(vocab)
        return vocab, blank_id

    @staticmethod
    def _loss_to_float(loss: Any) -> float:
        if hasattr(loss, "item"):
            return float(loss.item())
        return float(loss)

    def _require_ctc_backend(self) -> None:
        if getattr(self.model_wrapper, "training_backend", None) != "ctc":
            raise NotImplementedError(self.model_wrapper.training_not_supported_reason())

    def prepare_training(self, train_dataset: BaseASRDataset):
        logger.info("Preparing GigaAM training components")
        self._require_ctc_backend()
        self.train_model, self.processor, self.data_collator = self.model_wrapper.get_training_components()

        vocab, blank_id = self._extract_ctc_vocab()
        self.ctc_tokenizer = CTCCharTokenizer(vocab, blank_id=blank_id)
        self.tokenizer = self.ctc_tokenizer
        self.blank_id = self.ctc_tokenizer.blank_id
        logger.info(
            "Tokenizer ready: CTCCharTokenizer(vocab=%s, blank_id=%s)",
            len(vocab),
            self.ctc_tokenizer.blank_id,
        )

        self.train_ds_torch = CharCTCAudioDataset(train_dataset, self.tokenizer)
        self.train_loader = DataLoader(
            self.train_ds_torch,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator or ctc_audio_collate_fn,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=(self.device == "cuda"),
        )

        if self.eval_dataset:
            self.val_ds_torch = CharCTCAudioDataset(self.eval_dataset, self.tokenizer)
            self.val_loader = DataLoader(
                self.val_ds_torch,
                batch_size=self.config.per_device_eval_batch_size,
                collate_fn=self.data_collator or ctc_audio_collate_fn,
                num_workers=self.config.dataloader_num_workers,
            )

        self.optimizer = torch.optim.AdamW(
            self.train_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        inner = self.model_wrapper._inner_model if hasattr(self.model_wrapper, "_inner_model") else self.train_model
        if hasattr(inner, "freeze_feature_encoder") and not getattr(self, "_feature_encoder_frozen", False):
            inner.freeze_feature_encoder()
            self._feature_encoder_frozen = True
            logger.info("Feature encoder frozen")

    def train_step(self, batch: Any) -> Dict[str, float]:
        if batch is None:
            return {"loss": 0.0}

        input_values = batch["input_values"].to(self.device)
        labels = batch["labels"].to(self.device)

        self.optimizer.zero_grad()
        outputs = self.train_model(input_values=input_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        return {"loss": self._loss_to_float(loss)}

    def eval_step(self, batch: Any) -> Dict[str, Any]:
        if batch is None:
            return {"loss": 0.0, "preds": [], "refs": []}

        input_values = batch["input_values"].to(self.device)
        labels = batch["labels"].to(self.device)
        reference_texts = batch["reference_texts"]

        with torch.no_grad():
            outputs = self.train_model(input_values=input_values, labels=labels)

        loss = self._loss_to_float(outputs.loss)
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, tuple) and len(outputs) > 1:
            logits = outputs[1]

        preds: List[str] = []
        if logits is None or logits.numel() == 0:
            preds = [""] * len(reference_texts)
        else:
            pred_ids = torch.argmax(logits, dim=-1)
            for ids in pred_ids.tolist():
                collapsed = [token for idx, token in enumerate(ids) if idx == 0 or token != ids[idx - 1]]
                collapsed = [token for token in collapsed if token != self.blank_id]
                preds.append(self.tokenizer.decode(collapsed))

        return {"loss": loss, "preds": preds, "refs": reference_texts}

    def save_checkpoint(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.train_model.save_pretrained(path)
        if hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(path)
        elif hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(path)
        logger.info("Checkpoint saved: %s", path)
