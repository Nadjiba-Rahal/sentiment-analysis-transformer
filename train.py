"""
train.py
────────
Fine-Tuning Pipeline (Discriminative Training)

Strategy
────────
  • Discriminative Learning Rates: lower LR for early (general) layers,
    higher LR for later (task-specific) layers and the classifier head.
    This prevents catastrophic forgetting of pre-trained knowledge.
  • Linear Warm-Up + Linear Decay scheduler (standard for BERT fine-tuning).
  • Gradient clipping to stabilise training.
  • Mixed Precision (fp16) for faster GPU training.
  • Early Stopping on validation F1 to avoid over-fitting.
  • Best checkpoint is saved automatically.
"""

import os
import sys
import json
import yaml
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from model   import SentimentClassifier
from metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Discriminative Learning Rate Groups ─────────────────────────────────────

def get_layers(model):
    # BERT
    if hasattr(model, "encoder"):
        return model.encoder.layer

    # DistilBERT (your case)
    if hasattr(model, "distilbert"):
        return model.distilbert.transformer.layer

    raise ValueError("Unsupported model architecture")

def get_discriminative_param_groups(model, base_lr):

    no_decay = {"bias", "LayerNorm.weight"}

    # Get the encoder module (works for both BERT and DistilBERT)
    if hasattr(model, "encoder"):
        encoder = model.encoder
    elif hasattr(model, "distilbert"):
        encoder = model.distilbert
    else:
        raise ValueError("Unsupported model architecture: no 'encoder' or 'distilbert' attribute")

    # Get embeddings and layers based on model type
    if hasattr(encoder, "embeddings"):
        embedding_module = encoder.embeddings
    else:
        raise ValueError("No embeddings found in encoder")

    # DistilBERT uses transformer.layer, BERT uses encoder.layer
    if hasattr(encoder, "transformer"):
        layers = encoder.transformer.layer
    elif hasattr(encoder, "encoder"):
        layers = encoder.encoder.layer
    else:
        raise ValueError("No layers found in encoder")


    n_layers = len(layers)

    groups = []

    # embeddings
    groups += [
        {
            "params": [p for n, p in embedding_module.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "lr": base_lr * 0.1,
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in embedding_module.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "lr": base_lr * 0.1,
            "weight_decay": 0.0,
        },
    ]

    # transformer layers
    for i, layer in enumerate(layers):
        layer_lr = base_lr * (0.1 + 0.9 * i / max(n_layers - 1, 1))

        groups += [
            {
                "params": [p for n, p in layer.named_parameters()
                           if p.requires_grad and not any(nd in n for nd in no_decay)],
                "lr": layer_lr,
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in layer.named_parameters()
                           if p.requires_grad and any(nd in n for nd in no_decay)],
                "lr": layer_lr,
                "weight_decay": 0.0,
            },
        ]

    # classifier
    groups += [
        {
            "params": [p for n, p in model.classifier.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "lr": base_lr,
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.classifier.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "lr": base_lr,
            "weight_decay": 0.0,
        },
    ]

    return groups

# ── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: dict, model: SentimentClassifier,
                 train_loader, val_loader, device: torch.device):
        self.cfg          = cfg
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.tcfg         = cfg["training"]
        self.out_dir      = cfg["output"]["model_dir"]

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(cfg["output"]["logs_dir"], exist_ok=True)

        # ── Optimizer (Discriminative LRs) ───────────────────────────────────
        param_groups = get_discriminative_param_groups(
            model, self.tcfg["learning_rate"]
        )
        self.optimizer = AdamW(param_groups)

        # ── LR Scheduler (Warm-Up + Decay) ───────────────────────────────────
        total_steps  = len(train_loader) * self.tcfg["epochs"]
        warmup_steps = int(total_steps * self.tcfg["warmup_ratio"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # ── Mixed Precision ───────────────────────────────────────────────────
        self.scaler  = GradScaler(enabled=self.tcfg.get("fp16", False) and device.type == "cuda")
        self.use_fp16 = self.tcfg.get("fp16", False) and device.type == "cuda"

        # ── Training state ────────────────────────────────────────────────────
        self.best_metric = -float("inf")
        self.best_epoch  = 0
        self.history     = []

    # ── Single Training Epoch ────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]", leave=False)

        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_fp16):
                out  = self.model(**batch)
                loss = out["loss"]

            self.scaler.scale(loss).backward()

            # Gradient clipping – prevents exploding gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.tcfg["gradient_clip"]
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / n_batches

    # ── Validation Pass ──────────────────────────────────────────────────────

    @torch.inference_mode()
    def _evaluate(self, loader, split: str = "val") -> dict:
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss, n_batches = 0.0, 0

        for batch in tqdm(loader, desc=f"  [{split}]", leave=False):
            batch  = {k: v.to(self.device) for k, v in batch.items()}
            out    = self.model(**batch)
            preds  = out["logits"].argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            if "loss" in out:
                total_loss += out["loss"].item()
                n_batches  += 1

        metrics = compute_metrics(all_labels, all_preds)
        if n_batches:
            metrics["loss"] = total_loss / n_batches
        return metrics

    # ── Main Training Loop ───────────────────────────────────────────────────

    def train(self):
        logger.info(
            f"Starting fine-tuning for {self.tcfg['epochs']} epochs | "
            f"device={self.device} | fp16={self.use_fp16}"
        )
        logger.info(
            f"Total params: {self.model.num_parameters():,} | "
            f"Trainable: {self.model.num_parameters(trainable_only=True):,}"
        )

        save_metric = self.tcfg.get("save_best_metric", "f1")
        patience    = self.tcfg.get("early_stopping_patience", 3)
        epochs_no_improve = 0

        for epoch in range(1, self.tcfg["epochs"] + 1):
            t0 = time.time()

            train_loss = self._train_epoch(epoch)
            val_metrics = self._evaluate(self.val_loader, "val")

            elapsed = time.time() - t0
            current = val_metrics.get(save_metric, val_metrics.get("f1", 0))

            log_line = (
                f"Epoch {epoch}/{self.tcfg['epochs']} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics.get('loss', 0):.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"time={elapsed:.1f}s"
            )
            logger.info(log_line)

            row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            self.history.append(row)

            # ── Save Best Checkpoint ─────────────────────────────────────────
            if current > self.best_metric:
                self.best_metric = current
                self.best_epoch  = epoch
                epochs_no_improve = 0
                self.model.save(self.out_dir)
                logger.info(f"  ✔ New best {save_metric}={current:.4f} → saved checkpoint")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

        logger.info(
            f"\nTraining complete. Best epoch: {self.best_epoch} | "
            f"Best {save_metric}: {self.best_metric:.4f}"
        )

        # Save training history
        logs_path = Path(self.cfg["output"]["logs_dir"]) / "training_history.json"
        logs_path.write_text(json.dumps(self.history, indent=2))
        return self.history


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Imports here to avoid circular deps in standalone testing
    from preprocess import build_preprocessing_pipeline
    from dataset    import build_dataloaders

    normalizer, tokenizer = build_preprocessing_pipeline(config_path)
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, normalizer, tokenizer
    )

    model = SentimentClassifier(
        model_name=cfg["model"]["backbone"],
        num_labels=cfg["model"]["num_labels"],
        classifier_dropout=cfg["model"]["classifier_dropout"],
    )

    trainer = Trainer(cfg, model, train_loader, val_loader, device)
    trainer.train()

    logger.info("Running final evaluation on held-out test set …")
    from evaluate import run_evaluation
    run_evaluation(cfg, test_loader, device)


if __name__ == "__main__":
    main()
