"""
visualize.py
────────────
Plot training history and evaluation results.
  • Loss curves (train vs val)
  • Metric curves (F1, accuracy, MCC)
  • Confusion matrix heatmap
  • Precision-Recall curve
  • ROC curve
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

logger = logging.getLogger(__name__)
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def plot_training_history(history_path: str, out_dir: str = "outputs"):
    """Load training_history.json and plot loss + metric curves."""
    history = json.loads(Path(history_path).read_text())
    epochs  = [h["epoch"]      for h in history]
    t_loss  = [h["train_loss"] for h in history]
    v_loss  = [h.get("loss", 0) for h in history]
    f1      = [h["f1"]         for h in history]
    acc     = [h["accuracy"]   for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, t_loss, "o-", color=COLORS[0], label="Train Loss")
    ax.plot(epochs, v_loss, "s--", color=COLORS[1], label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curves"); ax.legend(); ax.grid(alpha=0.3)

    # Metrics
    ax = axes[1]
    ax.plot(epochs, f1,  "o-", color=COLORS[2], label="F1")
    ax.plot(epochs, acc, "s--", color=COLORS[3], label="Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Validation Metrics"); ax.legend(); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    out = Path(out_dir) / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


def plot_confusion_matrix(cm: list, out_dir: str = "outputs"):
    """Plot a 2×2 confusion matrix heatmap."""
    labels = ["Negative", "Positive"]
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    out = Path(out_dir) / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


def plot_pr_roc_curves(y_true, y_proba, out_dir: str = "outputs"):
    """Plot Precision-Recall and ROC curves side by side."""
    from sklearn.metrics import (
        precision_recall_curve, roc_curve, auc, average_precision_score
    )

    pos_proba = y_proba[:, 1]
    prec, rec, _  = precision_recall_curve(y_true, pos_proba)
    ap            = average_precision_score(y_true, pos_proba)
    fpr, tpr, _   = roc_curve(y_true, pos_proba)
    roc_auc       = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Model Evaluation Curves", fontsize=14, fontweight="bold")

    # PR curve
    ax = axes[0]
    ax.plot(rec, prec, color=COLORS[0], lw=2, label=f"AP = {ap:.3f}")
    ax.axhline(y=sum(y_true)/len(y_true), color="gray", linestyle="--", label="Baseline")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend(); ax.grid(alpha=0.3)

    # ROC curve
    ax = axes[1]
    ax.plot(fpr, tpr, color=COLORS[2], lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(out_dir) / "pr_roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")

def run_all_plots(report_path: str, history_path: str = None, out_dir: str = "outputs"):
    """Generate all plots from saved evaluation artefacts."""
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    report_file = Path(report_path)

    # ─────────────────────────────
    # SAFE LOAD REPORT (FIX)
    # ─────────────────────────────
    if not report_file.exists():
        logger.warning(f"⚠️ Report not found: {report_path}")
        logger.warning("Skipping evaluation plots.")
        report = None
    else:
        report = json.loads(report_file.read_text())

    # ─────────────────────────────
    # CONFUSION MATRIX
    # ─────────────────────────────
    if report and report.get("confusion_matrix"):
        plot_confusion_matrix(report["confusion_matrix"], out_dir)
    else:
        logger.warning("No confusion matrix found.")

    # ─────────────────────────────
    # TRAINING HISTORY
    # ─────────────────────────────
    if history_path:
        history_file = Path(history_path)

        if history_file.exists():
            plot_training_history(history_path, out_dir)
        else:
            logger.warning(f"⚠️ Training history not found: {history_path}")

    logger.info("All available plots saved.")