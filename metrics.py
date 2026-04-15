"""
metrics.py  /  evaluate.py (combined)
──────────────────────────────────────
Extrinsic Evaluation
────────────────────
We measure model performance on a *held-out test set* (data the model has
never seen during training or hyper-parameter selection).

Metrics Reported
────────────────
  • Accuracy        – overall fraction correct
  • Precision       – of all predicted positives, how many were truly positive?
  • Recall          – of all actual positives, how many did we catch?
  • F1 Score        – harmonic mean of Precision and Recall (primary metric)
  • ROC-AUC         – area under the Receiver Operating Characteristic curve
  • MCC             – Matthews Correlation Coefficient (robust to class imbalance)
  • Confusion Matrix
  • Full Classification Report (per-class precision / recall / f1)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Core Metric Computation ──────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Compute all evaluation metrics given ground-truth labels and predictions.

    Args
    ────
    y_true  : list / array of true class indices
    y_pred  : list / array of predicted class indices
    y_proba : (optional) array of shape [N, num_classes] – class probabilities
              required for ROC-AUC computation

    Returns
    ───────
    dict of metric_name → value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc       = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1":        round(float(f1),   4),
        "mcc":       round(float(mcc),  4),
        "confusion_matrix": cm,
    }

    # ROC-AUC (requires probability scores for the positive class)
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba[:, 1])
            metrics["roc_auc"] = round(float(auc), 4)
        except Exception:
            pass

    return metrics


def format_report(metrics: dict, label_names=("Negative", "Positive")) -> str:
    """Pretty-print a metrics dict into a human-readable report."""
    cm = metrics.get("confusion_matrix", [])
    lines = [
        "╔══════════════════════════════════════╗",
        "║        EVALUATION  REPORT            ║",
        "╠══════════════════════════════════════╣",
        f"║  Accuracy  : {metrics['accuracy']:.4f}                  ║",
        f"║  Precision : {metrics['precision']:.4f}                  ║",
        f"║  Recall    : {metrics['recall']:.4f}                  ║",
        f"║  F1 Score  : {metrics['f1']:.4f}  ← primary metric    ║",
        f"║  MCC       : {metrics['mcc']:.4f}                  ║",
    ]
    if "roc_auc" in metrics:
        lines.append(f"║  ROC-AUC   : {metrics['roc_auc']:.4f}                  ║")
    if "loss" in metrics:
        lines.append(f"║  Test Loss : {metrics['loss']:.4f}                  ║")
    lines.append("╠══════════════════════════════════════╣")
    lines.append("║  Confusion Matrix                    ║")
    if cm:
        lines.append(f"║    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}              ║")
        lines.append(f"║    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}              ║")
    lines.append("╚══════════════════════════════════════╝")
    return "\n".join(lines)


# ── Full Evaluation on Test Set ──────────────────────────────────────────────

@torch.inference_mode()
def run_evaluation(cfg: dict, test_loader, device: torch.device):
    """
    Load the best saved checkpoint and run extrinsic evaluation
    on the held-out test set.

    Saves a JSON report to cfg['output']['report_path'].
    """
    import yaml
    sys.path.insert(0, str(Path(__file__).parent))
    from model import SentimentClassifier

    out_cfg   = cfg["output"]
    model_cfg = cfg["model"]

    # Load the best checkpoint
    model = SentimentClassifier.load(
        path=out_cfg["model_dir"],
        model_name=model_cfg["backbone"],
        num_labels=model_cfg["num_labels"],
        device=str(device),
    )
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_proba = [], [], []
    total_loss, n_batches = 0.0, 0

    for batch in tqdm(test_loader, desc="Evaluating on test set"):
        batch   = {k: v.to(device) for k, v in batch.items()}
        out     = model(**batch)
        proba   = torch.softmax(out["logits"], dim=-1).cpu().numpy()
        preds   = proba.argmax(axis=-1)
        labels  = batch["labels"].cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_proba.append(proba)
        if "loss" in out:
            total_loss += out["loss"].item()
            n_batches  += 1

    all_proba = np.vstack(all_proba)
    metrics   = compute_metrics(all_labels, all_preds, all_proba)
    if n_batches:
        metrics["loss"] = round(total_loss / n_batches, 4)

    # Per-class report
    report = classification_report(
        all_labels, all_preds,
        target_names=["Negative", "Positive"],
    )
    metrics["classification_report"] = report

    # Print & save
    print("\n" + format_report(metrics))
    print("\nPer-Class Report:\n" + report)

    report_path = out_cfg.get("report_path", "outputs/evaluation_report.json")
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(metrics, indent=2))
    logger.info(f"Evaluation report saved → {report_path}")

    return metrics


if __name__ == "__main__":
    # Quick unit test for compute_metrics
    y_true  = [0, 1, 1, 0, 1, 0, 1, 1]
    y_pred  = [0, 1, 0, 0, 1, 1, 1, 1]
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4],
                        [0.85, 0.15], [0.1, 0.9], [0.4, 0.6],
                        [0.05, 0.95], [0.15, 0.85]])

    m = compute_metrics(y_true, y_pred, y_proba)
    print(format_report(m))
