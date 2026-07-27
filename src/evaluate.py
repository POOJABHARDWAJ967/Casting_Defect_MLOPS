"""evaluate.py — Stage 3: evaluation metrics, plots, failure-case analysis. 

Positive class = DEFECT (recall on defects is the headline QC metric). Implement
prediction, imbalance-aware metrics, confusion/ROC plots, and a misclassified-sample list.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


@torch.no_grad()
def predict(net, loader, return_embeddings: bool = False):
    """Run inference over a DataLoader; return (y_true, y_pred, y_prob[, embeddings])."""
    from src.model import EmbeddingExtractor

    net.eval()
    device = next(net.parameters()).device
    y_true, y_pred, y_prob = [], [], []
    emb_list = []
    extractor = EmbeddingExtractor(net).to(device).eval() if return_embeddings else None

    for x, labels in loader:
        x = x.to(device)
        logits = net(x)
        probs = F.softmax(logits, dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(probs.argmax(dim=1).tolist())
        y_prob.extend(probs[:, config.POSITIVE_IDX].tolist())
        if return_embeddings and extractor is not None:
            emb_list.append(extractor(x).cpu().numpy())

    result = (np.array(y_true), np.array(y_pred), np.array(y_prob))
    if return_embeddings:
        result = result + (np.vstack(emb_list),)
    return result


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Imbalance-aware metrics with DEFECT as the positive class."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix,
    )

    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_defect": float(precision_score(
            y_true, y_pred, pos_label=config.POSITIVE_IDX, zero_division=0)),
        "recall_defect": float(recall_score(
            y_true, y_pred, pos_label=config.POSITIVE_IDX, zero_division=0)),
        "f1_defect": float(f1_score(
            y_true, y_pred, pos_label=config.POSITIVE_IDX, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def plot_eval(y_true, y_pred, y_prob, out: Path | None = None) -> Path:
    """Save a confusion-matrix + ROC figure to artifacts/model_eval.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    out = out or (config.ARTIFACT_DIR / "model_eval.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=config.CLASSES,
        ax=axes[0],
        cmap="Blues",
    )
    axes[0].set_title("Confusion Matrix")

    # ROC curve
    RocCurveDisplay.from_predictions(
        y_true, y_prob,
        pos_label=config.POSITIVE_IDX,
        name="Defect",
        ax=axes[1],
    )
    axes[1].set_title("ROC Curve (Defect class)")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def failure_cases(items, y_true, y_pred, y_prob, limit: int = 20) -> list[dict]:
    """List misclassified samples with predicted p_defect (error analysis)."""
    misclassified = []
    for i, (item, yt, yp, prob) in enumerate(zip(items, y_true, y_pred, y_prob)):
        if yt != yp:
            path = item[0] if isinstance(item, (tuple, list)) else item
            misclassified.append({
                "index": int(i),
                "path": str(path),
                "true_label": config.IDX_TO_CLASS.get(int(yt), str(yt)),
                "pred_label": config.IDX_TO_CLASS.get(int(yp), str(yp)),
                "prob_defect": float(prob),
            })
    # Sort by confidence in the wrong answer (most confident errors first)
    misclassified.sort(key=lambda x: abs(x["prob_defect"] - 0.5), reverse=True)
    return misclassified[:limit]
