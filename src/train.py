"""train.py — Stage 2/3: transfer-learning training + MLflow tracking + registry. 

Implement: build splits, train the ResNet18 head (CrossEntropy, Adam on trainable params,
early stop on val F1), log params/metrics/model to MLflow, register + promote to the
@production alias, evaluate on test, and save a clean-data reference baseline (image
features + embeddings) for drift monitoring.   Run: python -m src.train
"""
from __future__ import annotations

import json, random
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep, evaluate
from src.dataset import CastingDataset
from src.model import build_model, trainable_parameters, save_model, EmbeddingExtractor
from torch.utils.data import DataLoader


def set_seed(seed: int = config.RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _subsample(items, cap):
    """Stratified subsample to `cap` images (or return all if cap is falsy/too large)."""
    if not cap or cap <= 0 or cap >= len(items):
        return items
    grouped: dict[int, list] = {}
    for item in items:
        grouped.setdefault(item[1], []).append(item)
    rng = random.Random(config.RANDOM_SEED)
    result = []
    for cls, cls_items in grouped.items():
        n = max(1, int(cap * len(cls_items) / len(items)))
        shuffled = list(cls_items)
        rng.shuffle(shuffled)
        result.extend(shuffled[:n])
    rng.shuffle(result)
    return result


def class_weights(items) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = Counter(y for _, y in items)
    total = sum(counts.values())
    n_classes = len(counts)
    weights = [total / (n_classes * counts[i]) for i in range(n_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def save_reference_baseline(net, ref_items) -> dict:
    """Save reference_features.csv + reference_embeddings.npz for drift monitoring."""
    from PIL import Image as PILImage

    extractor = EmbeddingExtractor(net).to(config.DEVICE).eval()
    tf = data_prep.get_transforms(train=False)
    features_list, embeddings_list = [], []

    for path, label in ref_items:
        with PILImage.open(path) as img:
            features_list.append(data_prep.image_features(img))
            x = tf(img.convert("L"))
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            emb = extractor(x.unsqueeze(0).to(config.DEVICE))
            embeddings_list.append(emb.cpu().numpy())

    pd.DataFrame(features_list).to_csv(
        config.ARTIFACT_DIR / "reference_features.csv", index=False
    )
    np.savez(config.REFERENCE_EMBED, embeddings=np.vstack(embeddings_list))
    return {"n_reference": len(ref_items)}


def main() -> int:
    import mlflow, mlflow.pytorch
    from mlflow import MlflowClient

    set_seed()

    # ── Stage 1: data discovery + quality + splits ──────────────────────────
    root = data_prep.find_data_root()
    qc = data_prep.validate_quality(root)
    (config.ARTIFACT_DIR / "data_quality_report.json").write_text(json.dumps(qc, indent=2))
    split_info = data_prep.build_splits(root, "v1")

    # ── Load splits + subsample train ───────────────────────────────────────
    train_items = data_prep.load_split("v1", "train", root)
    val_items = data_prep.load_split("v1", "val", root)
    test_items = data_prep.load_split("v1", "test", root)

    train_items = _subsample(train_items, config.MAX_TRAIN_IMAGES)
    print(f"[train] {len(train_items)} images  [val] {len(val_items)}  [test] {len(test_items)}")

    # ── DataLoaders ─────────────────────────────────────────────────────────
    g = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_ds = CastingDataset(train_items, train=True)
    val_ds = CastingDataset(val_items, train=False)
    test_ds = CastingDataset(test_items, train=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, generator=g)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS)

    # ── Model + optimiser + loss ────────────────────────────────────────────
    net = build_model(freeze=config.FREEZE_BACKBONE).to(config.DEVICE)
    trainable = trainable_parameters(net)
    optimizer = torch.optim.Adam(trainable, lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    cw = class_weights(train_items).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)

    total_params = sum(p.numel() for p in net.parameters())
    train_params = sum(p.numel() for p in trainable)
    print(f"Parameters — total: {total_params:,}  trainable: {train_params:,}")

    # ── MLflow ──────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "backbone": config.BACKBONE,
            "freeze_backbone": config.FREEZE_BACKBONE,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "max_train_images": config.MAX_TRAIN_IMAGES,
            "val_split": config.VAL_SPLIT,
            "seed": config.RANDOM_SEED,
            "img_size": config.IMG_SIZE,
            "train_size": len(train_items),
            "val_size": len(val_items),
            "test_size": len(test_items),
        })

        # ── Training loop with early stopping ──────────────────────────────
        best_val_f1 = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(1, config.EPOCHS + 1):
            # — Train —
            net.train()
            running_loss = 0.0
            n_batches = 0
            for x, y in train_loader:
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                optimizer.zero_grad()
                logits = net(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1

            avg_loss = running_loss / max(n_batches, 1)

            # — Validate —
            y_true, y_pred, y_prob = evaluate.predict(net, val_loader)
            val_metrics = evaluate.compute_metrics(y_true, y_pred, y_prob)

            print(f"Epoch {epoch}/{config.EPOCHS}  loss={avg_loss:.4f}  "
                  f"val_f1={val_metrics['f1_defect']:.4f}  "
                  f"val_recall={val_metrics['recall_defect']:.4f}")

            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_defect": val_metrics["f1_defect"],
                "val_recall_defect": val_metrics["recall_defect"],
                "val_precision_defect": val_metrics["precision_defect"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_roc_auc": val_metrics["roc_auc"],
            }, step=epoch)

            # — Early stopping —
            if val_metrics["f1_defect"] > best_val_f1:
                best_val_f1 = val_metrics["f1_defect"]
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {epoch} (patience={config.EARLY_STOP_PATIENCE})")
                    break

        # Restore best weights
        if best_state is not None:
            net.load_state_dict(best_state)
        net.to(config.DEVICE)

        # ── Test evaluation ─────────────────────────────────────────────────
        y_true, y_pred, y_prob = evaluate.predict(net, test_loader)
        test_metrics = evaluate.compute_metrics(y_true, y_pred, y_prob)
        print(f"\n=== Test Metrics ===")
        for k, v in test_metrics.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v:.4f}")

        # Plots
        evaluate.plot_eval(y_true, y_pred, y_prob)
        failures = evaluate.failure_cases(test_items, y_true, y_pred, y_prob)

        mlflow.log_metrics({
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_defect": test_metrics["f1_defect"],
            "test_recall_defect": test_metrics["recall_defect"],
            "test_precision_defect": test_metrics["precision_defect"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_roc_auc": test_metrics["roc_auc"],
        })

        # ── Save model ──────────────────────────────────────────────────────
        save_model(net)
        model_meta = {
            "run_id": run.info.run_id,
            "backbone": config.BACKBONE,
            "freeze_backbone": config.FREEZE_BACKBONE,
            "epochs_trained": epoch,
            "best_val_f1": best_val_f1,
            "test_metrics": test_metrics,
            "total_params": total_params,
            "trainable_params": train_params,
        }
        config.MODEL_META_PATH.write_text(json.dumps(model_meta, indent=2))
        config.METRICS_PATH.write_text(json.dumps(test_metrics, indent=2))

        # ── MLflow: log model + register + @production ──────────────────────
        mlflow.pytorch.log_model(net, "model")

        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, config.REGISTERED_MODEL)

        client = MlflowClient()
        client.set_registered_model_alias(
            config.REGISTERED_MODEL,
            config.PRODUCTION_ALIAS,
            mv.version,
        )
        model_meta["version"] = mv.version
        config.MODEL_META_PATH.write_text(json.dumps(model_meta, indent=2))
        print(f"\n✓ Registered {config.REGISTERED_MODEL} v{mv.version} → @{config.PRODUCTION_ALIAS}")

        # ── Reference baseline for drift monitoring ─────────────────────────
        ref_items = val_items[:200]  # clean sample for reference
        ref_info = save_reference_baseline(net, ref_items)
        print(f"✓ Saved reference baseline ({ref_info['n_reference']} images)")

        # Log artifacts
        if (config.ARTIFACT_DIR / "model_eval.png").exists():
            mlflow.log_artifact(str(config.ARTIFACT_DIR / "model_eval.png"))
        if config.METRICS_PATH.exists():
            mlflow.log_artifact(str(config.METRICS_PATH))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
