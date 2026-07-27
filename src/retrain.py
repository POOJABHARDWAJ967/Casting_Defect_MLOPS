"""retrain.py — Stage 4: drift-triggered retraining, version compare, rollback.

Implement: read drift_summary.json; if retraining is recommended, measure the production
model on a drifted batch, train a drift-augmented candidate, compare, then PROMOTE the
candidate to @production only if it improves (within PROMOTE_EPSILON) else ROLL BACK.
Record retraining_decision.json + manage MLflow registry versions.   Run: python -m src.retrain
"""
from __future__ import annotations

import json, random
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep, evaluate
from src.model import build_model, trainable_parameters, load_model, save_model
from src.monitoring import corrupt
from src.train import _subsample, set_seed, class_weights


class DriftAugmentedDataset(Dataset):
    """Dataset that corrupts a fraction of images to simulate drift-augmented training."""
    def __init__(self, items: list[tuple[Path, int]], corrupt_frac: float = 0.4):
        self.items = items
        self.corrupt_frac = corrupt_frac
        self.tf = data_prep.get_transforms(train=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        with Image.open(path) as im:
            img = im.convert("L")
            # Randomly corrupt some images
            if random.random() < self.corrupt_frac:
                img = corrupt(img)
            x = self.tf(img)
        return x, label


class CorruptedEvalDataset(Dataset):
    """Dataset that corrupts all images for drifted-batch evaluation."""
    def __init__(self, items: list[tuple[Path, int]], corrupt_frac: float = 1.0):
        self.items = items
        self.corrupt_frac = corrupt_frac
        self.tf = data_prep.get_transforms(train=False)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        with Image.open(path) as im:
            img = im.convert("L")
            if random.random() < self.corrupt_frac:
                img = corrupt(img)
            x = self.tf(img)
        return x, label


def main() -> int:
    import mlflow, mlflow.pytorch
    from mlflow import MlflowClient

    set_seed()

    drift = json.loads((config.ARTIFACT_DIR / "drift_summary.json").read_text())

    if not drift.get("retrain_recommended"):
        # No retrain needed — record the decision
        decision = {
            "action": "no_retrain",
            "reason": "Drift summary does not recommend retraining",
            "drift_summary": drift,
        }
        (config.ARTIFACT_DIR / "retraining_decision.json").write_text(
            json.dumps(decision, indent=2)
        )
        print("✓ No retraining needed. Decision recorded.")
        return 0

    print("=== Drift-Triggered Retraining ===")

    # ── Load data ───────────────────────────────────────────────────────────
    root = data_prep.find_data_root()
    train_items = data_prep.load_split("v1", "train", root)
    val_items = data_prep.load_split("v1", "val", root)
    test_items = data_prep.load_split("v1", "test", root)

    # Subsample for fast retraining
    train_items = _subsample(train_items, min(config.MAX_TRAIN_IMAGES, 1500))

    # ── Evaluate current production on a fully-drifted batch ────────────────
    print("\n── Evaluating production model on drifted batch ──")
    prod_net = load_model()
    prod_net.to(config.DEVICE).eval()

    drifted_ds = CorruptedEvalDataset(test_items, corrupt_frac=1.0)
    drifted_loader = DataLoader(drifted_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                                num_workers=config.NUM_WORKERS)

    y_true, y_pred, y_prob = evaluate.predict(prod_net, drifted_loader)
    prod_metrics = evaluate.compute_metrics(y_true, y_pred, y_prob)
    prod_f1 = prod_metrics["f1_defect"]
    print(f"  Production F1 on drifted batch: {prod_f1:.4f}")

    # ── Train drift-augmented candidate ─────────────────────────────────────
    print("\n── Training drift-augmented candidate ──")
    cand_net = build_model(freeze=config.FREEZE_BACKBONE).to(config.DEVICE)
    trainable = trainable_parameters(cand_net)
    optimizer = torch.optim.Adam(trainable, lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    cw = class_weights(train_items).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # Drift-augmented training (corrupt ~40% of images)
    drift_ds = DriftAugmentedDataset(train_items, corrupt_frac=0.4)
    g = torch.Generator().manual_seed(config.RANDOM_SEED)
    drift_train_loader = DataLoader(drift_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                                     num_workers=config.NUM_WORKERS, generator=g)

    val_eval_ds = CorruptedEvalDataset(val_items, corrupt_frac=0.5)
    val_eval_loader = DataLoader(val_eval_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                                  num_workers=config.NUM_WORKERS)

    retrain_epochs = min(config.EPOCHS, 3)  # fewer epochs for retraining
    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, retrain_epochs + 1):
        cand_net.train()
        running_loss = 0.0
        n_batches = 0
        for x, y in drift_train_loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            logits = cand_net(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)

        # Validate on mixed-drift val set
        y_true_v, y_pred_v, y_prob_v = evaluate.predict(cand_net, val_eval_loader)
        val_m = evaluate.compute_metrics(y_true_v, y_pred_v, y_prob_v)
        print(f"  Epoch {epoch}/{retrain_epochs}  loss={avg_loss:.4f}  val_f1={val_m['f1_defect']:.4f}")

        if val_m["f1_defect"] > best_val_f1:
            best_val_f1 = val_m["f1_defect"]
            best_state = {k: v.cpu().clone() for k, v in cand_net.state_dict().items()}

    if best_state is not None:
        cand_net.load_state_dict(best_state)
    cand_net.to(config.DEVICE)

    # ── Compare candidate vs production on drifted batch ────────────────────
    print("\n── Comparing candidate vs production on drifted batch ──")
    y_true_c, y_pred_c, y_prob_c = evaluate.predict(cand_net, drifted_loader)
    cand_metrics = evaluate.compute_metrics(y_true_c, y_pred_c, y_prob_c)
    cand_f1 = cand_metrics["f1_defect"]
    print(f"  Production F1: {prod_f1:.4f}")
    print(f"  Candidate F1:  {cand_f1:.4f}")

    # ── Promote or rollback ─────────────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    client = MlflowClient()

    if cand_f1 >= prod_f1 - config.PROMOTE_EPSILON:
        action = "promote"
        print(f"\n✓ PROMOTE: candidate ({cand_f1:.4f}) >= prod ({prod_f1:.4f}) - ε ({config.PROMOTE_EPSILON})")

        # Save + register candidate
        save_model(cand_net)

        with mlflow.start_run() as run:
            mlflow.log_params({
                "retrain": True,
                "corrupt_frac": 0.4,
                "epochs": retrain_epochs,
            })
            mlflow.log_metrics({
                "drifted_f1_defect": cand_f1,
                "prod_f1_defect": prod_f1,
            })
            mlflow.pytorch.log_model(cand_net, "model")

            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, config.REGISTERED_MODEL)

            client.set_registered_model_alias(
                config.REGISTERED_MODEL,
                config.PRODUCTION_ALIAS,
                mv.version,
            )
            new_version = mv.version

        # Update model meta
        model_meta = json.loads(config.MODEL_META_PATH.read_text())
        model_meta["version"] = new_version
        model_meta["retrained"] = True
        model_meta["drifted_f1"] = cand_f1
        config.MODEL_META_PATH.write_text(json.dumps(model_meta, indent=2))

    else:
        action = "rollback"
        new_version = None
        print(f"\n✗ ROLLBACK: candidate ({cand_f1:.4f}) < prod ({prod_f1:.4f}) - ε ({config.PROMOTE_EPSILON})")
        print("  Keeping incumbent production model.")

    # ── Record decision ─────────────────────────────────────────────────────
    decision = {
        "action": action,
        "production_f1_drifted": prod_f1,
        "candidate_f1_drifted": cand_f1,
        "promote_epsilon": config.PROMOTE_EPSILON,
        "candidate_version": new_version,
        "retrain_epochs": retrain_epochs,
        "corrupt_frac_train": 0.4,
        "drift_triggers": {
            "statistical": drift.get("statistical_drift", {}).get("drifted", False),
            "embedding": drift.get("embedding_drift", {}).get("drifted", False),
            "confidence": drift.get("confidence", {}).get("drifted", False),
        },
    }
    (config.ARTIFACT_DIR / "retraining_decision.json").write_text(
        json.dumps(decision, indent=2)
    )
    print(f"\n✓ Decision recorded: {action}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
