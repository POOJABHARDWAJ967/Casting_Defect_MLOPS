"""monitoring.py — Stage 4: statistical + embedding drift + confidence monitoring.

Implement the three drift signals against the clean reference baseline:
  1. statistical drift  — Evidently DataDriftPreset + PSI on image features
  2. embedding drift    — PSI on ResNet-embedding distance-to-centroid distribution
  3. confidence         — mean predicted confidence reference vs current
Use a corrupted copy of clean images as the simulated "current" production batch.
Outputs drift_report.html + drift_summary.json.   Run: python -m src.monitoring

Embedding drift (TODO 4) — see the conceptual walkthrough in
Operations_Monitoring_and_Evidence.ipynb (Stage 4.3):
  1. feature extraction  — penultimate 512-dim ResNet embedding (model.EmbeddingExtractor)
  2. embedding generation — embeddings for reference + current batches
  3. feature-space compare — reduce each to distance-to-reference-centroid (one distribution per batch)
  4. drift calculation   — PSI between the two distance distributions (> ~0.10 => drifted)
"""
from __future__ import annotations

import json, random
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep


def _lazy_model_import():
    try:
        from src.model import load_model, EmbeddingExtractor
    except ImportError as exc:
        raise ImportError(
            "Monitoring requires src.model and PyTorch. Install the project dependencies before running monitoring."
        ) from exc
    return load_model, EmbeddingExtractor


def psi(reference, current, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cur = np.asarray(current, dtype=np.float64).ravel()
    if ref.size == 0 or cur.size == 0:
        return 0.0

    quantiles = np.linspace(0.0, 100.0, bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = ref_counts.astype(np.float64) / ref_counts.sum()
    cur_pct = cur_counts.astype(np.float64) / cur_counts.sum()

    eps = 1e-8
    ref_pct = np.clip(ref_pct, eps, 1.0)
    cur_pct = np.clip(cur_pct, eps, 1.0)

    return float(np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct)))


def corrupt(img: Image.Image) -> Image.Image:
    """Simulate camera/lighting drift (brightness/blur/rotate/noise via config.DRIFT_SIM)."""
    sim = config.DRIFT_SIM

    # Brightness degradation
    img = ImageEnhance.Brightness(img).enhance(sim["brightness"])

    # Gaussian blur
    img = img.filter(ImageFilter.GaussianBlur(radius=sim["blur_radius"]))

    # Random rotation
    angle = random.uniform(-sim["rotate"], sim["rotate"])
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    # Additive Gaussian noise
    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, sim["noise_std"], arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def run() -> dict:
    """Full drift monitoring pipeline."""
    import torch
    load_model, EmbeddingExtractor = _lazy_model_import()

    print("=== Drift Monitoring ===")

    # ── Load model + reference data ─────────────────────────────────────────
    net = load_model()
    net.to(config.DEVICE).eval()
    extractor = EmbeddingExtractor(net).to(config.DEVICE).eval()
    tf = data_prep.get_transforms(train=False)

    # Load reference features + embeddings
    ref_features_df = pd.read_csv(config.ARTIFACT_DIR / "reference_features.csv")
    ref_embed_data = np.load(config.REFERENCE_EMBED)
    ref_embeddings = ref_embed_data["embeddings"]

    # ── Build current batch (corrupted test images) ─────────────────────────
    root = data_prep.find_data_root()
    test_items = data_prep.load_split("v1", "test", root)
    # Use a sample of test images (same size as reference)
    n_current = min(len(test_items), len(ref_features_df))
    rng = random.Random(config.RANDOM_SEED + 1)
    current_sample = list(test_items)
    rng.shuffle(current_sample)
    current_sample = current_sample[:n_current]

    print(f"Reference: {len(ref_features_df)} images  |  Current: {n_current} images (corrupted)")

    # ── Compute features + embeddings for current batch ─────────────────────
    cur_features_list = []
    cur_embeddings_list = []
    cur_confidences = []
    ref_confidences = []

    for path, label in current_sample:
        with Image.open(path) as img:
            # Corrupt the image to simulate drift
            corrupted = corrupt(img)
            cur_features_list.append(data_prep.image_features(corrupted))

            # Embedding + confidence on corrupted
            x = tf(corrupted.convert("L"))
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x_tensor = x.unsqueeze(0).to(config.DEVICE)

            emb = extractor(x_tensor).cpu().numpy()
            cur_embeddings_list.append(emb)

            logits = net(x_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            cur_confidences.append(probs.max().item())

    # Reference confidences
    ref_items_for_conf = list(test_items)[:len(ref_features_df)]
    for path, label in ref_items_for_conf:
        with Image.open(path) as img:
            x = tf(img.convert("L"))
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x_tensor = x.unsqueeze(0).to(config.DEVICE)
            logits = net(x_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            ref_confidences.append(probs.max().item())

    cur_features_df = pd.DataFrame(cur_features_list)
    cur_embeddings = np.vstack(cur_embeddings_list)

    # ── 1. Statistical drift: Evidently + PSI per feature ───────────────────
    feature_psi = {}
    for feat in config.DRIFT_FEATURES:
        feature_psi[feat] = psi(ref_features_df[feat].values, cur_features_df[feat].values)

    n_drifted = sum(1 for v in feature_psi.values() if v > config.PSI_THRESHOLD)
    drift_share = n_drifted / len(feature_psi) if feature_psi else 0.0
    statistical_drift = drift_share > config.DRIFT_SHARE_THRESHOLD

    print(f"\n── Statistical Drift ──")
    for feat, val in feature_psi.items():
        flag = "DRIFTED" if val > config.PSI_THRESHOLD else "ok"
        print(f"  {feat}: PSI={val:.4f}  [{flag}]")
    print(f"  Drift share: {drift_share:.2f} (threshold={config.DRIFT_SHARE_THRESHOLD})")

    # Evidently report
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_features_df, current_data=cur_features_df)
        report.save_html(str(config.ARTIFACT_DIR / "drift_report.html"))
        print("  ✓ Evidently drift_report.html saved")
    except Exception as e:
        print(f"  ⚠ Evidently report failed: {e}")

    # ── 2. Embedding drift: distance-to-centroid PSI ────────────────────────
    ref_centroid = ref_embeddings.mean(axis=0)
    ref_distances = np.linalg.norm(ref_embeddings - ref_centroid, axis=1)
    cur_distances = np.linalg.norm(cur_embeddings - ref_centroid, axis=1)
    embedding_psi_val = psi(ref_distances, cur_distances)
    embedding_drift = embedding_psi_val > config.EMBEDDING_DRIFT_THRESHOLD

    print(f"\n── Embedding Drift ──")
    print(f"  Distance-to-centroid PSI: {embedding_psi_val:.4f}  "
          f"(threshold={config.EMBEDDING_DRIFT_THRESHOLD})")
    print(f"  Embedding drift: {embedding_drift}")

    # ── 3. Confidence drift ─────────────────────────────────────────────────
    ref_conf_mean = float(np.mean(ref_confidences))
    cur_conf_mean = float(np.mean(cur_confidences))
    conf_drop = ref_conf_mean - cur_conf_mean
    confidence_drift = conf_drop > config.CONFIDENCE_DROP_THRESHOLD

    print(f"\n── Confidence Drift ──")
    print(f"  Reference mean confidence: {ref_conf_mean:.4f}")
    print(f"  Current mean confidence:   {cur_conf_mean:.4f}")
    print(f"  Drop: {conf_drop:.4f}  (threshold={config.CONFIDENCE_DROP_THRESHOLD})")

    # ── Retrain recommendation ──────────────────────────────────────────────
    retrain_recommended = statistical_drift or embedding_drift or confidence_drift

    summary = {
        "statistical_drift": {
            "feature_psi": feature_psi,
            "n_drifted_features": n_drifted,
            "drift_share": drift_share,
            "threshold": config.DRIFT_SHARE_THRESHOLD,
            "drifted": statistical_drift,
        },
        "embedding_drift": {
            "psi": embedding_psi_val,
            "threshold": config.EMBEDDING_DRIFT_THRESHOLD,
            "drifted": embedding_drift,
        },
        "confidence": {
            "reference_mean": ref_conf_mean,
            "current_mean": cur_conf_mean,
            "drop": conf_drop,
            "threshold": config.CONFIDENCE_DROP_THRESHOLD,
            "drifted": confidence_drift,
        },
        "retrain_recommended": retrain_recommended,
    }

    (config.ARTIFACT_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'⚠ RETRAIN RECOMMENDED' if retrain_recommended else '✓ No retrain needed'}")
    print("✓ drift_summary.json saved")

    return summary


if __name__ == "__main__":
    run()
