"""data_prep.py — Stage 1: discovery, quality validation, versioned splits, transforms. 

Implement: locate the casting folders; run data-quality checks (missing/corrupt/duplicate/
dimension/class-distribution/consistency); build reproducible stratified train/val/test
splits with a versioned snapshot + metadata.json; define preprocessing + augmentation
transforms; and per-image feature extraction used by drift monitoring.
"""
from __future__ import annotations

import hashlib, json, os, random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from pathlib import Path
from datetime import datetime
import json

from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_data_root(base: Path | None = None) -> Path:
    search_root = Path(base) if base else Path(config.DATA_DIR)

    train_dir = search_root / "train"
    test_dir = search_root / "test"

    if (
        train_dir.exists()
        and test_dir.exists()
        and (train_dir / "ok_front").exists()
        and (train_dir / "def_front").exists()
    ):
        return search_root

    for path in search_root.rglob("*"):
        if not path.is_dir():
            continue

        train_dir = path / "train"
        test_dir = path / "test"

        if (
            train_dir.exists()
            and test_dir.exists()
            and (train_dir / "ok_front").exists()
            and (train_dir / "def_front").exists()
        ):
            return path

    raise FileNotFoundError(
        f"Could not locate casting dataset under {search_root}"
    )


def list_images(split_dir: Path) -> list[tuple[Path, int]]:
    items = []
    for cls, idx in config.CLASS_TO_IDX.items():
        for p in sorted((split_dir / cls).glob("*")):
            if p.suffix.lower() in IMG_EXTS:
                items.append((p, idx))
    return items


def validate_quality(root: Path) -> dict:
    train_images = list_images(root / "train")
    test_images = list_images(root / "test")

    all_images = train_images + test_images

    corrupt = []
    invalid_size = []

    for img_path, label in all_images:

        try:
            img = Image.open(img_path)
            img.verify()          # Verify image integrity

            img = Image.open(img_path)

            if img.width == 0 or img.height == 0:
                invalid_size.append(img_path)

        except Exception:
            corrupt.append(img_path)

    report = {
        "total_images": len(all_images),
        "corrupt_images": len(corrupt),
        "invalid_dimensions": len(invalid_size),
        "passed": len(corrupt) == 0
    }

    return report
    


def build_splits(root: Path, version: str = "v1") -> dict:
    # Read images
    train_items = list_images(root / "train")
    test_items = list_images(root / "test")

    # Split paths and labels
    paths = [p for p, _ in train_items]
    labels = [y for _, y in train_items]

    # Stratified train-validation split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )

    # Convert absolute path → relative path
    def make_records(paths, labels):
        records = []
        for p, y in zip(paths, labels):
            rel = str(p.relative_to(root)).replace("\\", "/")
            records.append([rel, y])
        return records

    train_records = make_records(train_paths, train_labels)
    val_records = make_records(val_paths, val_labels)
    test_records = make_records(
        [p for p, _ in test_items],
        [y for _, y in test_items],
    )

    # Create version folder
    version_dir = config.SPLIT_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON files
    (version_dir / "train.json").write_text(
        json.dumps(train_records, indent=2)
    )

    (version_dir / "val.json").write_text(
        json.dumps(val_records, indent=2)
    )

    (version_dir / "test.json").write_text(
        json.dumps(test_records, indent=2)
    )

    # Metadata
    metadata = {
        "version": version,
        "created": datetime.now().isoformat(),
        "seed": config.RANDOM_SEED,
        "validation_size": config.VAL_SPLIT,
        "classes": config.CLASS_TO_IDX,
        "train_samples": len(train_records),
        "validation_samples": len(val_records),
        "test_samples": len(test_records),
    }

    (version_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    return metadata


def load_split(version: str, name: str, root: Path) -> list[tuple[Path, int]]:
    rel = json.loads((config.SPLIT_DIR / version / f"{name}.json").read_text())
    return [(root / r, y) for r, y in rel]


def get_transforms(train: bool):
    """Build preprocessing + augmentation transforms for train or eval."""
    from torchvision import transforms

    base_transforms = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    ]
    if train:
        base_transforms.extend([
            transforms.RandomHorizontalFlip(p=config.AUG["hflip_p"]),
            transforms.RandomAffine(
                degrees=config.AUG["rotation_degrees"],
                translate=(config.AUG["translate"], config.AUG["translate"]),
            ),
            transforms.ColorJitter(
                brightness=config.AUG["brightness"],
                contrast=config.AUG["contrast"],
            ),
        ])

    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    return transforms.Compose(base_transforms)


def image_features(img: Image.Image) -> dict:
    """Extract per-image features for statistical drift monitoring."""
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    mean_intensity = float(arr.mean())
    brightness = float(mean_intensity / 255.0)
    contrast = float(arr.std() / 255.0)

    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    edge_density = float((edges > 16).sum() / edges.size)

    laplacian = gray.filter(
        ImageFilter.Kernel(
            (3, 3),
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            scale=1,
            offset=0,
        )
    )
    sharpness = float(np.asarray(laplacian, dtype=np.float32).var() / (255.0 ** 2))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "sharpness": sharpness,
        "mean_intensity": mean_intensity,
    }

