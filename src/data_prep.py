"""data_prep.py — Stage 1: discovery, quality validation, versioned splits, transforms.

Implement: locate the casting folders; run data-quality checks (missing/corrupt/duplicate/
dimension/class-distribution/consistency); build reproducible stratified train/val/test
splits with a versioned snapshot + metadata.json; define preprocessing + augmentation
transforms; and per-image feature extraction used by drift monitoring.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_data_root(base: Path | None = None) -> Path:
    search_root = Path(base) if base else Path(config.DATA_DIR)

    if search_root.exists() and (search_root / "train").exists() and (search_root / "test").exists():
        if (search_root / "train" / "ok_front").exists() and (search_root / "train" / "def_front").exists():
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

    raise FileNotFoundError(f"Could not locate casting dataset under {search_root}")


def list_images(split_dir: Path) -> list[tuple[Path, int]]:
    items = []
    for cls, idx in config.CLASS_TO_IDX.items():
        for p in sorted((split_dir / cls).glob("*")):
            if p.suffix.lower() in IMG_EXTS:
                items.append((p, idx))
    return items


def validate_quality(root: Path) -> dict:
    """Scan train and test folders for basic data-quality issues."""
    issues: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    duplicates: list[str] = []
    hashes: dict[str, str] = {}

    for split_name in ("train", "test"):
        split_dir = root / split_name
        split_counts = Counter()
        split_items: list[tuple[Path, int]] = []

        if not split_dir.exists():
            issues.append(f"Missing split folder: {split_dir}")
            continue

        for cls_name in config.CLASSES:
            class_dir = split_dir / cls_name
            if not class_dir.exists():
                issues.append(f"Missing class folder: {class_dir}")
                continue
            for path in sorted(class_dir.glob("*")):
                if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
                    continue
                split_items.append((path, config.CLASS_TO_IDX[cls_name]))
                split_counts[cls_name] += 1

                try:
                    with Image.open(path) as img:
                        img.load()
                        width, height = img.size
                        if (width, height) != (300, 300):
                            issues.append(f"Unexpected size for {path}: {(width, height)}")
                    sha = hashlib.md5(path.read_bytes()).hexdigest()
                    if sha in hashes:
                        duplicates.append(f"Duplicate image: {path} matches {hashes[sha]}")
                    else:
                        hashes[sha] = str(path)
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    issues.append(f"Corrupt image: {path} ({exc})")

        counts[split_name] = dict(split_counts)

    report = {
        "root": str(root),
        "counts": counts,
        "issues": issues + duplicates,
        "passed": len(issues) + len(duplicates) == 0,
    }
    return report


def build_splits(root: Path, version: str = "v1") -> dict:
    """Create versioned, stratified train/val/test split snapshots."""
    root = Path(root)
    split_dir = config.SPLIT_DIR / version
    split_dir.mkdir(parents=True, exist_ok=True)

    train_items = list_images(root / "train")
    test_items = list_images(root / "test")

    rng = random.Random(config.RANDOM_SEED)
    train_by_class: dict[int, list[tuple[Path, int]]] = {idx: [] for idx in config.CLASS_TO_IDX.values()}
    for path, label in train_items:
        train_by_class[label].append((path, label))

    train_split: list[tuple[str, int]] = []
    val_split: list[tuple[str, int]] = []
    for label, items in train_by_class.items():
        rng.shuffle(items)
        val_size = max(1, int(len(items) * config.VAL_SPLIT)) if len(items) > 1 else 0
        for path, _ in items[val_size:]:
            train_split.append((str(path.relative_to(root)).replace(os.sep, "/"), label))
        for path, _ in items[:val_size]:
            val_split.append((str(path.relative_to(root)).replace(os.sep, "/"), label))

    test_split = [
        (str(path.relative_to(root)).replace(os.sep, "/"), label)
        for path, label in test_items
    ]

    for name, payload in (("train", train_split), ("val", val_split), ("test", test_split)):
        (split_dir / f"{name}.json").write_text(json.dumps(payload, indent=2))

    metadata = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "seed": config.RANDOM_SEED,
        "val_split": config.VAL_SPLIT,
        "class_to_idx": config.CLASS_TO_IDX,
        "split_sizes": {
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_split),
        },
        "class_distributions": {
            "train": dict(Counter(label for _, label in train_split)),
            "val": dict(Counter(label for _, label in val_split)),
            "test": dict(Counter(label for _, label in test_split)),
        },
    }
    (split_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def load_split(version: str, name: str, root: Path) -> list[tuple[Path, int]]:
    rel = json.loads((config.SPLIT_DIR / version / f"{name}.json").read_text())
    return [(root / r, y) for r, y in rel]


def get_transforms(train: bool):
    from torchvision import transforms

    base = [
        transforms.Grayscale(3),
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]

    if train:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=config.AUG["rotation_degrees"], translate=(config.AUG["translate"], config.AUG["translate"]), fill=0),
            transforms.ColorJitter(brightness=config.AUG["brightness"], contrast=config.AUG["contrast"]),
        ]
        return transforms.Compose(aug + base)

    return transforms.Compose(base)


def image_features(img: Image.Image) -> dict:
    """Return simple image-summary features for drift monitoring."""
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    mean_intensity = float(arr.mean())
    contrast = float(arr.std())

    edge_img = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edge_img, dtype=np.float32)
    edge_density = float(np.mean(edge_arr > 0))

    stat = ImageStat.Stat(gray)
    sharpness = float(stat.var[0])

    return {
        "brightness": mean_intensity,
        "contrast": contrast,
        "edge_density": edge_density,
        "sharpness": sharpness,
        "mean_intensity": mean_intensity,
    }
