"""app.py — Stage 3: FastAPI inference service.

Implement /health (liveness + loaded model info) and POST /predict (multipart image upload
→ {label, prob_defect, confidence}). Load the model once at startup; log every prediction
to artifacts/predictions.log.   Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io, json, time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import torch, torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

import config
from src import data_prep
from src.model import load_model

import requests

_state = {"model": None, "tf": None, "meta": {}}


def _load():
    """Load trained model + eval transforms + model metadata at startup."""
    if config.MODEL_PATH.exists():
        _state["model"] = load_model()
        _state["tf"] = data_prep.get_transforms(train=False)
        if config.MODEL_META_PATH.exists():
            _state["meta"] = json.loads(config.MODEL_META_PATH.read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load(); yield


app = FastAPI(title="Casting Defect Detection API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness check: status, model_loaded, classes, positive_class, test_metrics."""
    model_loaded = _state["model"] is not None
    result = {
        "status": "ok",
        "model_loaded": model_loaded,
        "classes": config.CLASS_TO_IDX,
        "positive_class": config.POSITIVE_CLASS,
    }
    # Include test metrics if available
    if config.METRICS_PATH.exists():
        result["test_metrics"] = json.loads(config.METRICS_PATH.read_text())
    if _state["meta"]:
        result["model_version"] = _state["meta"].get("version", "unknown")
    return result


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Image upload → {label, is_defective, prob_defect, confidence}."""
    # Validate model is loaded
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read and validate image
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
        img.load()  # force full decode to catch corrupt images
    except (UnidentifiedImageError, OSError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    # Preprocess
    x = _state["tf"](img.convert("L"))
    if not isinstance(x, torch.Tensor):
        import numpy as np
        x = torch.from_numpy(np.array(x, dtype="float32"))
    x = x.unsqueeze(0)  # add batch dim

    # Inference
    with torch.no_grad():
        logits = _state["model"](x)
        probs = F.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        prob_defect = probs[0, config.POSITIVE_IDX].item()
        confidence = probs[0, pred_idx].item()

    label = config.IDX_TO_CLASS[pred_idx]
    is_defective = pred_idx == config.POSITIVE_IDX

    result = {
        "label": label,
        "is_defective": is_defective,
        "prob_defect": round(prob_defect, 4),
        "confidence": round(confidence, 4),
    }

    # Log prediction with timestamp
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": file.filename,
        **result,
    }
    with open(config.PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return result
