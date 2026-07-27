"""model.py — Stage 2: transfer-learning model + embedding extractor.

Configure a pretrained ResNet18 backbone for transfer learning (freeze the backbone,
replace the final layer with a fresh 2-class head). The same backbone is reused as a
512-dim feature extractor for embedding drift. See notebook "Model Development".
"""
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def build_model(freeze: bool | None = None) -> nn.Module:
    """Load ImageNet-pretrained ResNet18; freeze backbone; replace fc with 2-class head."""
    from torchvision.models import resnet18, ResNet18_Weights

    freeze = config.FREEZE_BACKBONE if freeze is None else freeze
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 2.2.1 — freeze the backbone so only the head trains
    if freeze:
        for param in net.parameters():
            param.requires_grad = False

    # 2.2.2 — replace the final fc with a fresh 2-class head (always trainable)
    in_features = net.fc.in_features  # 512
    net.fc = nn.Linear(in_features, config.NUM_CLASSES)
    for param in net.fc.parameters():
        param.requires_grad = True

    return net


def trainable_parameters(net: nn.Module):
    return [p for p in net.parameters() if p.requires_grad]


class EmbeddingExtractor(nn.Module):
    """Expose the 512-dim penultimate features (drop the fc layer)."""
    def __init__(self, net: nn.Module):
        super().__init__()
        # Keep all layers except the final fc — gives the 512-dim embedding
        self.features = nn.Sequential(*list(net.children())[:-1])

    @torch.no_grad()
    def forward(self, x):
        return self.features(x).flatten(1)  # (B, 512)


def save_model(net: nn.Module, path: Path | None = None) -> None:
    torch.save(net.state_dict(), path or config.MODEL_PATH)


def load_model(path: Path | None = None, freeze: bool = True) -> nn.Module:
    net = build_model(freeze=freeze)
    net.load_state_dict(torch.load(path or config.MODEL_PATH, map_location=config.DEVICE))
    net.eval()
    return net
