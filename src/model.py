"""Image-only baseline model + size/latency helpers.

Backbone: torchvision EfficientNet-B0 with ImageNet weights (Tan & Le, 2019; torchvision BSD-3 code,
weights trained on ImageNet-1k - see README for citation and licence note).

The model returns (logits, features). Phase 2 will concatenate a text embedding to `features` before the
classifier, so the head is kept as a separate, easily replaceable module.
"""
import copy
import time

import torch
import torch.nn as nn
from torchvision import models


class ImageClassifier(nn.Module):
    """EfficientNet-B0 trunk -> global-average-pooled feature vector -> dropout -> linear classifier."""

    def __init__(self, num_classes: int, backbone: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        if backbone != "efficientnet_b0":
            # Phase 2 backbone ablation will add entries here; failing loudly beats silently using the wrong model.
            raise ValueError(f"Unknown backbone {backbone!r}")
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        self.features = net.features            # conv trunk
        self.pool = net.avgpool                 # global average pool -> (B, 1280, 1, 1)
        self.feature_dim = net.classifier[1].in_features   # 1280
        self.dropout = nn.Dropout(dropout)      # regularises the new head (the trunk is pretrained)
        self.classifier = nn.Linear(self.feature_dim, num_classes)   # NEW head sized to our classes

    def forward(self, images: torch.Tensor):
        feats = torch.flatten(self.pool(self.features(images)), 1)   # (B, 1280) pooled feature vector
        logits = self.classifier(self.dropout(feats))
        return logits, feats


def build_model(cfg: dict, num_classes: int) -> ImageClassifier:
    m = cfg["model"]
    return ImageClassifier(num_classes, m["name"], m["pretrained"], m["dropout"])


def count_parameters(model: nn.Module) -> dict:
    """Total and trainable parameter counts (the target is <= 25M total)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


@torch.no_grad()
def measure_latency(model: nn.Module, device: str, image_size: int, warmup: int = 10, runs: int = 50) -> dict:
    """Average ms per sample at batch size 1.

    Warm-up runs first: the first calls include lazy initialisation / kernel selection and would inflate the mean.
    On GPU we call synchronize(), because CUDA is asynchronous and the clock would otherwise stop early.
    A deep copy is used so measuring never changes the model being evaluated.
    """
    m = copy.deepcopy(model).to(device).eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    for _ in range(warmup):
        m(x)
    sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        m(x)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(times)
    return {"device": device, "mean_ms": t.mean().item(), "std_ms": t.std().item(), "runs": runs,
            "cpu_threads": torch.get_num_threads() if device == "cpu" else None}


def measure_latency_all(model, image_size, warmup, runs) -> dict:
    """CPU always; GPU too when one is available."""
    out = {"cpu": measure_latency(model, "cpu", image_size, warmup, runs)}
    if torch.cuda.is_available():
        out["gpu"] = measure_latency(model, "cuda", image_size, warmup, runs)
    return out
