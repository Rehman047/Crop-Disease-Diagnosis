"""Training loop for the image-only baseline.

Rules enforced here:
  * Model selection and early stopping use ID-VALIDATION macro F1 only.
  * This file never imports anything that loads PlantDoc, so training cannot see the shifted set.
Usage:  python -m src.train --config configs/baseline.yaml --seed 42 [--smoke] [--no-pretrained]
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import make_loaders, prepare_data
from .evaluate import macro_f1
from .model import build_model, count_parameters
from .utils import (apply_smoke_overrides, config_hash, get_device, load_config, run_dir, save_json, set_seed)


# ----------------------------------------------------------------------------------------------
# Loss factory (selectable by name so a loss ablation is a config change)
# ----------------------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss: cross-entropy scaled by (1 - p_true)^gamma, so easy examples contribute little.

    gamma=0 is plain cross-entropy. Optional per-class `weight` works like in nn.CrossEntropyLoss.
    """
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target[:, None])[:, 0]           # log-prob of the true class
        loss = -((1 - logp_t.exp()) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def inverse_frequency_weights(class_counts: np.ndarray) -> torch.Tensor:
    """w_c = N / (C * n_c): rare classes weigh more, so the loss is not dominated by the big classes.
    A class with 0 training images gets weight 0 (it cannot be learned anyway)."""
    counts = class_counts.astype(float)
    w = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def build_loss(cfg, class_counts, device) -> nn.Module:
    name = cfg["loss"]["name"]
    if name == "ce_weighted":
        return nn.CrossEntropyLoss(weight=inverse_frequency_weights(class_counts).to(device))
    if name == "focal":
        return FocalLoss(cfg["loss"]["focal_gamma"])
    if name == "ce_label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=cfg["loss"]["label_smoothing"])
    raise ValueError(f"Unknown loss {name!r}; choose ce_weighted | focal | ce_label_smoothing")


def build_scheduler(optimizer, cfg):
    t = cfg["train"]
    if t["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    if t["scheduler"] == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=t["step_size"], gamma=t["step_gamma"])
    raise ValueError(f"Unknown scheduler {t['scheduler']!r}")


# ----------------------------------------------------------------------------------------------
# One epoch of training / validation
# ----------------------------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x, y = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # AMP: forward in float16 where safe (faster, less memory on free GPUs); GradScaler avoids fp16 underflow
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, _ = model(x)
            loss = criterion(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple:
    """Returns (val loss, val macro F1) on the ID VALIDATION split."""
    model.eval()
    losses, ys, ps = 0.0, [], []
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        logits, _ = model(x)
        losses += criterion(logits.float(), y).item() * len(y)
        ys.append(y.cpu())
        ps.append(logits.argmax(1).cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return losses / len(y), macro_f1(y, p)


# ----------------------------------------------------------------------------------------------
# Full run for one seed
# ----------------------------------------------------------------------------------------------
def train_one_seed(cfg: dict, seed: int, bundle=None) -> dict:
    """Train, keep the checkpoint with the best ID-val macro F1, save curve CSV + config + summary."""
    set_seed(seed)
    device = get_device()
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    out = run_dir(cfg, seed)
    save_json({**cfg, "_seed": seed, "_config_hash": config_hash(cfg)}, out / "config_used.json")

    bundle = bundle or prepare_data(cfg)
    loaders = make_loaders(bundle, cfg, seed)
    model = build_model(cfg, bundle.num_classes).to(device)
    criterion = build_loss(cfg, bundle.train_class_counts(), device)
    t = cfg["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[seed {seed}] device={device} amp={use_amp} params={count_parameters(model)['total']/1e6:.2f}M "
          f"train/val={len(loaders['train'].dataset)}/{len(loaders['val'].dataset)}")

    best_f1, best_epoch, bad, curve = -1.0, -1, 0, []
    for epoch in range(1, t["epochs"] + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device, use_amp)
        val_loss, val_f1 = validate(model, loaders["val"], criterion, device)
        scheduler.step()
        curve.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=val_loss, val_macro_f1=val_f1,
                          lr=optimizer.param_groups[0]["lr"], seconds=time.time() - t0))
        print(f"  epoch {epoch:02d} train_loss {tr_loss:.4f} val_loss {val_loss:.4f} val_macro_F1 {val_f1:.4f}")
        if val_f1 > best_f1:                                   # checkpoint selection on ID-val macro F1 ONLY
            best_f1, best_epoch, bad = val_f1, epoch, 0
            torch.save({"model_state": copy.deepcopy(model.state_dict()), "class_names": bundle.class_names,
                        "epoch": epoch, "val_macro_f1": val_f1, "seed": seed, "config_hash": config_hash(cfg),
                        "selected_on": "id_val_macro_f1"}, out / "best.pt")
        else:
            bad += 1
            if bad >= t["early_stop_patience"]:
                print(f"  early stopping at epoch {epoch} (best epoch {best_epoch})")
                break
        pd.DataFrame(curve).to_csv(out / "training_curve.csv", index=False)   # saved every epoch (safe if Kaggle dies)
    pd.DataFrame(curve).to_csv(out / "training_curve.csv", index=False)

    summary = {"seed": seed, "experiment": cfg["experiment_name"], "config_hash": config_hash(cfg),
               "best_epoch": best_epoch, "best_val_macro_f1": best_f1, "epochs_run": len(curve),
               "train_seconds": float(sum(c["seconds"] for c in curve)), "smoke": cfg["smoke"]["enabled"]}
    save_json(summary, out / "train_summary.json")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the image-only baseline.")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", help="override config seeds")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU pipeline check (writes to results/smoke)")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline smoke tests)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.no_pretrained:
        cfg["model"]["pretrained"] = False
    if a.smoke:
        cfg = apply_smoke_overrides(cfg)
    bundle = prepare_data(cfg)
    for sd in a.seeds or cfg["seeds"]:
        train_one_seed(cfg, sd, bundle)
