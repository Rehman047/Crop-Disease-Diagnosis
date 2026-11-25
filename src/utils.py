"""Small helpers shared by every module: seeding, config loading, paths, JSON saving.

Kept deliberately tiny and dependency-light so it is easy to explain in a viva.
"""
import copy
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch, and ask cuDNN for deterministic kernels.

    WHY: ablations are only meaningful if two runs differ in the thing we changed,
    not in luck. Note: some CUDA ops (and AMP) can still be non-deterministic
    at the last decimal place; the seeds make runs *repeatable in practice*, and
    the 3-seed mean +/- std quantifies the remaining noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True   # same kernels every time
    torch.backends.cudnn.benchmark = False      # benchmark mode picks kernels non-deterministically


def seed_worker(worker_id: int) -> None:
    """DataLoader workers get their own RNGs; derive them from the torch seed so augmentation is repeatable."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(path: str) -> dict:
    """Read a YAML config into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_smoke_overrides(cfg: dict) -> dict:
    """Return a COPY of cfg shrunk for a fast CPU pipeline check.

    Results go to <results_dir>/smoke so they can never be confused with real runs.
    """
    cfg = copy.deepcopy(cfg)
    s = cfg["smoke"]
    s["enabled"] = True
    cfg["train"]["epochs"] = s["epochs"]
    cfg["train"]["early_stop_patience"] = max(cfg["train"]["early_stop_patience"], s["epochs"])
    cfg["data"]["image_size"] = s["image_size"]
    cfg["data"]["batch_size"] = s["batch_size"]
    cfg["data"]["num_workers"] = s["num_workers"]
    cfg["eval"]["latency_runs"] = 5
    cfg["eval"]["latency_warmup"] = 2
    cfg["paths"]["results_dir"] = str(Path(cfg["paths"]["results_dir"]) / "smoke")
    return cfg


def results_dir(cfg: dict) -> Path:
    """Top-level results folder (created on demand)."""
    p = Path(cfg["paths"]["results_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_dir(cfg: dict, seed: int) -> Path:
    """Folder for one (experiment, seed) run: checkpoint, curves, metrics, config copy."""
    p = results_dir(cfg) / "runs" / cfg["experiment_name"] / f"seed_{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_hash(cfg: dict) -> str:
    """Short hash of the config, stored in every metrics file so a result can be traced to its exact settings."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def save_json(obj, path) -> None:
    """Write JSON (pretty, so it is human-readable and diff-able)."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
