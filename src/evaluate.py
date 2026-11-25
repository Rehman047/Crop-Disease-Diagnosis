"""Metrics, calibration, plots, and the (guarded) evaluation on the shifted PlantDoc test set.

Shifted-set discipline: the ONLY code path that loads PlantDoc images is `evaluate_shifted`, which
  * takes a checkpoint FILE that train.py already selected on ID-validation macro F1 (so it cannot be called
    with a model that is mid-training or chosen by looking at PlantDoc),
  * appends a line to results/shifted_access_log.csv on every call, for transparency.
train.py does not import this function.
"""
import argparse
import copy
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .data import build_shifted_test, make_loaders, prepare_data
from .model import build_model, count_parameters, measure_latency_all
from .utils import (apply_smoke_overrides, config_hash, get_device, load_config, load_json,
                    results_dir, run_dir, save_json, set_seed)


# ----------------------------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device) -> dict:
    """Run the model over a loader. Returns softmax probabilities, labels and predictions as numpy arrays."""
    model.eval()
    probs, labels = [], []
    for batch in loader:
        logits, _ = model(batch["image"].to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(batch["label"])
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    return {"probs": probs, "labels": labels, "preds": probs.argmax(1)}


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def macro_f1(y_true, y_pred) -> float:
    """Macro F1 over classes that actually occur in y_true.

    WHY: with tiny classes (esp. PlantDoc) a class can have zero test images; counting it would add a fake 0
    and make scores depend on how the split happened to fall. The count of classes used is reported alongside.
    """
    present = np.unique(y_true)
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def resolve_high_cost(cfg, class_names) -> list:
    """Indices of the config's high-cost classes that exist in the class intersection (warn about the rest)."""
    wanted = cfg["eval"]["high_cost_classes"]
    idx = [class_names.index(n) for n in wanted if n in class_names]
    missing = [n for n in wanted if n not in class_names]
    if missing:
        print(f"[WARNING] high-cost classes not in the class intersection (ignored): {missing}")
    if not idx:
        raise ValueError("None of eval.high_cost_classes exists in the class intersection - fix the config.")
    return idx


def classification_metrics(y_true, y_pred, class_names, high_cost_idx) -> tuple:
    """Returns (summary dict, per-class DataFrame)."""
    C = len(class_names)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(C), zero_division=0)
    per_class = pd.DataFrame({"class_name": class_names, "precision": p, "recall": r, "f1": f, "support": s,
                              "high_cost": [i in high_cost_idx for i in range(C)]})
    present = per_class["support"] > 0
    hc = per_class[per_class["high_cost"] & present]
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_recall": float(per_class.loc[present, "recall"].mean()),
        "macro_precision": float(per_class.loc[present, "precision"].mean()),
        # recall averaged over high-cost classes only (a missed fast-spreading disease is the costly error)
        "high_cost_macro_recall": float(hc["recall"].mean()) if len(hc) else float("nan"),
        "n_high_cost_with_support": int(len(hc)),
        "n_classes_with_support": int(present.sum()),
        "n_samples": int(len(y_true)),
    }
    return summary, per_class


def expected_calibration_error(probs, labels, n_bins: int = 15) -> tuple:
    """ECE = sum over confidence bins of (bin_size/N) * |accuracy_in_bin - mean_confidence_in_bin|.

    Returns (ece, bins DataFrame) - the DataFrame also feeds the reliability diagram.
    """
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append(dict(lo=lo, hi=hi, count=0, accuracy=np.nan, confidence=np.nan))
            continue
        acc, c = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - c)
        rows.append(dict(lo=lo, hi=hi, count=int(m.sum()), accuracy=acc, confidence=c))
    return float(ece), pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names))).astype(float)
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)       # row-normalise: each row = "what happens to class X"
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * n), max(5, 0.4 * n)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability_diagram(bins: pd.DataFrame, ece: float, path, title=""):
    """Bars = accuracy per confidence bin; diagonal = perfect calibration. Bars below diagonal = over-confident."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = bins["count"] > 0
    mid = (bins["lo"] + bins["hi"]) / 2
    width = (bins["hi"] - bins["lo"]).iloc[0]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.bar(mid[ok], bins["accuracy"][ok], width=width, edgecolor="k", alpha=0.7, label="accuracy")
    ax.plot([0, 1], [0, 1], "r--", label="perfect")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE={ece:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Evaluating a trained checkpoint
# ----------------------------------------------------------------------------------------------
def load_checkpoint(cfg, ckpt_path, num_classes, device):
    """Rebuild the model from the config and load saved weights (no need to download pretrained weights again)."""
    cfg = copy.deepcopy(cfg)
    cfg["model"]["pretrained"] = False
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg, num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval(), ckpt


def _full_metrics(pred: dict, class_names, high_cost_idx, n_bins, out_dir: Path, tag: str) -> dict:
    """Metrics + per-class CSV + confusion matrix + reliability diagram for one test set."""
    summary, per_class = classification_metrics(pred["labels"], pred["preds"], class_names, high_cost_idx)
    ece, bins = expected_calibration_error(pred["probs"], pred["labels"], n_bins)
    summary["ece"] = ece
    per_class.to_csv(out_dir / f"per_class_{tag}.csv", index=False)
    bins.to_csv(out_dir / f"reliability_bins_{tag}.csv", index=False)
    plot_confusion_matrix(pred["labels"], pred["preds"], class_names, out_dir / f"confusion_{tag}.png", f"{tag} confusion")
    plot_reliability_diagram(bins, ece, out_dir / f"reliability_{tag}.png", tag)
    return summary


def evaluate_shifted(model, ckpt_path, cfg, bundle, seed: int, high_cost_idx, out_dir: Path) -> dict:
    """Evaluate on PlantDoc (the shifted set). Guarded and logged - see module docstring."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("selected_on") != "id_val_macro_f1":
        raise RuntimeError("Refusing to touch the shifted set: this checkpoint was not selected on ID validation macro F1.")
    shifted_df, loader = build_shifted_test(bundle, cfg, seed)
    log_path = results_dir(cfg) / "shifted_access_log.csv"
    row = pd.DataFrame([dict(timestamp=dt.datetime.now().isoformat(timespec="seconds"), checkpoint=str(ckpt_path),
                             seed=seed, experiment=cfg["experiment_name"], config_hash=config_hash(cfg),
                             n_images=len(shifted_df))])
    row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)   # log BEFORE evaluating
    pred = predict(model, loader, get_device())
    return _full_metrics(pred, bundle.class_names, high_cost_idx, cfg["eval"]["ece_bins"], out_dir, "shifted")


def evaluate_run(cfg, seed: int, bundle=None) -> dict:
    """Evaluate the saved best checkpoint of (config, seed) on ID test + shifted set; save metrics.json."""
    set_seed(seed)
    device = get_device()
    bundle = bundle or prepare_data(cfg, verbose=False)
    out_dir = run_dir(cfg, seed)
    ckpt_path = out_dir / "best.pt"
    model, ckpt = load_checkpoint(cfg, ckpt_path, bundle.num_classes, device)
    hc = resolve_high_cost(cfg, bundle.class_names)
    n_bins = cfg["eval"]["ece_bins"]

    id_pred = predict(model, make_loaders(bundle, cfg, seed)["test"], device)
    id_metrics = _full_metrics(id_pred, bundle.class_names, hc, n_bins, out_dir, "id_test")
    shifted_metrics = evaluate_shifted(model, ckpt_path, cfg, bundle, seed, hc, out_dir)

    e = cfg["eval"]
    metrics = {
        "experiment": cfg["experiment_name"], "seed": seed, "config_hash": config_hash(cfg),
        "smoke": cfg["smoke"]["enabled"], "best_epoch": ckpt["epoch"], "val_macro_f1": ckpt["val_macro_f1"],
        "high_cost_classes_used": [bundle.class_names[i] for i in hc],
        "params": count_parameters(model),
        "latency_ms": measure_latency_all(model, cfg["data"]["image_size"], e["latency_warmup"], e["latency_runs"]),
        "id_test": id_metrics, "shifted_test": shifted_metrics,
        "id_to_shifted_drop": {k: id_metrics[k] - shifted_metrics[k]
                               for k in ("macro_f1", "macro_recall", "high_cost_macro_recall", "accuracy")},
    }
    save_json(metrics, out_dir / "metrics.json")
    print(f"[seed {seed}] ID macro-F1 {id_metrics['macro_f1']:.4f} | shifted macro-F1 {shifted_metrics['macro_f1']:.4f}")
    return metrics


# ----------------------------------------------------------------------------------------------
# Aggregation over seeds
# ----------------------------------------------------------------------------------------------
def aggregate_runs(cfg) -> pd.DataFrame:
    """Read every seed's metrics.json and report mean +/- std (sample std, ddof=1) with achieved-vs-target."""
    files = sorted((results_dir(cfg) / "runs" / cfg["experiment_name"]).glob("seed_*/metrics.json"))
    if not files:
        raise FileNotFoundError("No metrics.json found - run evaluate first.")
    runs = [load_json(f) for f in files]

    def ms(vals):
        v = np.array(vals, dtype=float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))

    def row(name, key_fn, target=None, higher_better=True):
        m, s = ms([key_fn(r) for r in runs])
        met = "" if target is None else ("yes" if (m >= target if higher_better else m <= target) else "NO")
        return dict(metric=name, mean=m, std=s, target=target, met=met)

    tg = cfg["eval"]["targets"]
    rows = []
    for split, label in (("id_test", "ID"), ("shifted_test", "Shifted")):
        t_hc = tg["high_cost_recall"]
        rows += [row(f"{label} macro F1", lambda r, s=split: r[s]["macro_f1"],
                     tg["shifted_macro_f1"] if split == "shifted_test" else None),
                 row(f"{label} high-cost macro recall", lambda r, s=split: r[s]["high_cost_macro_recall"], t_hc),
                 row(f"{label} ECE", lambda r, s=split: r[s]["ece"], tg["ece"], higher_better=False),
                 row(f"{label} accuracy", lambda r, s=split: r[s]["accuracy"])]
    rows += [row("Drop in macro F1 (ID - shifted)", lambda r: r["id_to_shifted_drop"]["macro_f1"]),
             row("Drop in high-cost recall (ID - shifted)", lambda r: r["id_to_shifted_drop"]["high_cost_macro_recall"]),
             row("Params (M)", lambda r: r["params"]["total"] / 1e6, tg["params_m"], higher_better=False),
             row("CPU latency (ms, batch 1)", lambda r: r["latency_ms"]["cpu"]["mean_ms"], tg["latency_ms"], higher_better=False)]
    if all("gpu" in r["latency_ms"] for r in runs):
        rows.append(row("GPU latency (ms, batch 1)", lambda r: r["latency_ms"]["gpu"]["mean_ms"], tg["latency_ms"], higher_better=False))
    table = pd.DataFrame(rows)
    table["n_seeds"] = len(runs)
    table["seeds"] = ",".join(str(r["seed"]) for r in runs)
    table.to_csv(results_dir(cfg) / f"summary_{cfg['experiment_name']}.csv", index=False)
    return table


def format_table(table: pd.DataFrame) -> str:
    """Pretty 'mean +/- std' text table."""
    out = table.copy()
    out["value"] = [f"{m:.4f} +/- {s:.4f}" if s == s else f"{m:.4f}" for m, s in zip(out["mean"], out["std"])]
    return out[["metric", "value", "target", "met"]].to_string(index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate saved checkpoints (ID test + shifted set).")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", help="override config seeds")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.smoke:
        cfg = apply_smoke_overrides(cfg)
    bundle = prepare_data(cfg, verbose=False)
    for sd in a.seeds or cfg["seeds"]:
        evaluate_run(cfg, sd, bundle)
    print(format_table(aggregate_runs(cfg)))
