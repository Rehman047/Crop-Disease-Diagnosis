"""Data pipeline: dataset discovery, class intersection, leakage-aware split, transforms, loaders, analysis plots.

Design rules (see README):
  * PlantVillage (PV) = in-distribution. Split 70/15/15 with near-duplicate groups kept together.
  * PlantDoc = SHIFTED TEST SET ONLY. Everything that touches PlantDoc images lives in the
    "SHIFTED TEST SET" section at the bottom and is imported only by evaluate.py / the notebook.
    train.py never imports it, so a training run cannot see PlantDoc.
  * Nothing here invents numbers: every count comes from the files on disk.
"""
import json
import math
import os
import re
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import results_dir, seed_worker

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op wrapper
    def tqdm(x, **kw):
        return x

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SEP = "___"  # separates crop and condition in our canonical class names, e.g. "tomato___late blight"


# ----------------------------------------------------------------------------------------------
# 1. Finding the datasets and printing their real structure (Step 0: inspect, don't guess)
# ----------------------------------------------------------------------------------------------
def _is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMG_EXTS


def list_class_folders(root) -> dict:
    """{folder_name: [image file names]} for every direct sub-folder of root that holds images."""
    out = {}
    for d in sorted(Path(root).iterdir()):
        if d.is_dir():
            files = sorted(f.name for f in d.iterdir() if f.is_file() and _is_image(f.name))
            if files:
                out[d.name] = files
    return out


def print_dataset_structure(root, title="", max_classes=80) -> None:
    """Print folder -> image count, so we can see the REAL class names before trusting any mapping."""
    root = Path(root)
    print(f"=== {title}: {root}")
    subdirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    print(f"sub-folders ({len(subdirs)}): {subdirs[:max_classes]}")
    classes = list_class_folders(root)
    if classes:
        print(f"{len(classes)} class folders, {sum(len(v) for v in classes.values())} images")
        for name, files in list(classes.items())[:max_classes]:
            print(f"   {name!r}: {len(files)}")


def _find_class_containers(base, max_depth=5):
    """Directories under base that look like 'a folder of class folders' (>=5 sub-folders that hold images)."""
    found = []
    base = str(base)
    for dirpath, dirnames, _ in os.walk(base):
        if dirpath[len(base):].count(os.sep) >= max_depth:
            dirnames[:] = []
            continue
        if len(dirnames) >= 5:
            with_images = sum(
                any(_is_image(f) for f in os.listdir(os.path.join(dirpath, d))) for d in dirnames
            )
            if with_images >= 0.8 * len(dirnames):
                found.append(dirpath)
    return found


def resolve_dataset_roots(cfg) -> tuple:
    """Return (plantvillage_root, plantdoc_root). Config paths win; otherwise try /kaggle/input.

    We never hardcode personal paths. If auto-detection fails we stop with a clear message
    rather than silently guessing.
    """
    pv, pdoc = cfg["paths"].get("plantvillage_root"), cfg["paths"].get("plantdoc_root")
    kaggle = Path("/kaggle/input")
    if (pv is None or pdoc is None) and kaggle.exists():
        prefer = cfg["data"].get("plantvillage_prefer_subdir", "color").lower()
        split_dirs = cfg["data"]["plantdoc_dirs"]
        if pv is None:
            cands = [c for c in _find_class_containers(kaggle)
                     if re.search(r"plant ?_?village|new plant diseases", c.lower())]
            preferred = [c for c in cands if Path(c).name.lower() == prefer]
            pool = preferred or sorted(cands, key=len)
            pv = pool[0] if pool else None
            print(f"[auto-detect] PlantVillage candidates: {cands} -> chose {pv}")
            if pv and "augmented" in pv.lower():
                print("[WARNING] Chosen PlantVillage folder looks like an OFFLINE-AUGMENTED copy. "
                      "That leaks near-identical images across splits. Prefer the original (color/) images.")
        if pdoc is None:
            for dirpath, dirnames, _ in os.walk(kaggle):
                if "plantdoc" in dirpath.lower().replace("_", "").replace("-", "") and \
                        all(d in dirnames for d in split_dirs):
                    pdoc = dirpath
                    break
            print(f"[auto-detect] PlantDoc root -> {pdoc}")
    if pv is None or pdoc is None:
        raise FileNotFoundError(
            "Could not locate datasets. Set paths.plantvillage_root and paths.plantdoc_root in the config "
            "(PlantVillage = folder containing class folders; PlantDoc = folder containing train/ and test/)."
        )
    return Path(pv), Path(pdoc)


def plantdoc_class_folders(plantdoc_root, split_dirs) -> dict:
    """{class_folder: [relative image paths 'split/class/file']} across the configured PlantDoc split folders."""
    out = {}
    for sd in split_dirs:
        sub = Path(plantdoc_root) / sd
        if not sub.is_dir():
            raise FileNotFoundError(f"PlantDoc split folder not found: {sub}")
        for cls, files in list_class_folders(sub).items():
            out.setdefault(cls, []).extend(f"{sd}/{cls}/{f}" for f in files)
    return out


# ----------------------------------------------------------------------------------------------
# 2. Class-name normalisation and the PV / PlantDoc intersection
# ----------------------------------------------------------------------------------------------
def _clean(text: str) -> str:
    """lower-case, turn separators/brackets into spaces, collapse whitespace."""
    text = re.sub(r"[_\-,()\[\]]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _normalise_condition(cond_text: str, crop: str, cfg) -> str:
    """Condition text -> canonical condition: strip the crop word, ignore filler words, apply aliases."""
    d = cfg["data"]
    cond = _clean(cond_text)
    if cond.startswith(crop + " "):          # "tomato mosaic virus" -> "mosaic virus"
        cond = cond[len(crop) + 1:]
    cond = " ".join(w for w in cond.split() if w not in set(d["ignore_words"]))
    cond = d["condition_aliases"].get(cond, cond)
    return cond or "healthy"                  # PlantDoc names a healthy leaf just "Apple leaf"


def _canon_crop(crop_text: str, cfg) -> str:
    crop_text = re.sub(r"\(.*?\)", "", crop_text)   # "Corn_(maize)" -> "Corn_"
    crop = _clean(crop_text)
    return cfg["data"]["crop_aliases"].get(crop, crop)


def parse_plantvillage_folder(name: str, cfg) -> tuple:
    """'Tomato___Late_blight' -> ('tomato', 'late blight'). PV uses '___' between crop and condition."""
    if SEP in name:
        crop_raw, cond_raw = name.split(SEP, 1)
    else:  # unexpected layout: treat the first word as crop, and say so
        print(f"[WARNING] PlantVillage folder {name!r} has no '___'; using first word as crop")
        parts = re.split(r"[_ ]", name, maxsplit=1)
        crop_raw, cond_raw = parts[0], parts[1] if len(parts) > 1 else ""
    crop = _canon_crop(crop_raw, cfg)
    return crop, _normalise_condition(cond_raw, crop, cfg)


def parse_plantdoc_folder(name: str, cfg, crop_vocab: list) -> tuple:
    """'Tomato leaf late blight' -> ('tomato', 'late blight'). PlantDoc has no separator, so we look for a known crop
    (vocabulary taken from PlantVillage + aliases) and treat the remaining words as the condition."""
    text = _clean(name)
    for crop_text in sorted(crop_vocab, key=len, reverse=True):          # longest first: "bell pepper" before "pepper"
        m = re.search(rf"\b{re.escape(crop_text)}\b", text)
        if m:
            crop = cfg["data"]["crop_aliases"].get(crop_text, crop_text)
            rest = (text[:m.start()] + " " + text[m.end():]).strip()
            return crop, _normalise_condition(rest, crop, cfg)
    return None, None                                                   # crop not recognised -> reported as dropped


def build_class_mapping(pv_classes, pd_classes, cfg) -> tuple:
    """Intersect the two datasets on (crop, condition). Returns (mapping_df, dropped_df).

    The intersection is COMPUTED from the real folder names; the printed table is what the team must verify.
    """
    pv_parsed = {f: parse_plantvillage_folder(f, cfg) for f in pv_classes}
    crop_vocab = sorted({c for c, _ in pv_parsed.values()} | set(cfg["data"]["crop_aliases"].keys()))
    pd_parsed = {f: parse_plantdoc_folder(f, cfg, crop_vocab) for f in pd_classes}

    pv_by_key = {}
    for f, key in pv_parsed.items():
        pv_by_key.setdefault(key, []).append(f)
    pd_by_key = {}
    for f, key in pd_parsed.items():
        if key[0] is not None:
            pd_by_key.setdefault(key, []).append(f)

    shared = sorted(set(pv_by_key) & set(pd_by_key))
    rows = [dict(class_id=i, class_name=f"{c}{SEP}{k}", crop=c, condition=k,
                 plantvillage_folder=";".join(pv_by_key[(c, k)]), plantdoc_folders=";".join(pd_by_key[(c, k)]))
            for i, (c, k) in enumerate(shared)]
    mapping = pd.DataFrame(rows)

    dropped = []
    for key, fs in pv_by_key.items():
        if key not in pd_by_key:
            dropped += [dict(side="plantvillage_only", folder=f, parsed_crop=key[0], parsed_condition=key[1]) for f in fs]
    for f, key in pd_parsed.items():
        if key[0] is None or key not in pv_by_key:
            dropped.append(dict(side="plantdoc_only", folder=f, parsed_crop=key[0], parsed_condition=key[1]))
    return mapping, pd.DataFrame(dropped)


# ----------------------------------------------------------------------------------------------
# 3. Near-duplicate detection (perceptual hash) and leakage-aware splitting
# ----------------------------------------------------------------------------------------------
# The 8 symmetries of a square (4 rotations x optional flip). PlantVillage re-photographs the same leaf in
# different orientations, so we compare against all of them.
_DIHEDRAL = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
             Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
             Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE]


def _phash_job(args):
    """Worker: perceptual hashes (64-bit ints) of one image, optionally for all 8 orientations."""
    import imagehash
    path, dihedral = args
    try:
        img = Image.open(path)
        img.draft("L", (128, 128))                        # JPEG fast-decode at reduced size
        img = img.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
        ops = _DIHEDRAL if dihedral else _DIHEDRAL[:1]
        return [int(str(imagehash.phash(img if op is None else img.transpose(op))), 16) for op in ops]
    except Exception as e:                                # corrupt file: unique pseudo-hash so it never groups
        print(f"[WARNING] could not hash {path}: {e}")
        return [int.from_bytes(os.urandom(8), "big")]


def compute_hashes(df, root, cfg, cache_path: Path, num_proc=4) -> list:
    """Hash every image (cached on disk, because 54k images take a few minutes)."""
    dihedral = cfg["data"]["phash_dihedral"]
    cache = {}
    if cache_path.exists():
        c = pd.read_csv(cache_path, dtype=str)
        cache = {r.path: [int(h, 16) for h in r.hashes.split(";")] for r in c.itertuples()}
    todo = [p for p in df["path"] if p not in cache or (dihedral and len(cache[p]) != 8)]
    if todo:
        jobs = [(str(Path(root) / p), dihedral) for p in todo]
        with Pool(num_proc) as pool:
            for p, h in zip(todo, tqdm(pool.imap(_phash_job, jobs, chunksize=64), total=len(jobs), desc="phash")):
                cache[p] = h
        pd.DataFrame({"path": list(cache), "hashes": [";".join(f"{x:016x}" for x in v) for v in cache.values()]}
                     ).to_csv(cache_path, index=False)
    return [cache[p] for p in df["path"]]


def group_near_duplicates(hashes: list, max_dist: int) -> np.ndarray:
    """Union images whose perceptual hashes are within `max_dist` bits. Returns a group id per image.

    Brute force is 54k x 54k comparisons (too slow in Python). Trick (multi-index hashing): cut the 64 bits into
    (max_dist+1) blocks. If two hashes differ in <= max_dist bits, at least ONE block is identical (pigeonhole).
    So we only compare images that share a block value, then verify with the exact Hamming distance.
    """
    n = len(hashes)
    nblocks = max_dist + 1
    bounds = np.linspace(0, 64, nblocks + 1).astype(int)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    buckets = [dict() for _ in range(nblocks)]            # block value -> list of (image idx, hash) over ALL orientations
    for i, hs in enumerate(hashes):
        for h in hs:
            for b in range(nblocks):
                key = (h >> int(bounds[b])) & ((1 << int(bounds[b + 1] - bounds[b])) - 1)
                buckets[b].setdefault(key, []).append((i, h))

    for i, hs in enumerate(hashes):
        h0 = hs[0]                                         # query with the unrotated hash against everyone's orientations
        for b in range(nblocks):
            key = (h0 >> int(bounds[b])) & ((1 << int(bounds[b + 1] - bounds[b])) - 1)
            for j, hj in buckets[b].get(key, ()):
                if j != i and find(i) != find(j) and (h0 ^ hj).bit_count() <= max_dist:
                    parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n)])
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids


def group_stratified_split(df, ratios: dict, seed: int) -> pd.Series:
    """Assign each GROUP to train/val/test so each class hits ~70/15/15 and no group is split.

    Greedy per class: biggest groups first, each goes to the partition currently furthest below its target.
    Simple to explain, and group integrity is guaranteed by construction.
    """
    rng = np.random.RandomState(seed)
    split = pd.Series("", index=df.index)
    parts = list(ratios)
    # A group occasionally spans two classes (look-alike images). It must still land in ONE partition, so each
    # group is stratified by its majority label.
    group_label = df.groupby("group_id")["label"].agg(lambda s: s.value_counts().idxmax())
    df = df.assign(strat_label=df["group_id"].map(group_label))
    for _, cdf in df.groupby("strat_label"):
        groups = [g.index.tolist() for _, g in cdf.groupby("group_id")]
        rng.shuffle(groups)                                # random tie-break, reproducible via seed
        groups.sort(key=len, reverse=True)
        target = {p: ratios[p] * len(cdf) for p in parts}
        count = {p: 0 for p in parts}
        for idxs in groups:
            p = max(parts, key=lambda q: target[q] - count[q])
            split.loc[idxs] = p
            count[p] += len(idxs)
    return split


@dataclass
class DataBundle:
    """Everything training needs, and nothing about PlantDoc images."""
    cfg: dict
    pv_root: Path
    plantdoc_root: Path
    mapping: pd.DataFrame
    class_names: list
    splits: dict            # {"train": df, "val": df, "test": df}
    dup_report: dict

    @property
    def num_classes(self):
        return len(self.class_names)

    def train_class_counts(self) -> np.ndarray:
        return np.bincount(self.splits["train"]["label"], minlength=self.num_classes)


def prepare_data(cfg, verbose=True) -> DataBundle:
    """Discover datasets -> class mapping -> (smoke subsample) -> near-dup groups -> grouped stratified split.

    Splits are saved to results/splits/*.csv and reused while the settings are unchanged, so every seed and
    every later ablation sees exactly the same partition.
    """
    rdir = results_dir(cfg)
    pv_root, pd_root = resolve_dataset_roots(cfg)
    pv_classes = list_class_folders(pv_root)
    pd_classes = plantdoc_class_folders(pd_root, cfg["data"]["plantdoc_dirs"])   # folder NAMES only (no image use)

    mapping, dropped = build_class_mapping(list(pv_classes), list(pd_classes), cfg)
    if mapping.empty:
        raise RuntimeError("Class intersection is empty - check the printed folder names and the alias tables.")
    mapping.to_csv(rdir / "class_mapping.csv", index=False)
    dropped.to_csv(rdir / "class_dropped.csv", index=False)
    if verbose:
        print(f"\nClass intersection: {len(mapping)} classes (saved to {rdir/'class_mapping.csv'})")
        print(mapping.to_string(index=False))
        print("\nDROPPED (present in only one dataset):")
        print(dropped.to_string(index=False) if len(dropped) else "  none")

    class_names = mapping["class_name"].tolist()
    # one row per PV image in the intersection
    rows = []
    for r in mapping.itertuples():
        for folder in r.plantvillage_folder.split(";"):
            rows += [dict(path=f"{folder}/{f}", label=r.class_id, class_name=r.class_name,
                          crop=r.crop, condition=r.condition) for f in pv_classes[folder]]
    all_df = pd.DataFrame(rows)

    smoke = cfg["smoke"]["enabled"]
    if smoke:  # take a small, class-spread sample so the pipeline check is fast
        per_class = max(3, math.ceil(cfg["smoke"]["n_images"] / len(class_names)))
        # shuffle, then keep the first `per_class` rows of each class (version-proof, unlike groupby.apply)
        all_df = (all_df.sample(frac=1, random_state=cfg["split_seed"])
                  .groupby("label").head(per_class).sort_values("path").reset_index(drop=True))

    meta = dict(split_seed=cfg["split_seed"], ratios=cfg["data"]["split_ratios"], n_images=len(all_df),
                phash_max_distance=cfg["data"]["phash_max_distance"], dihedral=cfg["data"]["phash_dihedral"],
                classes=class_names, smoke=smoke)
    sdir = rdir / "splits"
    sdir.mkdir(exist_ok=True)
    if (sdir / "meta.json").exists() and json.loads((sdir / "meta.json").read_text()) == json.loads(json.dumps(meta)):
        splits = {s: pd.read_csv(sdir / f"{s}.csv") for s in ("train", "val", "test")}
        dup_report = json.loads((rdir / "duplicate_report.json").read_text())
        if verbose:
            print(f"\nReusing saved splits from {sdir} (settings unchanged).")
        return DataBundle(cfg, pv_root, pd_root, mapping, class_names, splits, dup_report)

    hashes = compute_hashes(all_df, pv_root, cfg, rdir / "phash_cache.csv", num_proc=max(1, min(8, os.cpu_count() or 1)))
    all_df["group_id"] = group_near_duplicates(hashes, cfg["data"]["phash_max_distance"])
    sizes = all_df.groupby("group_id").size()
    cross = (all_df.groupby("group_id")["label"].nunique() > 1).sum()
    dup_report = dict(n_images=len(all_df), n_groups=int(len(sizes)), n_groups_with_duplicates=int((sizes > 1).sum()),
                      n_images_in_duplicate_groups=int(sizes[sizes > 1].sum()), largest_group=int(sizes.max()),
                      n_groups_spanning_multiple_classes=int(cross),
                      phash_max_distance=cfg["data"]["phash_max_distance"], dihedral=cfg["data"]["phash_dihedral"])
    (rdir / "duplicate_report.json").write_text(json.dumps(dup_report, indent=2))
    if verbose:
        print("\nNear-duplicate report:", json.dumps(dup_report, indent=2))
        if dup_report["largest_group"] > 0.02 * len(all_df):
            print("[WARNING] Largest duplicate group is >2% of the data: the hash threshold may be too loose.")

    all_df["split"] = group_stratified_split(all_df, cfg["data"]["split_ratios"], cfg["split_seed"])
    splits = {s: all_df[all_df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    for s, d in splits.items():
        d.to_csv(sdir / f"{s}.csv", index=False)
    (sdir / "meta.json").write_text(json.dumps(meta, indent=2))
    # sanity: no group may appear in two partitions
    gsplits = all_df.groupby("group_id")["split"].nunique()
    assert (gsplits == 1).all(), "BUG: a duplicate group was split across partitions"
    if verbose:
        print({s: len(d) for s, d in splits.items()}, "(images per split)")
        empty = [(c, s) for s, d in splits.items() for c in range(len(class_names)) if (d["label"] == c).sum() == 0]
        if empty:
            print(f"[WARNING] {len(empty)} (class, split) cells are empty (tiny classes): {empty[:10]}")
    return DataBundle(cfg, pv_root, pd_root, mapping, class_names, splits, dup_report)


# ----------------------------------------------------------------------------------------------
# 4. Transforms, Dataset, loaders
# ----------------------------------------------------------------------------------------------
def build_transforms(image_size: int, train: bool):
    """Train: light, label-preserving augmentation. Eval: deterministic resize + center crop + ImageNet normalisation."""
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)  # the pretrained weights expect this
    if train:
        return transforms.Compose([
            # scale>=0.6: a tiny crop could remove the lesion and leave a "healthy-looking" image with a disease label
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),                  # leaves have no canonical "up"
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),      # small: colour is a real disease cue (yellowing, browning)
            transforms.ToTensor(), norm])
    return transforms.Compose([
        transforms.Resize(int(round(image_size * 256 / 224))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(), norm])


class LeafDataset(Dataset):
    """Returns a dict with the image tensor, integer label and the metadata (crop, condition).

    Phase 2 will turn crop/condition into a text note; carrying them now means no data-code rewrite later.
    """
    def __init__(self, df: pd.DataFrame, root, transform):
        self.df, self.root, self.transform = df.reset_index(drop=True), Path(root), transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r["path"]).convert("RGB")   # PlantDoc contains RGBA/greyscale files
        return {"image": self.transform(img), "label": int(r["label"]), "crop": r["crop"],
                "condition": r["condition"], "index": i}


def _loader(ds, cfg, shuffle, seed):
    d = cfg["data"]
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=d["batch_size"], shuffle=shuffle, num_workers=d["num_workers"],
                      worker_init_fn=seed_worker, generator=g, pin_memory=torch.cuda.is_available(),
                      drop_last=shuffle and len(ds) > d["batch_size"])   # drop a ragged last batch only when training


def make_loaders(bundle: DataBundle, cfg, seed: int) -> dict:
    """In-distribution loaders ONLY (train/val/test of PlantVillage)."""
    size = cfg["data"]["image_size"]
    return {s: _loader(LeafDataset(df, bundle.pv_root, build_transforms(size, train=(s == "train"))),
                       cfg, shuffle=(s == "train"), seed=seed)
            for s, df in bundle.splits.items()}


# ----------------------------------------------------------------------------------------------
# 5. SHIFTED TEST SET (PlantDoc). Imported by evaluate.py / notebook only - never by train.py.
# ----------------------------------------------------------------------------------------------
def build_shifted_test(bundle: DataBundle, cfg, seed: int = 0):
    """PlantDoc images for the intersection classes -> (dataframe, DataLoader). There is NO train/val variant.

    Labels are mapped to the SAME class ids as PlantVillage via the saved class mapping.
    """
    folders = plantdoc_class_folders(bundle.plantdoc_root, cfg["data"]["plantdoc_dirs"])
    rows = []
    for r in bundle.mapping.itertuples():
        for folder in r.plantdoc_folders.split(";"):
            rows += [dict(path=p, label=r.class_id, class_name=r.class_name, crop=r.crop, condition=r.condition)
                     for p in folders[folder]]
    df = pd.DataFrame(rows)
    if cfg["smoke"]["enabled"]:
        df = df.sample(min(len(df), cfg["smoke"]["n_shifted_images"]), random_state=cfg["split_seed"]).reset_index(drop=True)
    ds = LeafDataset(df, bundle.plantdoc_root, build_transforms(cfg["data"]["image_size"], train=False))
    return df, _loader(ds, cfg, shuffle=False, seed=seed)


# ----------------------------------------------------------------------------------------------
# 6. Data analysis outputs (PNG + CSV)
# ----------------------------------------------------------------------------------------------
def analyse_data(bundle: DataBundle, shifted_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Class distribution (train / val / test / PlantDoc), imbalance ratios, bar chart. Saved to results/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rdir = results_dir(cfg)
    C = bundle.num_classes
    tab = pd.DataFrame({"class_name": bundle.class_names})
    for s, d in bundle.splits.items():
        tab[f"pv_{s}"] = np.bincount(d["label"], minlength=C)
    tab["plantdoc"] = np.bincount(shifted_df["label"], minlength=C)
    tab.to_csv(rdir / "class_distribution.csv", index=False)

    def ratio(counts):   # imbalance ratio = largest class / smallest NON-EMPTY class
        nz = counts[counts > 0]
        return float(nz.max() / nz.min()) if len(nz) else float("nan")
    imb = {"pv_train_imbalance_ratio": ratio(tab["pv_train"].values),
           "plantdoc_imbalance_ratio": ratio(tab["plantdoc"].values),
           "plantdoc_empty_classes": int((tab["plantdoc"] == 0).sum())}
    (rdir / "imbalance.json").write_text(json.dumps(imb, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(14, max(4, 0.28 * C)), sharey=True)
    y = np.arange(C)
    ax[0].barh(y, tab["pv_train"], color="tab:blue")
    ax[0].set_title(f"PlantVillage train (imbalance {imb['pv_train_imbalance_ratio']:.1f}x)")
    ax[1].barh(y, tab["plantdoc"], color="tab:orange")
    ax[1].set_title(f"PlantDoc shifted test (imbalance {imb['plantdoc_imbalance_ratio']:.1f}x)")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(bundle.class_names, fontsize=7)
    ax[0].invert_yaxis()
    for a in ax:
        a.set_xlabel("images")
    fig.tight_layout()
    fig.savefig(rdir / "class_distribution.png", dpi=150)
    plt.close(fig)
    return tab


def plot_shift_grid(bundle: DataBundle, shifted_df: pd.DataFrame, cfg, n_classes=6, n_per_side=3):
    """Same class, PlantVillage (left) vs PlantDoc (right): a visual of the domain shift."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rdir = results_dir(cfg)
    rng = np.random.RandomState(cfg["split_seed"])
    train = bundle.splits["train"]
    both = [c for c in range(bundle.num_classes)
            if (train["label"] == c).sum() >= n_per_side and (shifted_df["label"] == c).sum() >= n_per_side]
    chosen = rng.choice(both, size=min(n_classes, len(both)), replace=False) if both else []
    if len(chosen) == 0:
        print("[WARNING] no class has enough images in both datasets for the shift grid")
        return
    fig, axes = plt.subplots(len(chosen), 2 * n_per_side, figsize=(2.2 * 2 * n_per_side, 2.3 * len(chosen)))
    axes = np.atleast_2d(axes)
    for row, c in enumerate(chosen):
        for side, (df, root, tag) in enumerate([(train, bundle.pv_root, "PlantVillage"),
                                                (shifted_df, bundle.plantdoc_root, "PlantDoc")]):
            sel = df[df["label"] == c].sample(n_per_side, random_state=int(rng.randint(1e6)))
            for k, p in enumerate(sel["path"]):
                a = axes[row, side * n_per_side + k]
                a.imshow(Image.open(Path(root) / p).convert("RGB"))
                a.axis("off")
                if k == 0:
                    a.set_title(f"{tag}: {bundle.class_names[c]}", fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(rdir / "shift_grid.png", dpi=130)
    plt.close(fig)
