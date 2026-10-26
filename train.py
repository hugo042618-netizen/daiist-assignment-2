"""
Assignment 2 training pipeline: hard hat compliance from site camera snapshots.

    uv run python main.py train

Start to finish, with no manual steps:

1. Download the dataset version from Roboflow Universe (key via api_key.py).
2. Describe it: images and instances per class and split, object sizes,
   violation and compliant images, and near-duplicate photos shared between
   train and the val/test splits.
3. Finetune yolo11n from the COCO checkpoint and copy best.pt into weights/.
4. Evaluate on the test split, with one evaluator for all three models: the
   finetuned model, the COCO checkpoint zero-shot, and a predict-nothing
   baseline. Per-class AP, precision and recall, plus a confusion matrix.
5. Score every validation and test snapshot (highest bare-head confidence),
   sweep the threshold on validation, choose the single threshold and the
   two-threshold review band with the lowest expected cost, and check both on
   test.
6. Save everything app.py needs into artifacts/, so the app never trains.

Settings can be overridden with environment variables: DEVICE (cpu, mps, 0),
EPOCHS, IMGSZ, BATCH. DATASET_DIR points at an already downloaded copy of the
same dataset version, for re-running without network access. SKIP_TRAIN=1
reuses the weights of the last finetune in runs/finetune and redoes steps 4 to
6 only, for when a cost assumption changes but the model does not.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.metrics import ap_per_class  # noqa: E402

import costing  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
WEIGHTS = ROOT / "weights"
ARTIFACTS = ROOT / "artifacts"
EXAMPLES = ROOT / "examples"
RUNS = ROOT / "runs"

# ---------------------------------------------------------------- dataset ---
WORKSPACE = "joseph-nelson"
PROJECT = "hard-hat-workers"
VERSION = 5  # raw_HeadHelmetClasses: raw images (no resize), 70/20/10 split, classes head and helmet
UNIVERSE_URL = f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{VERSION}"

# A snapshot is a violation (the positive class) if any of these is found in it.
# `head` is a head without a hard hat. Snapshots with only `helmet` boxes are compliant.
POSITIVE_CLASSES = ["head"]
# COCO classes that mean the same thing as one of ours. COCO has `person` but no
# head and no helmet class, so the zero-shot model has no box it is allowed to
# claim. That is the honest mapping, and the point of the baseline. Its person
# boxes are still used as an image-level proxy in save_metrics.
COCO_TO_DATASET: dict[str, str] = {}
COCO_PROXY_CLASS = "person"

# --------------------------------------------------------------- training ---
BASE_MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("EPOCHS", 20))  # 57 minutes on an M4 Pro (MPS); val mAP50 is already 0.91 after one epoch
IMGSZ = int(os.environ.get("IMGSZ", 640))
BATCH = int(os.environ.get("BATCH", 16))
PATIENCE = 15
SEED = 0
SKIP_TRAIN = os.environ.get("SKIP_TRAIN", "") not in {"", "0"}

# ----------------------------------------------------------- costs/review ---
# Every figure is argued in REPORT.md. The app reads them back from artifacts/.
COSTS = costing.Costs(
    fn=50.0,              # bare head missed: expected fine (about EUR 25) plus expected injury cost (about EUR 25)
    fp=4.0,               # supervisor sent for nothing: about 10 minutes at EUR 25 an hour
    tp=4.0,               # supervisor sent to a real violation: the same walk, now useful
    tn=0.0,               # compliant snapshot passed: nothing happens
    hourly_cost=20.0,     # loaded cost of a monitoring operator hour
    review_seconds=10.0,  # operator looks at one snapshot
    base_rate=0.05,       # assumed share of snapshots with a bare head
    cases_per_day=6000,   # 10 cameras x 1 snapshot a minute x 10 hours
    reviewer_hours_per_day=6.5,
)
MAX_REVIEWERS = 1.0  # one monitoring operator's shift goes to review, no more

IOUV = np.linspace(0.5, 0.95, 10)


def pick_device() -> str:
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


def log(message: str) -> None:
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ============================================================ 1. download ===
def download_dataset() -> Path:
    if os.environ.get("DATASET_DIR"):
        location = Path(os.environ["DATASET_DIR"]).resolve()
        log(f"Using the local copy of {PROJECT} v{VERSION} at {location}")
    else:
        from roboflow import Roboflow

        from api_key import load_api_key

        location = DATASETS / f"{PROJECT}-v{VERSION}"
        log(f"Downloading {WORKSPACE}/{PROJECT} version {VERSION} from Roboflow Universe")
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("yolov11", location=str(location))

    source = yaml.safe_load((location / "data.yaml").read_text())
    names = source["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    local = {
        "path": str(location),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: n for i, n in enumerate(names)},
    }
    data_yaml = location / "data_local.yaml"
    data_yaml.write_text(yaml.safe_dump(local, sort_keys=False))
    missing = [c for c in POSITIVE_CLASSES if c not in names]
    if missing:
        raise SystemExit(f"Positive classes {missing} are not in the dataset classes {names}")
    return data_yaml


def class_names(data_yaml: Path) -> list[str]:
    names = yaml.safe_load(data_yaml.read_text())["names"]
    return [names[k] for k in sorted(names)]


def split_images(data_yaml: Path, split: str) -> list[Path]:
    root = Path(yaml.safe_load(data_yaml.read_text())["path"])
    folder = root / {"train": "train", "val": "valid", "test": "test"}[split] / "images"
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def label_path(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def read_labels(image: Path) -> dict:
    """Ground truth boxes in pixels. Polygon rows are reduced to their bounding box."""
    with Image.open(image) as im:
        w, h = im.size
    cls, boxes, rel = [], [], []
    path = label_path(image)
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c, values = int(float(parts[0])), np.array(parts[1:], dtype=float)
            if len(values) == 4:
                cx, cy, bw, bh = values
            else:  # polygon x1 y1 x2 y2 ...
                xs, ys = values[0::2], values[1::2]
                cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
                bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
            cls.append(c)
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            rel.append([bw, bh])
    return {
        "cls": np.array(cls, dtype=int),
        "xyxy": np.array(boxes, dtype=float).reshape(-1, 4),
        "rel_wh": np.array(rel, dtype=float).reshape(-1, 2),
        "size": (w, h),
    }


# ============================================================ 2. describe ===
def dhash(image: Path) -> int:
    with Image.open(image) as im:
        small = np.asarray(im.convert("L").resize((9, 8)), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicates(test: list[Path], train: list[Path], max_bits: int = 4) -> int:
    """Test photos whose 64-bit difference hash is within max_bits of some train photo."""
    train_hashes = np.array([dhash(p) for p in train], dtype=np.uint64)
    count = 0
    for p in test:
        x = np.bitwise_xor(train_hashes, np.uint64(dhash(p)))
        bits = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        count += int((bits <= max_bits).any())
    return count


def describe_dataset(data_yaml: Path) -> dict:
    names = class_names(data_yaml)
    pos_ids = {names.index(c) for c in POSITIVE_CLASSES}
    stats = {"classes": names, "positive_classes": POSITIVE_CLASSES, "splits": {}}
    size_rows = []
    for split in ("train", "val", "test"):
        images = split_images(data_yaml, split)
        inst = np.zeros(len(names), dtype=int)
        imgs_with = np.zeros(len(names), dtype=int)
        background = positive = 0
        sizes = set()
        for image in images:
            gt = read_labels(image)
            sizes.add(gt["size"])
            for c in gt["cls"]:
                inst[c] += 1
            for c in set(gt["cls"].tolist()):
                imgs_with[c] += 1
            if len(gt["cls"]) == 0:
                background += 1
            if pos_ids & set(gt["cls"].tolist()):
                positive += 1
            for c, (bw, bh) in zip(gt["cls"], gt["rel_wh"]):
                size_rows.append({"split": split, "class": names[c], "rel_area": bw * bh, "rel_w": bw, "rel_h": bh})
        stats["splits"][split] = {
            "images": len(images),
            "instances": dict(zip(names, inst.tolist())),
            "images_with_class": dict(zip(names, imgs_with.tolist())),
            "background_images": background,
            "positive_images": positive,
            "negative_images": len(images) - positive,
            "positive_share": positive / max(len(images), 1),
            "image_sizes": sorted(f"{w}x{h}" for w, h in sizes)[:5],
        }
    sizes = pd.DataFrame(size_rows)
    if len(sizes):
        by_class = sizes.groupby("class")["rel_area"].describe(percentiles=[0.1, 0.5, 0.9])
        stats["object_size_rel_area"] = {
            c: {
                "p10": float(by_class.loc[c, "10%"]),
                "median": float(by_class.loc[c, "50%"]),
                "p90": float(by_class.loc[c, "90%"]),
                "share_small_lt_1pct": float((sizes.loc[sizes["class"] == c, "rel_area"] < 0.01).mean()),
            }
            for c in by_class.index
        }
    stats["test_near_duplicates_of_train"] = near_duplicates(
        split_images(data_yaml, "test"), split_images(data_yaml, "train")
    )
    stats["val_near_duplicates_of_train"] = near_duplicates(
        split_images(data_yaml, "val"), split_images(data_yaml, "train")
    )
    write_json(ARTIFACTS / "dataset_stats.json", stats)
    for split, s in stats["splits"].items():
        print(f"  {split:5s} {s['images']:5d} images, {s['positive_images']} with a bare head, "
              f"{s['background_images']} unlabelled, instances {s['instances']}")
    print(f"  near-duplicates of a train photo: test {stats['test_near_duplicates_of_train']}, "
          f"val {stats['val_near_duplicates_of_train']}")
    return stats


# ============================================================ 3. finetune ===
def machine_name() -> str:
    chip = platform.processor() or "n/a"
    if platform.system() == "Darwin":
        try:
            import subprocess
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, check=True).stdout.strip() or chip
        except (OSError, subprocess.CalledProcessError):
            pass
    return f"{platform.system()} {platform.machine()} ({chip})"


def finetune(data_yaml: Path) -> Path:
    RUNS.mkdir(exist_ok=True)
    if SKIP_TRAIN:
        if not (WEIGHTS / "best.pt").exists() or not (ARTIFACTS / "training_config.json").exists():
            raise SystemExit("SKIP_TRAIN=1 needs weights/best.pt and artifacts/training_config.json "
                             "from a full run. Run without SKIP_TRAIN first.")
        log("SKIP_TRAIN=1: reusing weights/best.pt. Training history and config are kept as they are.")
        return WEIGHTS / "best.pt"
    model = YOLO(str(RUNS / BASE_MODEL))
    started = time.time()
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        seed=SEED,
        deterministic=True,
        device=DEVICE,
        project=str(RUNS),
        name="finetune",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    minutes = (time.time() - started) / 60
    run_dir = RUNS / "finetune"
    WEIGHTS.mkdir(exist_ok=True)
    shutil.copy(run_dir / "weights" / "best.pt", WEIGHTS / "best.pt")
    history = pd.read_csv(run_dir / "results.csv")
    history.columns = [c.strip() for c in history.columns]
    history.to_csv(ARTIFACTS / "training_history.csv", index=False)
    write_json(ARTIFACTS / "training_config.json", {
        "base_model": BASE_MODEL,
        "epochs_requested": EPOCHS,
        "epochs_run": int(len(history)),
        "best_epoch": int(history["metrics/mAP50-95(B)"].idxmax() + 1),
        "imgsz": IMGSZ,
        "batch": BATCH,
        "patience": PATIENCE,
        "seed": SEED,
        "device": DEVICE,
        "minutes": round(minutes, 1),
        "machine": machine_name(),
        "torch": torch.__version__,
        "dataset": f"{WORKSPACE}/{PROJECT}/{VERSION}",
    })
    return WEIGHTS / "best.pt"


# ============================================================ 4. evaluate ===
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (M,4) and every box in b (N,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_matches(iou: np.ndarray, threshold: float) -> np.ndarray:
    """(gt, pred) index pairs, highest IoU first, each gt and pred used once."""
    pairs = np.argwhere(iou >= threshold)
    if len(pairs) > 1:
        pairs = pairs[iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return pairs


def correct_matrix(pred: dict, gt: dict) -> np.ndarray:
    """(N preds, 10 IoU thresholds) true/false, same rule as the Ultralytics validator."""
    correct = np.zeros((len(pred["cls"]), len(IOUV)), dtype=bool)
    if len(pred["cls"]) == 0 or len(gt["cls"]) == 0:
        return correct
    iou = iou_matrix(gt["xyxy"], pred["xyxy"]) * (gt["cls"][:, None] == pred["cls"][None, :])
    for i, t in enumerate(IOUV):
        pairs = greedy_matches(iou, t)
        if len(pairs):
            correct[pairs[:, 1], i] = True
    return correct


def predict(model: YOLO, images: list[Path], class_map: dict[int, int] | None = None) -> list[dict]:
    """Raw detections at conf 0.001 (as validation does), optionally remapped to our class ids."""
    out = []
    for start in range(0, len(images), 16):
        batch = [str(p) for p in images[start:start + 16]]
        for result in model.predict(batch, conf=0.001, iou=0.7, imgsz=IMGSZ, max_det=300,
                                    device=DEVICE, verbose=False):
            boxes = result.boxes
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            if class_map is not None:
                keep = np.array([c in class_map for c in cls], dtype=bool)
                cls = np.array([class_map[c] for c in cls[keep]], dtype=int)
                conf, xyxy = conf[keep], xyxy[keep]
            out.append({"cls": cls, "conf": conf, "xyxy": xyxy.reshape(-1, 4)})
    return out


def detection_metrics(preds: list[dict], gts: list[dict], names: list[str], op_threshold: float | None = None) -> dict:
    """AP50, AP50-95, precision and recall per class, from one shared evaluator."""
    tps, confs, pcls = [], [], []
    tcls = np.concatenate([g["cls"] for g in gts]) if gts else np.zeros(0, dtype=int)
    for p, g in zip(preds, gts):
        tps.append(correct_matrix(p, g))
        confs.append(p["conf"])
        pcls.append(p["cls"])
    tp = np.concatenate(tps) if tps else np.zeros((0, len(IOUV)), dtype=bool)
    conf = np.concatenate(confs) if confs else np.zeros(0)
    pred_cls = np.concatenate(pcls) if pcls else np.zeros(0, dtype=int)

    per_class = {n: {"instances": int((tcls == i).sum()), "precision": 0.0, "recall": 0.0,
                     "ap50": 0.0, "ap50_95": 0.0} for i, n in enumerate(names)}
    if len(tcls):
        _, _, p, r, _, ap, unique, *_ = ap_per_class(tp, conf, pred_cls, tcls, names=dict(enumerate(names)))
        for k, c in enumerate(unique):
            per_class[names[c]].update(precision=float(p[k]), recall=float(r[k]),
                                       ap50=float(ap[k, 0]), ap50_95=float(ap[k].mean()))
        overall = {"map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean()),
                   "precision": float(p.mean()), "recall": float(r.mean())}
    else:
        overall = {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0}

    if op_threshold is not None:  # precision/recall at IoU 0.5 for boxes we would actually show
        keep = conf >= op_threshold
        for i, n in enumerate(names):
            mine = keep & (pred_cls == i)
            hits = int(tp[mine, 0].sum())
            n_pred, n_gt = int(mine.sum()), int((tcls == i).sum())
            per_class[n]["precision_at_op"] = hits / n_pred if n_pred else 0.0
            per_class[n]["recall_at_op"] = hits / n_gt if n_gt else 0.0
            per_class[n]["boxes_at_op"] = n_pred
    return {"overall": overall, "per_class": per_class}


def detection_confusion(preds: list[dict], gts: list[dict], nc: int, conf_t: float, iou_t: float = 0.5) -> np.ndarray:
    """Box-level confusion matrix [predicted, true], last row/column = background."""
    cm = np.zeros((nc + 1, nc + 1), dtype=int)
    for p, g in zip(preds, gts):
        keep = p["conf"] >= conf_t
        pc, pb = p["cls"][keep], p["xyxy"][keep]
        pairs = greedy_matches(iou_matrix(g["xyxy"], pb), iou_t) if len(pc) and len(g["cls"]) else np.zeros((0, 2), int)
        for gi, pi in pairs:
            cm[pc[pi], g["cls"][gi]] += 1
        for gi in set(range(len(g["cls"]))) - set(pairs[:, 0].tolist()):
            cm[nc, g["cls"][gi]] += 1
        for pi in set(range(len(pc))) - set(pairs[:, 1].tolist()):
            cm[pc[pi], nc] += 1
    return cm


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Chance that a violation snapshot outscores a compliant one (ties count half)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def image_scores(preds: list[dict], gts: list[dict], images: list[Path], names: list[str]) -> pd.DataFrame:
    """One row per snapshot: does it really show a bare head, and how sure is the model that it does."""
    pos_ids = [names.index(c) for c in POSITIVE_CLASSES]
    rows = []
    for image, p, g in zip(images, preds, gts):
        row = {"image": image.name, "positive": int(bool(set(g["cls"].tolist()) & set(pos_ids)))}
        mask = np.isin(p["cls"], pos_ids)
        row["score"] = float(p["conf"][mask].max()) if mask.any() else 0.0
        for c in POSITIVE_CLASSES:
            m = p["cls"] == names.index(c)
            row[f"score_{c}"] = float(p["conf"][m].max()) if m.any() else 0.0
            row[f"gt_{c}"] = int((g["cls"] == names.index(c)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_models(data_yaml: Path, weights: Path) -> dict:
    names = class_names(data_yaml)
    images = {s: split_images(data_yaml, s) for s in ("val", "test")}
    gts = {s: [read_labels(p) for p in images[s]] for s in images}

    finetuned = YOLO(str(weights))
    preds = {s: predict(finetuned, images[s]) for s in images}

    coco = YOLO(str(RUNS / BASE_MODEL))
    coco_map = {i: names.index(COCO_TO_DATASET[n]) for i, n in coco.names.items() if n in COCO_TO_DATASET}
    zero_shot = predict(coco, images["test"], class_map=coco_map)
    coco_raw = predict(coco, images["test"])  # what COCO thinks it sees, for the report
    proxy_id = next(i for i, n in coco.names.items() if n == COCO_PROXY_CLASS)
    coco_proxy = np.array([float(p["conf"][p["cls"] == proxy_id].max()) if (p["cls"] == proxy_id).any() else 0.0
                           for p in coco_raw])
    nothing = [{"cls": np.zeros(0, int), "conf": np.zeros(0), "xyxy": np.zeros((0, 4))} for _ in images["test"]]

    scores = {s: image_scores(preds[s], gts[s], images[s], names) for s in images}
    for s, frame in scores.items():
        frame.to_csv(ARTIFACTS / f"scores_{s}.csv", index=False)

    # Ultralytics' own validator on the same test split, as a cross-check of our evaluator.
    official = finetuned.val(data=str(data_yaml), split="test", imgsz=IMGSZ, batch=BATCH, device=DEVICE,
                             project=str(RUNS), name="test_eval", exist_ok=True, plots=True, verbose=False)
    coco_seen = pd.Series([coco.names[c] for p in coco_raw for c in p["cls"][p["conf"] >= 0.25]]).value_counts()

    return {
        "names": names,
        "images": images,
        "gts": gts,
        "preds": preds,
        "scores": scores,
        "zero_shot": zero_shot,
        "zero_shot_proxy_scores": coco_proxy,
        "nothing": nothing,
        "official_test": {"map50": float(official.box.map50), "map50_95": float(official.box.map),
                          "precision": float(official.box.mp), "recall": float(official.box.mr)},
        "coco_seen_on_test": coco_seen.head(10).to_dict(),
    }


# ======================================================= 5. cost & review ===
def bootstrap_cost(scores: pd.DataFrame, t_low: float, t_high: float, c: costing.Costs, n: int = 2000) -> dict:
    rng = np.random.default_rng(SEED)
    s, y = scores["score"].to_numpy(), scores["positive"].to_numpy()
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = []
    for _ in range(n):  # resample violation and compliant snapshots separately: their mix is set by the base rate
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        draws.append(costing.evaluate_band(s[idx], y[idx], t_low, t_high, c)["cost_per_1000"])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"low": float(lo), "high": float(hi)}


def do_nothing_cost(c: costing.Costs) -> float:
    return 1000 * (c.base_rate * c.fn + (1 - c.base_rate) * c.tn)


def check_everything_cost(c: costing.Costs) -> float:
    return 1000 * (c.review + c.base_rate * c.tp + (1 - c.base_rate) * c.tn)


def sensitivity(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Re-choose the thresholds on validation under other assumptions, report on test."""
    rows = []
    variants = [("base case", {})]
    variants += [(f"base rate {b:.0%}", {"base_rate": b}) for b in (0.01, 0.02, 0.03, 0.08, 0.10, 0.15)]
    variants += [(f"missed bare head costs EUR {v:.0f}", {"fn": v}) for v in (10, 25, 100, 200)]
    variants += [(f"review takes {v:.0f} s", {"review_seconds": v}) for v in (5, 20, 30)]
    variants += [(f"false alert costs EUR {v:.0f}", {"fp": v}) for v in (2, 8)]
    variants += [(f"caught violation costs EUR {v:.0f}", {"tp": v}) for v in (0, 8)]
    variants += [(f"operator cap {v:g} people", {"max_reviewers": v}) for v in (0.5, 2.0)]
    variants += [("no operator cap", {"max_reviewers": None})]
    for label, change in variants:
        change = dict(change)
        cap = change.pop("max_reviewers", MAX_REVIEWERS)
        c = costing.Costs.from_dict({**COSTS.to_dict(), **change})
        sweep = costing.threshold_sweep(val["score"], val["positive"], c)
        t_single = costing.best_single_threshold(sweep)
        band = costing.band_search(val["score"], val["positive"], c, max_reviewers=cap)
        on_test = costing.evaluate_band(test["score"], test["positive"], band["t_low"], band["t_high"], c)
        alone = costing.evaluate_band(test["score"], test["positive"], t_single, t_single, c)
        rows.append({
            "scenario": label,
            "t_single": t_single,
            "t_low": band["t_low"],
            "t_high": band["t_high"],
            "test_cost_band": on_test["cost_per_1000"],
            "test_cost_model_alone": alone["cost_per_1000"],
            "cost_do_nothing": do_nothing_cost(c),
            "cost_check_everything": check_everything_cost(c),
            "review_share": on_test["review_share"],
            "reviewers_needed": on_test["reviewers_needed"],
            "recall_violation": on_test["recall_violation"],
        })
    return pd.DataFrame(rows)


def break_even(val: pd.DataFrame, test: pd.DataFrame, param: str, lo: float, hi: float, against) -> dict:
    """Value of one cost assumption at which the chosen design (thresholds re-chosen on
    validation under that assumption, operator cap kept, cost on test) costs the same as
    an alternative policy. Scans a log grid for the first sign change, then bisects."""

    def gap(x: float) -> float:
        c = costing.Costs.from_dict({**COSTS.to_dict(), param: float(x)})
        band = costing.band_search(val["score"], val["positive"], c, max_reviewers=MAX_REVIEWERS)
        if band["t_low"] >= 1.0:  # the band passes everything, so it is just "do nothing"
            return 1e-9
        cost = costing.evaluate_band(test["score"], test["positive"], band["t_low"], band["t_high"], c)["cost_per_1000"]
        return cost - against(c)

    xs = np.geomspace(lo, hi, 16)
    gaps = [gap(x) for x in xs]
    for i in range(1, len(xs)):
        if np.sign(gaps[i]) != np.sign(gaps[i - 1]):
            a, b, g_a = xs[i - 1], xs[i], gaps[i - 1]
            for _ in range(10):
                mid = float(np.sqrt(a * b))
                g_mid = gap(mid)
                if np.sign(g_mid) == np.sign(g_a):
                    a, g_a = mid, g_mid
                else:
                    b = mid
            return {"value": float(np.sqrt(a * b)), "design_cheaper": "below" if gaps[0] < 0 else "above",
                    "searched": [lo, hi]}
    return {"value": None, "design_cheaper": "always" if all(g < 0 for g in gaps) else "never", "searched": [lo, hi]}


def choose_operating_point(scores: dict) -> dict:
    val, test = scores["val"], scores["test"]
    sweep_val = costing.threshold_sweep(val["score"], val["positive"], COSTS)
    sweep_test = costing.threshold_sweep(test["score"], test["positive"], COSTS)
    sweep = sweep_val.merge(sweep_test, on="threshold", suffixes=("_val", "_test"))
    sweep.to_csv(ARTIFACTS / "cost_sweep.csv", index=False)

    t_single = costing.best_single_threshold(sweep_val)
    unconstrained = costing.band_search(val["score"], val["positive"], COSTS)
    band = costing.band_search(val["score"], val["positive"], COSTS, max_reviewers=MAX_REVIEWERS)
    t_low, t_high = band["t_low"], band["t_high"]

    test_band = costing.evaluate_band(test["score"], test["positive"], t_low, t_high, COSTS)
    test_alone = costing.evaluate_band(test["score"], test["positive"], t_single, t_single, COSTS)
    alt = costing.alternatives(test["score"], test["positive"], COSTS, t_single, t_low, t_high)
    alt.to_csv(ARTIFACTS / "alternatives.csv", index=False)
    sens = sensitivity(val, test)
    sens.to_csv(ARTIFACTS / "sensitivity.csv", index=False)

    op = {
        "costs": COSTS.to_dict(),
        "max_reviewers": MAX_REVIEWERS,
        "chosen_on": "val",
        "t_single": t_single,
        "t_low": t_low,
        "t_high": t_high,
        "unconstrained_band_on_val": unconstrained,
        "val_band": band,
        "val_model_alone": costing.evaluate_band(val["score"], val["positive"], t_single, t_single, COSTS),
        "test_band": test_band,
        "test_model_alone": test_alone,
        "test_band_ci95": bootstrap_cost(test, t_low, t_high, COSTS),
        "test_model_alone_ci95": bootstrap_cost(test, t_single, t_single, COSTS),
        "test_counts_band": costing.outcome_counts(test["score"], test["positive"], t_low, t_high).to_dict(),
        "test_counts_model_alone": costing.outcome_counts(test["score"], test["positive"], t_single, t_single).to_dict(),
        "cost_do_nothing": do_nothing_cost(COSTS),
        "cost_check_everything": check_everything_cost(COSTS),
        "break_even": {
            "base_rate_vs_do_nothing": break_even(val, test, "base_rate", 1e-4, 0.3, do_nothing_cost),
            "base_rate_vs_check_everything": break_even(val, test, "base_rate", 1e-3, 0.3, check_everything_cost),
            "fn_vs_check_everything": break_even(val, test, "fn", 1.0, 500.0, check_everything_cost),
            "fn_vs_do_nothing": break_even(val, test, "fn", 0.1, 500.0, do_nothing_cost),
        },
    }
    write_json(ARTIFACTS / "operating_point.json", op)
    plot_cost_curve(sweep, op)
    print(alt.to_string(index=False))
    print(f"  single threshold {t_single:.3g}; band {t_low:.3g} to {t_high:.3g}")
    return op


def plot_cost_curve(sweep: pd.DataFrame, op: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sweep["threshold"], sweep["cost_per_1000_val"], label="validation (used to choose)")
    ax.plot(sweep["threshold"], sweep["cost_per_1000_test"], label="test (reported)", linestyle="--")
    ax.axvline(op["t_single"], color="black", linewidth=1)
    ax.annotate(f"chosen {op['t_single']:.3g}", (op["t_single"], ax.get_ylim()[1] * 0.92), xytext=(5, 0),
                textcoords="offset points")
    ax.axvspan(op["t_low"], op["t_high"], color="orange", alpha=0.15, label="review band")
    ax.axhline(do_nothing_cost(COSTS), color="grey", linestyle=":", label="do nothing")
    ax.axhline(check_everything_cost(COSTS), color="green", linestyle="--", linewidth=1,
               label="operator checks every snapshot")
    ax.axhline(op["test_band"]["cost_per_1000"], color="darkorange", linestyle="-.", linewidth=1.2,
               label=f"chosen band {op['t_low']:.3g} to {op['t_high']:.3g} (test)")
    ax.set_xlabel("threshold on bare-head score")
    ax.set_ylabel("expected cost per 1,000 snapshots (EUR)")
    ax.set_title("Model alone: cost against threshold (lines show the other policies)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "cost_curve.png", dpi=130)
    plt.close(fig)


# ================================================================ 6. save ===
def recall_breakdown(preds: list[dict], gts: list[dict], names: list[str], conf_t: float) -> dict:
    """Recall of true bare-head boxes (IoU 0.5, boxes kept at conf_t), split by head size
    and by how many labelled heads (people) the photo has. Shows where the average hides failures."""
    c = names.index(POSITIVE_CLASSES[0])
    size_bins = [(0.0, 0.0025, "small head (<0.25% of image)"), (0.0025, 0.01, "medium head (0.25-1%)"),
                 (0.01, 1.01, "large head (>1%)")]
    crowd_bins = [(1, 3, "1-3 people in photo"), (4, 8, "4-8 people in photo"), (9, 10**6, "9+ people in photo")]
    found = {label: [0, 0] for *_, label in size_bins + crowd_bins}
    for p, g in zip(preds, gts):
        gmask = g["cls"] == c
        if not gmask.any():
            continue
        keep = (p["cls"] == c) & (p["conf"] >= conf_t)
        matched = np.zeros(int(gmask.sum()), dtype=bool)
        pairs = greedy_matches(iou_matrix(g["xyxy"][gmask], p["xyxy"][keep]), 0.5)
        if len(pairs):
            matched[pairs[:, 0]] = True
        areas = g["rel_wh"][gmask].prod(axis=1)
        for lo, hi, label in size_bins:
            sel = (areas >= lo) & (areas < hi)
            found[label][0] += int(matched[sel].sum())
            found[label][1] += int(sel.sum())
        people = len(g["cls"])
        for lo, hi, label in crowd_bins:
            if lo <= people <= hi:
                found[label][0] += int(matched.sum())
                found[label][1] += len(matched)
    return {label: {"found": f, "total": t, "recall": f / t if t else None} for label, (f, t) in found.items()}


def save_failure_sheet(ev: dict, op: dict) -> None:
    """The 6 violation snapshots the model scores lowest and the 6 compliant ones it scores
    highest on test, with what the chosen band does with each. Red = labelled bare head,
    yellow = predicted bare head (conf >= 0.25)."""
    from PIL import ImageDraw

    test = ev["scores"]["test"]
    index = {p.name: i for i, p in enumerate(ev["images"]["test"])}
    head = ev["names"].index(POSITIVE_CLASSES[0])
    picks = [("violation", r) for _, r in test[test["positive"] == 1].nsmallest(6, "score").iterrows()]
    picks += [("compliant", r) for _, r in test[test["positive"] == 0].nlargest(6, "score").iterrows()]
    tile_w, tile_h, cols = 320, 240, 6
    sheet = Image.new("RGB", (cols * tile_w, 2 * (tile_h + 22)), "white")
    for k, (kind, r) in enumerate(picks):
        i = index[r["image"]]
        g, p = ev["gts"]["test"][i], ev["preds"]["test"][i]
        with Image.open(ev["images"]["test"][i]) as im:
            im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        for box in g["xyxy"][g["cls"] == head]:
            draw.rectangle(box.tolist(), outline=(220, 0, 0), width=3)
        for box in p["xyxy"][(p["cls"] == head) & (p["conf"] >= 0.25)]:
            draw.rectangle(box.tolist(), outline=(255, 200, 0), width=2)
        im.thumbnail((tile_w, tile_h))
        x, y = (k % cols) * tile_w, (k // cols) * (tile_h + 22)
        sheet.paste(im, (x, y + 22))
        decision = "PASS" if r["score"] < op["t_low"] else ("FAIL" if r["score"] >= op["t_high"] else "REVIEW")
        outcome = {("violation", "PASS"): "missed", ("violation", "REVIEW"): "operator catches it",
                   ("violation", "FAIL"): "alert", ("compliant", "PASS"): "correctly passed",
                   ("compliant", "REVIEW"): "operator clears it", ("compliant", "FAIL"): "false alert"}[(kind, decision)]
        ImageDraw.Draw(sheet).text((x + 4, y + 4), f"{kind}, score {r['score']:.3f}: {decision}, {outcome}",
                                   fill=(0, 0, 0))
    sheet.save(ARTIFACTS / "failures.jpg", quality=85)


def save_examples(ev: dict) -> None:
    """A few test snapshots for the app, chosen by rule (not by how well the model does on them)."""
    EXAMPLES.mkdir(exist_ok=True)
    for old in EXAMPLES.glob("*.jpg"):
        old.unlink()
    test = ev["scores"]["test"]
    picked = list(test[test["positive"] == 1]["image"].iloc[::max(1, test["positive"].sum() // 3)][:3])
    picked += list(test[test["positive"] == 0]["image"].iloc[::max(1, (1 - test["positive"]).sum() // 3)][:3])
    by_name = {p.name: p for p in ev["images"]["test"]}
    for i, name in enumerate(picked):
        with Image.open(by_name[name]) as im:
            im.convert("RGB").save(EXAMPLES / f"test_{i + 1}.jpg", quality=88)


def save_metrics(ev: dict, op: dict) -> dict:
    names = ev["names"]
    t = op["t_single"]
    test_gts = ev["gts"]["test"]
    results = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, "url": UNIVERSE_URL},
        "positive_class": "violation = snapshot with at least one " + " or ".join(POSITIVE_CLASSES) + " box",
        "finetuned": detection_metrics(ev["preds"]["test"], test_gts, names, op_threshold=t),
        "finetuned_val": detection_metrics(ev["preds"]["val"], ev["gts"]["val"], names, op_threshold=t),
        "zero_shot_coco": detection_metrics(ev["zero_shot"], test_gts, names, op_threshold=t),
        "predict_nothing": detection_metrics(ev["nothing"], test_gts, names, op_threshold=t),
        "ultralytics_val_on_test": ev["official_test"],
        "coco_classes_seen_on_test_conf25": ev["coco_seen_on_test"],
        "confusion_matrix": {
            "labels": names + ["background"],
            "iou_threshold": 0.5,
            "at_conf_0.25": detection_confusion(ev["preds"]["test"], test_gts, len(names), 0.25).tolist(),
            "at_operating_threshold": detection_confusion(ev["preds"]["test"], test_gts, len(names), t).tolist(),
            "operating_threshold": t,
            "layout": "rows = predicted class, columns = true class",
        },
    }
    results["head_recall_breakdown"] = {
        "at_conf_0.25": recall_breakdown(ev["preds"]["test"], test_gts, names, 0.25),
        "at_operating_threshold": recall_breakdown(ev["preds"]["test"], test_gts, names, t),
    }
    test = ev["scores"]["test"]
    for key, label in (("finetuned", "finetuned"), ("zero_shot_coco", "zero_shot"), ("predict_nothing", "nothing")):
        if label == "finetuned":
            s = test["score"].to_numpy()
        elif label == "zero_shot":  # no COCO class maps to head, so score a snapshot by its best person box
            s = ev["zero_shot_proxy_scores"]
        else:
            s = np.zeros(len(test))
        y = test["positive"].to_numpy()
        flag = s >= t if label != "nothing" else np.zeros(len(y), bool)
        results[key]["image_level_at_op"] = {
            "threshold": t,
            "tp": int((flag & (y == 1)).sum()), "fn": int((~flag & (y == 1)).sum()),
            "fp": int((flag & (y == 0)).sum()), "tn": int((~flag & (y == 0)).sum()),
            "accuracy": float((flag == (y == 1)).mean()),
            "recall_violation": float(flag[y == 1].mean()) if (y == 1).any() else 0.0,
            "precision_violation": float(y[flag].mean()) if flag.any() else 0.0,
            "cost_per_1000": costing.evaluate_band(s, y, t, t, COSTS)["cost_per_1000"],
            # the cheapest threshold for this model, picked on test itself: generous to the baselines
            "best_cost_per_1000_any_threshold": float(costing.threshold_sweep(s, y, COSTS)["cost_per_1000"].min()),
            "roc_auc": roc_auc(s, y),
            "score": {"finetuned": "highest head confidence",
                      "zero_shot": f"highest COCO {COCO_PROXY_CLASS} confidence (COCO has no head class)",
                      "nothing": "always 0"}[label],
        }
    write_json(ARTIFACTS / "metrics.json", results)
    return results


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    log(f"Device: {DEVICE}. Torch {torch.__version__}")
    data_yaml = download_dataset()
    log("Describing the dataset")
    describe_dataset(data_yaml)
    if not SKIP_TRAIN:
        log(f"Finetuning {BASE_MODEL} for up to {EPOCHS} epochs at {IMGSZ}px on {DEVICE}")
    weights = finetune(data_yaml)
    log("Evaluating the finetuned model and both baselines on the test split")
    ev = evaluate_models(data_yaml, weights)
    log("Sweeping thresholds and pricing the review band")
    op = choose_operating_point(ev["scores"])
    results = save_metrics(ev, op)
    save_examples(ev)
    save_failure_sheet(ev, op)
    f = results["finetuned"]["overall"]
    log(f"Done. Test mAP50 {f['map50']:.3f}, mAP50-95 {f['map50_95']:.3f}. "
        f"Weights in {WEIGHTS / 'best.pt'}, dashboard data in {ARTIFACTS}/")


if __name__ == "__main__":
    main()
