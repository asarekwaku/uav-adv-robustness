"""VisDrone-DET2019 download/conversion and fixed letterboxed evaluation sets.

Conversion rules (documented for the report):
  * VisDrone category 0 ("ignored regions", score 0) and 11 ("others") are dropped.
  * Categories 1..10 map to YOLO classes 0..9 (see config.VISDRONE_NAMES).
  * Boxes are clipped to the image and converted to normalised (cx, cy, w, h).
Splits: VisDrone2019-DET-train -> train, -val -> val, -test-dev -> test.

Evaluation sets: every test/val image is letterboxed ONCE to imgsz x imgsz and saved as a
lossless PNG. Clean, attacked and defended conditions are all evaluated from these same
pixels, so preprocessing is identical across conditions (proposal Sec. IV-B).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import VISDRONE_NAMES

URLS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "test": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
}
SOURCE_DIR = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------------------------- download / convert
def download_split(root: Path, split: str):
    from ultralytics.utils.downloads import download

    root.mkdir(parents=True, exist_ok=True)
    if (root / "images" / split).exists():
        print(f"[data] {split}: already converted, skipping download")
        return
    download([URLS[split]], dir=root, threads=1)


def convert_split(root: Path, split: str) -> dict:
    src = root / SOURCE_DIR[split]
    img_out, lbl_out = root / "images" / split, root / "labels" / split
    if img_out.exists() and lbl_out.exists() and not src.exists():
        return {"split": split, "skipped": True}
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    stats = {"split": split, "images": 0, "boxes": 0, "dropped_ignored": 0, "dropped_others": 0, "dropped_degenerate": 0}
    for ann in sorted((src / "annotations").glob("*.txt")):
        img_src = src / "images" / f"{ann.stem}.jpg"
        h, w = cv2.imread(str(img_src)).shape[:2]
        lines = []
        for row in ann.read_text().strip().splitlines():
            vals = [v for v in row.strip().split(",") if v != ""]
            if len(vals) < 6:
                continue
            x, y, bw, bh, score, cat = map(int, vals[:6])
            if cat == 0 or score == 0:
                stats["dropped_ignored"] += 1
                continue
            if cat == 11:
                stats["dropped_others"] += 1
                continue
            x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
            if x1 - x0 < 1 or y1 - y0 < 1:
                stats["dropped_degenerate"] += 1
                continue
            lines.append(f"{cat - 1} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        (lbl_out / f"{ann.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        shutil.move(str(img_src), img_out / img_src.name)
        stats["images"] += 1
        stats["boxes"] += len(lines)
    shutil.rmtree(src)
    return stats


def write_dataset_yaml(root: Path) -> Path:
    path = root / "visdrone.yaml"
    path.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                    "test": "images/test", "names": VISDRONE_NAMES}, sort_keys=False))
    return path


# ----------------------------------------------------------------------------- letterboxed eval sets
def letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    return out, {"orig_hw": [h, w], "ratio": r, "top": top, "left": left, "new_hw": [nh, nw]}


def make_eval_set(data_root: Path, split: str, out_dir: Path, imgsz: int, limit: int | None = None) -> Path:
    """Create <out_dir>/clean/{images,labels} + meta.json. Returns the clean condition dir."""
    clean = out_dir / "clean"
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["imgsz"] == imgsz and meta["limit"] == limit:
            return clean
        raise RuntimeError(f"{out_dir} exists with different settings; delete it or change paths.eval_dir")
    imgs = sorted(p for p in (data_root / "images" / split).iterdir() if p.suffix.lower() in IMG_EXTS)
    if limit:
        imgs = imgs[:limit]
    (clean / "images").mkdir(parents=True, exist_ok=True)
    (clean / "labels").mkdir(parents=True, exist_ok=True)
    per_image = {}
    for p in imgs:
        lb, info = letterbox(cv2.imread(str(p)), imgsz)
        cv2.imwrite(str(clean / "images" / f"{p.stem}.png"), lb)
        h, w = info["orig_hw"]
        r, top, left = info["ratio"], info["top"], info["left"]
        rows = []
        lbl = data_root / "labels" / split / f"{p.stem}.txt"
        for line in (lbl.read_text().splitlines() if lbl.exists() else []):
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w * r + left, float(cy) * h * r + top, float(bw) * w * r, float(bh) * h * r
            rows.append(f"{c} {cx / imgsz:.6f} {cy / imgsz:.6f} {bw / imgsz:.6f} {bh / imgsz:.6f}")
        (clean / "labels" / f"{p.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        per_image[p.stem] = info
    write_condition_yaml(clean)
    meta_path.write_text(json.dumps({"split": split, "imgsz": imgsz, "limit": limit, "n": len(imgs), "images": per_image}))
    return clean


def write_condition_yaml(cond_dir: Path) -> Path:
    path = cond_dir / "data.yaml"
    path.write_text(yaml.safe_dump({"path": str(cond_dir), "train": "images", "val": "images", "names": VISDRONE_NAMES},
                                   sort_keys=False))
    return path


def new_condition(clean_dir: Path, cond_dir: Path) -> Path:
    """Make an empty condition dir sharing the clean labels."""
    (cond_dir / "images").mkdir(parents=True, exist_ok=True)
    if (cond_dir / "labels").exists():
        shutil.rmtree(cond_dir / "labels")
    shutil.copytree(clean_dir / "labels", cond_dir / "labels")
    for cache in cond_dir.glob("*.cache"):
        cache.unlink()
    write_condition_yaml(cond_dir)
    return cond_dir


def pad_mask(meta: dict, stem: str, imgsz: int) -> np.ndarray:
    """1 inside the real image, 0 in letterbox padding (attacker cannot touch padding)."""
    info = meta["images"][stem]
    m = np.zeros((imgsz, imgsz), dtype=np.float32)
    nh, nw = info["new_hw"]
    m[info["top"]:info["top"] + nh, info["left"]:info["left"] + nw] = 1
    return m


def read_labels(path: Path, imgsz: int):
    """Return (cls[N], xyxy_pixels[N,4])."""
    rows = [list(map(float, l.split())) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    if not rows:
        return np.zeros(0, dtype=int), np.zeros((0, 4), dtype=np.float32)
    a = np.array(rows, dtype=np.float32)
    cx, cy, w, h = (a[:, 1:] * imgsz).T
    return a[:, 0].astype(int), np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
