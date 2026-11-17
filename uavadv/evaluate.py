"""Evaluation: mAP (Ultralytics validator), per-GT recall / attack success, and latency.

Metrics (proposal Sec. V-A):
  * mAP@0.5, mAP@0.5:0.95, per-class P/R/AP from the standard Ultralytics validator.
  * Recall at an operating point (conf >= match_conf): a ground-truth box counts as detected if some
    prediction of the same class has IoU >= match_iou with it. Reported overall, per class and per
    scale (COCO area thresholds 32^2 / 96^2 in the 640-px evaluation frame).
  * Attack success (object-hiding rate) = |detected on clean AND missed on attacked| / |detected on clean|.
    Denominator: GT objects detected on the clean image under the same rule.
  * Latency: median and mean ms/image at batch 1 for defense + full predict (pre/infer/NMS).
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from . import defenses
from .data import read_labels

SCALES = {"small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, float("inf"))}


def make_validator(defense: dict | None):
    class DefendedValidator(DetectionValidator):
        def preprocess(self, batch):
            batch = super().preprocess(batch)
            batch["img"] = defenses.apply_torch(batch["img"].float(), defense).to(batch["img"].dtype)
            return batch

    return DefendedValidator


def map_metrics(yolo: YOLO, cond_dir: Path, *, defense, imgsz, batch, conf, iou, device, runs_dir) -> dict:
    m = yolo.val(data=str(cond_dir / "data.yaml"), split="val", imgsz=imgsz, batch=batch, conf=conf, iou=iou,
                 device=device, plots=False, verbose=False, project=str(runs_dir), name="val", exist_ok=True,
                 validator=make_validator(defense), workers=0)
    out = {"map50": float(m.box.map50), "map50_95": float(m.box.map), "precision": float(m.box.mp),
           "recall": float(m.box.mr), **{f"speed_{k}_ms": float(v) for k, v in m.speed.items()}}
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        name = m.names[int(c)]
        out.update({f"P_{name}": float(p), f"R_{name}": float(r), f"AP50_{name}": float(ap50), f"AP_{name}": float(ap)})
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def detected_gt(yolo: YOLO, cond_dir: Path, *, defense, imgsz, conf, match_iou, device, batch=16) -> dict:
    """Return {stem: (gt_cls, gt_area, detected_bool)} for every image in a condition."""
    paths = sorted((cond_dir / "images").glob("*.png"))
    out = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = [apply_bgr(cv2.imread(str(p)), defense) for p in chunk]
        for p, res in zip(chunk, yolo.predict(imgs, imgsz=imgsz, conf=conf, device=device, verbose=False)):
            gc, gb = read_labels(cond_dir / "labels" / f"{p.stem}.txt", imgsz)
            pb, pc = res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)
            hit = np.zeros(len(gc), bool)
            if len(gc) and len(pb):
                ious = _iou(gb, pb)
                ious[gc[:, None] != pc[None, :]] = 0
                hit = ious.max(1) >= match_iou
            out[p.stem] = (gc, (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]), hit)
    return out


def apply_bgr(img_bgr: np.ndarray, defense) -> np.ndarray:
    return defenses.apply_rgb_uint8(img_bgr[..., ::-1], defense)[..., ::-1].copy() if defense else img_bgr


def recall_summary(det: dict, names: dict) -> dict:
    cls = np.concatenate([v[0] for v in det.values()]) if det else np.zeros(0, int)
    area = np.concatenate([v[1] for v in det.values()]) if det else np.zeros(0)
    hit = np.concatenate([v[2] for v in det.values()]) if det else np.zeros(0, bool)
    out = {"op_recall": float(hit.mean()) if len(hit) else float("nan"), "n_gt": int(len(hit))}
    for c, name in names.items():
        sel = cls == c
        out[f"opR_{name}"] = float(hit[sel].mean()) if sel.any() else float("nan")
    for s, (lo, hi) in SCALES.items():
        sel = (area >= lo) & (area < hi)
        out[f"opR_{s}"] = float(hit[sel].mean()) if sel.any() else float("nan")
        out[f"n_{s}"] = int(sel.sum())
    return out


def attack_success(clean_det: dict, adv_det: dict) -> dict:
    was = np.concatenate([clean_det[k][2] for k in clean_det])
    now = np.concatenate([adv_det[k][2] for k in clean_det])
    denom = int(was.sum())
    return {"hidden_rate": float((was & ~now).sum() / denom) if denom else float("nan"), "hidden_denominator": denom}


def latency(yolo: YOLO, cond_dir: Path, *, defense, imgsz, device, n: int) -> dict:
    paths = sorted((cond_dir / "images").glob("*.png"))[:n]
    imgs = [cv2.imread(str(p)) for p in paths]
    sync = (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else (
        (lambda: torch.mps.synchronize()) if str(device) == "mps" else (lambda: None))
    for im in imgs[:5]:  # warm-up
        yolo.predict(apply_bgr(im, defense), imgsz=imgsz, device=device, verbose=False)
    times = []
    for im in imgs:
        sync()
        t = time.perf_counter()
        yolo.predict(apply_bgr(im, defense), imgsz=imgsz, device=device, verbose=False)
        sync()
        times.append((time.perf_counter() - t) * 1000)
    return {"latency_ms_median": float(np.median(times)), "latency_ms_mean": float(np.mean(times)),
            "latency_n": len(times)}
