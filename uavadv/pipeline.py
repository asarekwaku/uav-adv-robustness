"""Shared orchestration: generate attacked sets and evaluate (condition, defense) pairs into result rows."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from . import defenses
from .attacks import condition_name, generate_condition
from .config import resolve_device, seed_everything, torch_device
from .evaluate import attack_success, detected_gt, latency, map_metrics, recall_summary
from .model import load_attack_model


class Experiment:
    def __init__(self, cfg: dict, split: str, weights: str | None = None, tag: str | None = None):
        self.cfg = cfg
        self.split = split
        self.weights = weights or cfg["model"]["weights"]
        self.tag = tag or cfg["model"]["tag"]
        self.imgsz = cfg["eval"]["imgsz"]
        self.set_dir = Path(cfg["paths"]["eval_dir"]) / f"{split}_{self.imgsz}"
        self.clean_dir = self.set_dir / "clean"
        if not (self.set_dir / "meta.json").exists():
            raise FileNotFoundError(f"{self.set_dir} missing - run scripts/prepare_data.py first")
        self.meta = json.loads((self.set_dir / "meta.json").read_text())
        self.device = resolve_device(cfg["device"])
        self.tdevice = torch_device(cfg["device"])
        self.yolo = YOLO(self.weights)
        self._attack_model = None
        self._clean_det: dict[str, dict] = {}

    @property
    def attack_model(self):
        if self._attack_model is None:
            self._attack_model, _ = load_attack_model(self.weights, self.tdevice)
        return self._attack_model

    def attacked(self, method: str, eps255: int, seed: int, defense: dict | None = None) -> Path:
        a = self.cfg["attack"]
        seed_everything(seed)
        name = condition_name(method, a["objective"], eps255, seed, defense)
        return generate_condition(
            self.attack_model, self.clean_dir, self.set_dir / self.tag / name, self.meta, imgsz=self.imgsz,
            device=self.tdevice, batch_size=a["batch"], method=method, eps255=eps255, steps=a["pgd_steps"],
            alpha255=a["pgd_alpha"], random_start=a["random_start"], objective=a["objective"], seed=seed,
            defense=defense)

    def evaluate(self, cond_dir: Path, defense: dict | None, *, with_latency=False, **labels) -> dict:
        e = self.cfg["eval"]
        row = {"model": self.tag, "split": self.split, "condition": cond_dir.name, "defense": defenses.name_of(defense),
               "n_images": self.meta["n"], **labels}
        row.update(map_metrics(self.yolo, cond_dir, defense=defense, imgsz=self.imgsz, batch=e["batch"], conf=e["conf"],
                               iou=e["iou"], device=self.device, runs_dir=self.cfg["paths"]["runs_dir"]))
        det = detected_gt(self.yolo, cond_dir, defense=defense, imgsz=self.imgsz, conf=e["match_conf"],
                          match_iou=e["match_iou"], device=self.device, batch=e["batch"])
        row.update(recall_summary(det, self.yolo.names))
        key = defenses.name_of(defense)
        if cond_dir.name == "clean":
            self._clean_det[key] = det
        elif key in self._clean_det:
            row.update(attack_success(self._clean_det[key], det))
        if with_latency:
            row.update(latency(self.yolo, cond_dir, defense=defense, imgsz=self.imgsz, device=self.device,
                               n=e["latency_images"]))
        print(f"[eval] {self.tag} {cond_dir.name:40s} defense={key:16s} mAP50={row['map50']:.4f} "
              f"mAP50-95={row['map50_95']:.4f} opR={row['op_recall']:.3f}")
        return row


def append_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
