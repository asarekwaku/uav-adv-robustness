"""Config loading, CLI overrides, seeding and run metadata."""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

VISDRONE_NAMES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _set_dotted(cfg: dict, dotted: str):
    key, _, raw = dotted.partition("=")
    if not _:
        raise ValueError(f"--set expects key=value, got {dotted!r}")
    node = cfg
    *parents, leaf = key.strip().split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(raw)


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", action="append", default=[], help="extra YAML layered on configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. attack.eps=[2,4]")
    return p


def load_config(args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    for extra in args.config:
        cfg = _merge(cfg, yaml.safe_load(Path(extra).read_text()))
    for s in args.set:
        _set_dotted(cfg, s)
    # resolve relative paths against the repo root so scripts work from any cwd
    for k, v in cfg["paths"].items():
        p = Path(os.path.expandvars(str(v))).expanduser()
        cfg["paths"][k] = str(p if p.is_absolute() else ROOT / p)
    w = Path(cfg["model"]["weights"]).expanduser()
    if not w.is_absolute() and (ROOT / w).exists():
        cfg["model"]["weights"] = str(ROOT / w)
    return cfg


def resolve_device(device) -> str:
    import torch

    if str(device) != "auto":
        return str(device)
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_device(device: str):
    import torch

    d = resolve_device(device)
    return torch.device(f"cuda:{d.split(',')[0]}" if d[0].isdigit() else d)


def seed_everything(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_metadata(cfg: dict) -> dict:
    """Everything the proposal says must be recorded for reproducibility."""
    import torch
    import ultralytics

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu,
        "device": resolve_device(cfg["device"]),
        "config": cfg,
    }


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
