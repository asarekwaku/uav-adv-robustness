"""Load an Ultralytics YOLO detector as a differentiable torch module for white-box attacks."""
from __future__ import annotations

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg


def load_attack_model(weights: str, device: torch.device):
    yolo = YOLO(weights)
    model = yolo.model.float().to(device).eval()
    # The detection loss reads hyper-parameters (box/cls/dfl gains) as attributes; checkpoints store a dict.
    saved = model.args if isinstance(model.args, dict) else vars(model.args)
    hyp = get_cfg(overrides={k: v for k, v in saved.items() if k in {"box", "cls", "dfl"}})
    model.args = hyp
    model.criterion = None
    for p in model.parameters():
        p.requires_grad_(False)
    return model, yolo.names


def detection_loss(model, x: torch.Tensor, batch: dict) -> torch.Tensor:
    """YOLOv8/11 training loss (box + cls + DFL) w.r.t. ground truth. Gradient ascent = untargeted attack."""
    preds = model(x)
    loss, _ = model.loss({**batch, "img": x}, preds)
    return loss.sum()


def vanish_loss(model, x: torch.Tensor, batch: dict) -> torch.Tensor:
    """Object-hiding objective: negative sum of each anchor's max class confidence.

    Ascending this pushes every confidence toward zero, i.e. objects disappear. Reported separately
    from untargeted degradation, as the proposal requires.
    """
    y = model(x)[0]  # (B, 4 + nc, anchors); class scores are already sigmoid
    return -y[:, 4:, :].amax(1).sum()


OBJECTIVES = {"untargeted": detection_loss, "vanish": vanish_loss}
