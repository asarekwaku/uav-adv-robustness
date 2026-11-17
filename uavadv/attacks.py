"""FGSM / PGD L_inf attacks on a YOLO detector, materialised as lossless PNG evaluation sets.

Proposal Eq. (1):  x_adv = clip_[0,1](x + eps * sign(grad_x L(f(x), y)))
PGD (Madry et al.): random start in the eps-ball, `steps` signed-gradient steps of size alpha,
projection back onto the ball and [0,1] after each step.

Implementation notes for the report:
  * Perturbations are applied only inside the real image region; letterbox padding is masked
    (the attacker modifies the camera image, padding is added later by the pipeline).
  * Final images are rounded to the 8-bit grid and saved as PNG. With eps an integer multiple of
    1/255 rounding never leaves the eps-ball.
  * An adaptive attack against a preprocessing defense passes `defense`, so the gradient is taken
    through defense(x) (exact for Gaussian blur, BPDA for JPEG / median / bit depth).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from . import defenses
from .data import new_condition, pad_mask, read_labels
from .model import OBJECTIVES


def _load_batch(paths, label_dir: Path, meta: dict, imgsz: int, device):
    imgs, masks, cls, boxes, bidx = [], [], [], [], []
    for i, p in enumerate(paths):
        imgs.append(cv2.imread(str(p))[..., ::-1].copy())  # BGR -> RGB (Ultralytics tensors are RGB)
        masks.append(pad_mask(meta, p.stem, imgsz))
        c, xyxy = read_labels(label_dir / f"{p.stem}.txt", imgsz)
        if len(c):
            xywh = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                             xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1) / imgsz
            cls.append(c), boxes.append(xywh), bidx.append(np.full(len(c), i))
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous().float().div(255).to(device)
    m = torch.from_numpy(np.stack(masks))[:, None].to(device)
    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape)
    batch = {
        "cls": torch.from_numpy(cat(cls, (0,))).float().view(-1, 1).to(device),
        "bboxes": torch.from_numpy(cat(boxes, (0, 4))).float().to(device),
        "batch_idx": torch.from_numpy(cat(bidx, (0,))).float().to(device),
    }
    return x, m, batch


def perturb(model, x, mask, batch, *, method: str, eps: float, steps: int = 10, alpha: float | None = None,
            random_start: bool = True, objective: str = "untargeted", defense: dict | None = None,
            generator: torch.Generator | None = None) -> torch.Tensor:
    """Return x_adv with ||x_adv - x||_inf <= eps (eps in [0,1] units)."""
    loss_fn = OBJECTIVES[objective]

    def grad(z):
        z = z.detach().requires_grad_(True)
        loss = loss_fn(model, defenses.forward_for_attack(z, defense), batch)
        return torch.autograd.grad(loss, z)[0]

    if method == "fgsm":
        return (x + eps * grad(x).sign() * mask).clamp(0, 1).detach()
    if method != "pgd":
        raise ValueError(method)
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    delta = torch.zeros_like(x)
    if random_start:
        delta = (torch.rand(x.shape, generator=generator, device="cpu").to(x.device) * 2 - 1) * eps
    x_adv = (x + delta * mask).clamp(0, 1)
    for _ in range(steps):
        x_adv = x_adv + alpha * grad(x_adv).sign() * mask
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1).detach()
    return x_adv


def generate_condition(model, clean_dir: Path, out_dir: Path, meta: dict, *, imgsz: int, device, batch_size: int,
                       method: str, eps255: int, steps: int, alpha255: float | None, random_start: bool,
                       objective: str, seed: int, defense: dict | None = None) -> Path:
    """Write an attacked copy of the clean evaluation set to out_dir. Skips if already complete."""
    done = out_dir / "attack.json"
    images = sorted((clean_dir / "images").glob("*.png"))
    if done.exists() and len(list((out_dir / "images").glob("*.png"))) == len(images):
        return out_dir
    new_condition(clean_dir, out_dir)
    g = torch.Generator().manual_seed(seed)
    eps, alpha = eps255 / 255, (alpha255 / 255 if alpha255 is not None else None)
    linf = 0.0
    for i in range(0, len(images), batch_size):
        paths = images[i:i + batch_size]
        x, m, batch = _load_batch(paths, clean_dir / "labels", meta, imgsz, device)
        x_adv = perturb(model, x, m, batch, method=method, eps=eps, steps=steps, alpha=alpha,
                        random_start=random_start, objective=objective, defense=defense, generator=g)
        q = (x_adv * 255).round().clamp(0, 255).byte()
        linf = max(linf, (q.float() - (x * 255).round()).abs().max().item())
        for p, arr in zip(paths, q.permute(0, 2, 3, 1).cpu().numpy()):
            cv2.imwrite(str(out_dir / "images" / p.name), arr[..., ::-1])
        print(f"\r[attack] {out_dir.name}: {min(i + batch_size, len(images))}/{len(images)}", end="", flush=True)
    print()
    assert linf <= eps255 + 1e-6, f"perturbation {linf} exceeds budget {eps255}"
    done.write_text(json.dumps({"method": method, "eps255": eps255, "steps": steps if method == "pgd" else 1,
                                "alpha255": alpha255, "random_start": random_start, "objective": objective,
                                "seed": seed, "adaptive_defense": defense, "measured_linf255": linf,
                                "n_images": len(images)}, indent=2))
    return out_dir


def condition_name(method: str, objective: str, eps255: int, seed: int, defense: dict | None = None) -> str:
    name = f"{method}_{objective}_eps{eps255}"
    if method == "pgd":
        name += f"_s{seed}"
    if defense:
        name += f"_adaptive-{defenses.name_of(defense)}"
    return name
