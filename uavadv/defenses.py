"""Input-preprocessing defenses (proposal Sec. IV-C).

Each defense is a dict like {"name": "jpeg", "quality": 75}. Three views of the same op:
  * apply_rgb_uint8(img, d)  - reference implementation on an HxWx3 RGB uint8 array
  * apply_torch(x, d)        - batch version on NCHW float [0,1] (used inside the validator)
  * forward_for_attack(x, d) - what an adaptive attacker differentiates through:
        gaussian -> the exact differentiable blur
        others   -> BPDA (Athalye et al. 2018): exact forward, identity gradient
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def name_of(d: dict | None) -> str:
    if not d:
        return "none"
    params = "-".join(f"{k}{v}" for k, v in d.items() if k != "name")
    return f"{d['name']}-{params}" if params else d["name"]


def apply_rgb_uint8(img: np.ndarray, d: dict | None) -> np.ndarray:
    if not d:
        return img
    n = d["name"]
    if n == "jpeg":
        ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(d["quality"])])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy()
    if n == "median":
        return cv2.medianBlur(np.ascontiguousarray(img), int(d["k"]))
    if n == "gaussian":
        return cv2.GaussianBlur(img, (0, 0), sigmaX=float(d["sigma"]), borderType=cv2.BORDER_REFLECT)
    if n == "bitdepth":
        shift = 8 - int(d["bits"])
        q = (img >> shift) << shift
        return (q + (1 << shift) // 2 if shift else q).clip(0, 255).astype(np.uint8)
    raise ValueError(f"unknown defense {n}")


def _gaussian_kernel(sigma: float, device, dtype):
    radius = max(1, math.ceil(3 * sigma))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(xs**2) / (2 * sigma**2))
    return k / k.sum(), radius


def gaussian_torch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k, r = _gaussian_kernel(sigma, x.device, x.dtype)
    c = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def apply_torch(x: torch.Tensor, d: dict | None) -> torch.Tensor:
    """x: NCHW RGB float in [0,1]. Non-differentiable ops round-trip through uint8 on CPU."""
    if not d:
        return x
    if d["name"] == "gaussian":
        return gaussian_torch(x, float(d["sigma"])).clamp(0, 1)
    arr = (x.detach().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    out = np.stack([apply_rgb_uint8(a, d) for a in arr])
    return torch.from_numpy(out).to(x.device).permute(0, 3, 1, 2).contiguous().to(x.dtype) / 255


def forward_for_attack(x: torch.Tensor, d: dict | None) -> torch.Tensor:
    if not d:
        return x
    if d["name"] == "gaussian":
        return gaussian_torch(x, float(d["sigma"]))
    return x + (apply_torch(x, d) - x).detach()  # BPDA with identity backward
