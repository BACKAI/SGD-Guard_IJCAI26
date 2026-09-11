"""Differentiable photometric/geometric transform bank for SD-EOT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformConfig:
    brightness: float = 0.10
    color: float = 0.10
    contrast: float = 0.10
    crop_ratio: float = 0.95
    gamma: float = 0.10
    hue: float = 0.05
    rotate_degrees: float = 15.0
    saturation: float = 0.10
    scale_ratio: float = 0.90
    sharpness: float = 0.20
    translate_ratio: float = 0.05


DEFAULT_TRANSFORMS = (
    "brightness",
    "color",
    "contrast",
    "crop",
    "gamma",
    "hue",
    "rotate",
    "saturation",
    "scale",
    "sharpness",
    "translateX",
    "translateY",
)


def _rgb_to_hsv(x: Tensor) -> Tensor:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxv, argmax = x.max(dim=1)
    minv = x.min(dim=1).values
    delta = maxv - minv
    h = torch.zeros_like(maxv)
    safe = delta > 1e-8
    rc = ((g - b) / delta.clamp_min(1e-8)) % 6
    gc = (b - r) / delta.clamp_min(1e-8) + 2
    bc = (r - g) / delta.clamp_min(1e-8) + 4
    h = torch.where(safe & (argmax == 0), rc, h)
    h = torch.where(safe & (argmax == 1), gc, h)
    h = torch.where(safe & (argmax == 2), bc, h) / 6.0
    s = delta / maxv.clamp_min(1e-8)
    return torch.stack([h % 1.0, s, maxv], dim=1)


def _hsv_to_rgb(hsv: Tensor) -> Tensor:
    h, s, v = hsv[:, 0] * 6.0, hsv[:, 1], hsv[:, 2]
    i = torch.floor(h).long() % 6
    f = h - torch.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    table = torch.stack(
        [torch.stack([v, t, p], 1), torch.stack([q, v, p], 1), torch.stack([p, v, t], 1),
         torch.stack([p, q, v], 1), torch.stack([t, p, v], 1), torch.stack([v, p, q], 1)], 1
    )
    gather = i[:, None, None].expand(-1, 1, 3)
    return table.gather(1, gather).squeeze(1)


def _affine(x: Tensor, angle: float = 0.0, tx: float = 0.0, ty: float = 0.0) -> Tensor:
    radians = torch.tensor(angle * torch.pi / 180.0, device=x.device, dtype=x.dtype)
    c, s = torch.cos(radians), torch.sin(radians)
    theta = torch.zeros((x.shape[0], 2, 3), device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = c, -s
    theta[:, 1, 0], theta[:, 1, 1] = s, c
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def apply_transform(x: Tensor, name: str, config: TransformConfig | None = None) -> Tensor:
    """Apply one fixed differentiable transform while preserving NCHW shape."""
    config = config or TransformConfig()
    if name == "brightness":
        out = x + config.brightness
    elif name == "color":
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand_as(x)
        out = gray + (x - gray) * (1 + config.color)
    elif name == "contrast":
        out = x.mean(dim=(2, 3), keepdim=True) + (x - x.mean(dim=(2, 3), keepdim=True)) * (1 + config.contrast)
    elif name == "crop":
        ratio = config.crop_ratio
        h, w = x.shape[-2:]
        top, left = int((1 - ratio) * h / 2), int((1 - ratio) * w / 2)
        out = F.interpolate(x[..., top:h - top, left:w - left], size=(h, w), mode="bilinear", align_corners=False)
    elif name == "gamma":
        out = x.clamp(0, 1).pow(1 + config.gamma)
    elif name == "hue":
        hsv = _rgb_to_hsv(x.clamp(0, 1))
        hsv[:, 0] = (hsv[:, 0] + config.hue) % 1.0
        out = _hsv_to_rgb(hsv)
    elif name == "rotate":
        out = _affine(x, angle=config.rotate_degrees)
    elif name == "saturation":
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand_as(x)
        out = gray + (x - gray) * (1 + config.saturation)
    elif name == "scale":
        ratio = config.scale_ratio
        h, w = x.shape[-2:]
        hh, ww = max(1, int(h * ratio)), max(1, int(w * ratio))
        small = F.interpolate(x, size=(hh, ww), mode="bilinear", align_corners=False)
        out = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    elif name == "sharpness":
        blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        out = x + config.sharpness * (x - blur)
    elif name == "translateX":
        out = _affine(x, tx=config.translate_ratio)
    elif name == "translateY":
        out = _affine(x, ty=config.translate_ratio)
    else:
        raise KeyError(f"Unknown transform: {name}")
    return out.clamp(0, 1)


def apply_bank(x: Tensor, names: tuple[str, ...] = DEFAULT_TRANSFORMS, config: TransformConfig | None = None) -> list[Tensor]:
    return [apply_transform(x, name, config) for name in names]

