"""Real JPEG forward pass with a differentiable BPDA surrogate."""

from __future__ import annotations

import io

import torch
from torch import Tensor
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor


@torch.no_grad()
def real_jpeg(x: Tensor, quality: int) -> Tensor:
    """Apply the actual PIL JPEG codec to an NCHW tensor in [0, 1]."""
    outputs = []
    for image in x.detach().float().cpu().clamp(0, 1):
        buffer = io.BytesIO()
        to_pil_image(image).save(buffer, format="JPEG", quality=int(quality), optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            outputs.append(pil_to_tensor(decoded.convert("RGB")).float() / 255.0)
    return torch.stack(outputs).to(x.device)


def differentiable_jpeg(x: Tensor, quality: int) -> Tensor:
    """A smooth JPEG-like surrogate used only for the backward pass.

    Quantization is not differentiable.  The quality-dependent low-pass
    surrogate approximates the dominant effect of JPEG on adversarial
    high-frequency energy while retaining a stable gradient.
    """
    strength = max(0.0, min(1.0, (100.0 - float(quality)) / 100.0))
    blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    if quality <= 50:
        h, w = x.shape[-2:]
        small = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        blurred = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    return ((1.0 - 0.35 * strength) * x + 0.35 * strength * blurred).clamp(0, 1)


def bpda_jpeg(x: Tensor, quality: int) -> Tensor:
    """Forward with real JPEG values and surrogate gradients (Eq. 13)."""
    forward = real_jpeg(x, quality)
    backward = differentiable_jpeg(x, quality)
    return forward + (backward - backward.detach())

