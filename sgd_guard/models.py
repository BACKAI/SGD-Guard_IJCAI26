"""Frozen feature encoder and LoRA purifier loaders.

All loaders accept the checkpoint layouts used by the original research
repository, while avoiding imports from that repository at runtime.  A
checkpoint containing a serialized custom ``nn.Module`` can still be loaded
with ``--models-root`` pointing to the old repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _freeze(model: nn.Module, device: torch.device) -> nn.Module:
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_arcface(path: str | Path, device: torch.device, models_root: str | Path | None = None) -> nn.Module:
    if models_root:
        root = str(Path(models_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    checkpoint = torch.load(str(path), map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), nn.Module):
        model = checkpoint["model"]
    else:
        raise ValueError(
            "ArcFace checkpoint must contain a serialized nn.Module (directly or under 'model'). "
            "State-dict-only files need a model-specific constructor."
        )
    if hasattr(model, "fp16"):
        model.fp16 = False
    return _freeze(model, device)


def load_facenet(device: torch.device, weights: str = "vggface2") -> nn.Module:
    from facenet_pytorch import InceptionResnetV1

    return _freeze(InceptionResnetV1(pretrained=weights, classify=False), device)


def load_clip(farl_path: str | Path | None, device: torch.device, allow_base_clip: bool = False):
    """Load FaRL on the OpenAI CLIP ViT-B/16 implementation.

    FaRL is the paper's CLIP image encoder.  A base CLIP fallback is opt-in so
    an accidental missing FaRL checkpoint cannot silently change experiments.
    """
    import clip

    if farl_path is None and not allow_base_clip:
        raise FileNotFoundError("FaRL checkpoint is required; pass --allow-base-clip only for debugging")
    model, _ = clip.load("ViT-B/16", device=device, jit=False)
    if farl_path is not None:
        checkpoint = torch.load(str(farl_path), map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(state, dict):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[FaRL] warning: {len(missing)} missing keys")
        if unexpected:
            print(f"[FaRL] warning: {len(unexpected)} unexpected keys")
    return _freeze(model, device)


def arcface_features(model: nn.Module, images: Tensor) -> Tensor:
    x = F.interpolate(images, (112, 112), mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5
    result = model(x.float())
    if isinstance(result, (tuple, list)):
        result = result[0]
    return F.normalize(result.float(), dim=1)


def facenet_features(model: nn.Module, images: Tensor) -> Tensor:
    x = F.interpolate(images, (160, 160), mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5
    result = model(x)
    if isinstance(result, (tuple, list)):
        result = result[0]
    return F.normalize(result.float(), dim=1)


def clip_image_features(model: nn.Module, images: Tensor) -> Tensor:
    x = F.interpolate(images, (224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    encoded = model.encode_image((x - mean) / std)
    return F.normalize(encoded.float(), dim=1)


def clip_text_features(model: nn.Module, texts: list[str], device: torch.device) -> Tensor:
    import clip

    tokens = clip.tokenize(texts).to(device)
    return F.normalize(model.encode_text(tokens).float(), dim=1)

