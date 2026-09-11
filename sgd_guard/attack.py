"""Online SGD-Guard objectives (SD-attack, SD-EOT, and SR-JPEG)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .augmentations import DEFAULT_TRANSFORMS, TransformConfig, apply_transform
from .gallery import GalleryBank, semantic_distortion_loss
from .jpeg import bpda_jpeg
from .models import arcface_features, clip_image_features, facenet_features


@dataclass
class DirectionBank:
    names: tuple[str, ...]
    vectors: Tensor

    @classmethod
    def load(cls, path: str | Path, device: torch.device) -> "DirectionBank":
        data = np.load(path, allow_pickle=False)
        names = tuple(str(name) for name in data["transform_names"].tolist())
        vectors = torch.from_numpy(data["mean_vectors"]).float().to(device)
        if vectors.ndim != 2 or vectors.shape[0] != len(names):
            raise ValueError("direction file has inconsistent names and mean_vectors")
        return cls(names, F.normalize(vectors, dim=1))


def save_directions(path: str | Path, names: tuple[str, ...], vectors: Tensor) -> None:
    np.savez_compressed(
        path,
        transform_names=np.asarray(names),
        mean_vectors=F.normalize(vectors.detach().float(), dim=1).cpu().numpy().astype(np.float32),
    )


def _identity_features(model: nn.Module, images: Tensor, encoder: str) -> Tensor:
    if encoder == "arcface":
        return arcface_features(model, images)
    if encoder == "facenet":
        return facenet_features(model, images)
    raise ValueError(f"unknown identity encoder: {encoder}")


def joint_features(clip_model: nn.Module, identity_model: nn.Module, images: Tensor, identity_encoder: str) -> tuple[Tensor, Tensor, Tensor]:
    clip_feature = clip_image_features(clip_model, images)
    identity_feature = _identity_features(identity_model, images, identity_encoder)
    joint = F.normalize(torch.cat([clip_feature, identity_feature], dim=1), dim=1)
    return clip_feature, identity_feature, joint


def sd_attack_loss(
    clip_model: nn.Module,
    identity_model: nn.Module,
    images: Tensor,
    gallery: GalleryBank,
    identity_encoder: str,
    anchors: dict[str, dict[str, Tensor]],
    clip_projection: nn.Module | None = None,
    identity_projection: nn.Module | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    clip_feature, identity_feature, joint = joint_features(clip_model, identity_model, images, identity_encoder)
    loss, debug = semantic_distortion_loss(
        clip_feature, identity_feature, anchors, clip_projection, identity_projection
    )

    return loss, debug


def sd_eot_loss(
    clip_model: nn.Module,
    identity_model: nn.Module,
    images: Tensor,
    gallery: GalleryBank,
    anchors: dict[str, dict[str, Tensor]],
    directions: DirectionBank,
    identity_encoder: str,
    transform_config: TransformConfig,
    clip_projection: nn.Module | None = None,
    identity_projection: nn.Module | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute Eq. (9)-(12) with direction-adaptive weights."""
    base_loss, base_debug = sd_attack_loss(
        clip_model, identity_model, images, gallery, identity_encoder, anchors, clip_projection, identity_projection
    )
    direction = -torch.autograd.grad(base_loss, base_debug["current_joint"], retain_graph=True, create_graph=False)[0]
    direction = F.normalize(direction.detach(), dim=1)
    means = directions.vectors
    scores = 1.0 - direction @ means.T
    weights = F.softmax(scores, dim=1).detach()
    transformed_losses = []
    for index, name in enumerate(directions.names):
        transformed = apply_transform(images, name, transform_config)
        transformed_loss, _ = sd_attack_loss(
            clip_model, identity_model, transformed, gallery, identity_encoder, anchors,
            clip_projection, identity_projection,
        )
        transformed_losses.append(transformed_loss)
    losses = torch.stack(transformed_losses).T
    eot_loss = (weights * losses).sum(dim=1).mean()
    return eot_loss, {"weights": weights, "scores": scores, "base_loss": base_loss.detach()}


def sr_jpeg_loss(
    clip_model: nn.Module,
    identity_model: nn.Module,
    images: Tensor,
    gallery: GalleryBank,
    anchors: dict[str, dict[str, Tensor]],
    qualities: tuple[int, ...],
    identity_encoder: str,
    clip_projection: nn.Module | None = None,
    identity_projection: nn.Module | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute Eq. (13): real codec forward, differentiable surrogate backward."""
    losses = []
    for quality in qualities:
        jpeg_image = bpda_jpeg(images, quality)
        loss, _ = sd_attack_loss(
            clip_model, identity_model, jpeg_image, gallery, identity_encoder, anchors,
            clip_projection, identity_projection,
        )
        losses.append(loss)
    result = torch.stack(losses).mean()
    return result, {"qualities": torch.tensor(qualities, device=images.device)}

