"""Feature-gallery storage, CI-JES anchors, and semantic distortion loss."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .models import clip_text_features


ATTRIBUTES = ("eyebrows", "eyes", "nose", "lips")


@dataclass
class GalleryBank:
    clip: Tensor
    identity: Tensor
    labels: np.ndarray | None = None
    paths: list[str] | None = None

    @classmethod
    def load(cls, path: str | Path, device: torch.device) -> "GalleryBank":
        with h5py.File(path, "r") as file:
            def read(*names):
                for name in names:
                    if name in file:
                        return file[name][:]
                raise KeyError(f"None of {names} found in {path}")

            clip = read("gallery_clip", "clip").astype(np.float32)
            identity = read("gallery_id", "gallery_identity", "identity").astype(np.float32)
            labels = file["labels"][:] if "labels" in file else None
            paths = None
            if "paths" in file:
                paths = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in file["paths"][:]]
        if clip.ndim != 2 or identity.ndim != 2 or clip.shape != identity.shape or clip.shape[1] != 512:
            raise ValueError(f"Gallery features must both have shape (N,512), got {clip.shape}, {identity.shape}")
        return cls(
            F.normalize(torch.from_numpy(clip).to(device), dim=1),
            F.normalize(torch.from_numpy(identity).to(device), dim=1),
            labels,
            paths,
        )


def save_gallery(path: str | Path, clip: np.ndarray, identity: np.ndarray, labels: np.ndarray, paths: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        file.create_dataset("gallery_clip", data=clip.astype(np.float32), compression="gzip", compression_opts=4)
        file.create_dataset("gallery_id", data=identity.astype(np.float32), compression="gzip", compression_opts=4)
        file.create_dataset("labels", data=labels.astype(np.int64), compression="gzip", compression_opts=4)
        file.create_dataset("paths", data=np.asarray(paths, dtype=h5py.string_dtype("utf-8")))


class ProjectionAdapter(nn.Module):
    """Optional directional projection used for Eq. (3).

    The paper does not publish a projection checkpoint.  The default is the
    exact shared-space identity map (both encoders are 512-D); a checkpoint can
    replace it with the advertised lightweight 512 -> 256 -> 256 heads.
    """

    def __init__(self, in_dim: int = 512, out_dim: int | None = None):
        super().__init__()
        if out_dim is None or out_dim == in_dim:
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def load_projection(path: str | Path | None, device: torch.device) -> tuple[ProjectionAdapter, ProjectionAdapter]:
    if path is None:
        return ProjectionAdapter().to(device).eval(), ProjectionAdapter().to(device).eval()
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "clip" not in checkpoint or "identity" not in checkpoint:
        raise ValueError("Projection checkpoint must contain 'clip' and 'identity' state_dicts")
    clip_state, id_state = checkpoint["clip"], checkpoint["identity"]
    first_weight = next(value for key, value in clip_state.items() if key.endswith("weight") and value.ndim == 2)
    adapter_c = ProjectionAdapter(first_weight.shape[1], first_weight.shape[0])
    adapter_i = ProjectionAdapter(first_weight.shape[1], first_weight.shape[0])
    adapter_c.load_state_dict(clip_state, strict=True)
    adapter_i.load_state_dict(id_state, strict=True)
    for adapter in (adapter_c, adapter_i):
        adapter.to(device).eval()
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
    return adapter_c, adapter_i


def _joint(clip_feature: Tensor, identity_feature: Tensor) -> Tensor:
    return F.normalize(torch.cat([F.normalize(clip_feature, dim=-1), F.normalize(identity_feature, dim=-1)], dim=-1), dim=-1)


def build_anchors(
    gallery: GalleryBank,
    clip_model: nn.Module,
    device: torch.device,
    k: int = 10,
    attributes: tuple[str, ...] = ATTRIBUTES,
) -> dict[str, dict[str, Tensor]]:
    if k < 1 or k > len(gallery.clip):
        raise ValueError(f"k must be in [1, {len(gallery.clip)}]")
    texts = clip_text_features(clip_model, list(attributes), device)
    anchors: dict[str, dict[str, Tensor]] = {}
    for index, attribute in enumerate(attributes):
        scores = gallery.clip @ texts[index]
        top_idx = scores.topk(k=k, largest=True).indices
        bottom_idx = scores.topk(k=k, largest=False).indices
        top_clip = F.normalize(gallery.clip[top_idx].mean(dim=0), dim=0)
        bottom_clip = F.normalize(gallery.clip[bottom_idx].mean(dim=0), dim=0)
        top_id = F.normalize(gallery.identity[top_idx].mean(dim=0), dim=0)
        bottom_id = F.normalize(gallery.identity[bottom_idx].mean(dim=0), dim=0)
        divergence = torch.stack([1 - F.cosine_similarity(top_clip, bottom_clip, dim=0), 1 - F.cosine_similarity(top_id, bottom_id, dim=0)]).mean()
        anchors[attribute] = {
            "top_clip": top_clip,
            "bottom_clip": bottom_clip,
            "top_identity": top_id,
            "bottom_identity": bottom_id,
            "top_anchor": _joint(top_clip, top_id),
            "bottom_anchor": _joint(bottom_clip, bottom_id),
            "divergence": divergence,
            "top_indices": top_idx,
            "bottom_indices": bottom_idx,
        }
    divergences = torch.stack([anchors[a]["divergence"] for a in attributes])
    weights = F.softmax(divergences, dim=0)
    for i, attribute in enumerate(attributes):
        anchors[attribute]["attribute_weight"] = weights[i]
    return anchors


def semantic_distortion_loss(
    clip_feature: Tensor,
    identity_feature: Tensor,
    anchors: dict[str, dict[str, Tensor]],
    clip_projection: nn.Module | None = None,
    identity_projection: nn.Module | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute Eq. (5)-(6), including current-image consensus weights."""
    clip_projection = clip_projection or nn.Identity()
    identity_projection = identity_projection or nn.Identity()
    current_clip = F.normalize(clip_feature, dim=-1)
    current_identity = F.normalize(identity_feature, dim=-1)
    current_joint = _joint(current_clip, current_identity)
    losses, consensus = [], []
    for attribute, anchor in anchors.items():
        c_top = F.normalize(clip_projection(anchor["top_clip"] - current_clip), dim=-1)
        c_bottom = F.normalize(clip_projection(anchor["bottom_clip"] - current_clip), dim=-1)
        i_top = F.normalize(identity_projection(anchor["top_identity"] - current_identity), dim=-1)
        i_bottom = F.normalize(identity_projection(anchor["bottom_identity"] - current_identity), dim=-1)
        agree_top = F.cosine_similarity(c_top, i_top, dim=-1)
        agree_bottom = F.cosine_similarity(c_bottom, i_bottom, dim=-1)
        r = F.softmax(torch.stack([agree_top, agree_bottom]), dim=0)
        sim_top = F.cosine_similarity(current_joint, anchor["top_anchor"], dim=-1)
        sim_bottom = F.cosine_similarity(current_joint, anchor["bottom_anchor"], dim=-1)
        offset = F.relu(r[1] * sim_bottom - r[0] * sim_top)
        losses.append(anchor["attribute_weight"] * offset)
        consensus.append(torch.stack([agree_top, agree_bottom]))
    loss = torch.stack(losses).sum()
    return loss, {"current_joint": current_joint, "offsets": torch.stack(losses), "consensus": torch.stack(consensus)}

