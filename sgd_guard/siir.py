"""Semantic Iterative Identity Refinement (SIIR).

The implementation follows Section 3.6 and Algorithm 1 of the paper.  The
identity encoders are deliberately kept outside this module: SIIR consumes
their cached 512-dimensional embeddings, which makes gallery construction
and deployment reproducible and memory efficient.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


def l2_normalize(x: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


class RBTBlock(nn.Module):
    """Residual bottleneck transformation used by cross-model compatibility."""

    def __init__(self, in_dim: int, out_dim: int, num_paths: int = 4):
        super().__init__()
        if not 0 <= num_paths <= 4:
            raise ValueError("num_paths must be in [0, 4]")
        self.num_paths = num_paths
        self.paths = nn.ModuleList([self._path(in_dim, out_dim) for _ in range(num_paths)])

    @staticmethod
    def _path(in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 16, bias=False),
            nn.BatchNorm1d(16, eps=2e-5, momentum=0.9),
            nn.PReLU(16),
            nn.Linear(16, 16, bias=False),
            nn.BatchNorm1d(16, eps=2e-5, momentum=0.9),
            nn.PReLU(16),
            nn.Linear(16, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, eps=2e-5, momentum=0.9),
            nn.PReLU(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for path in self.paths:
            out = out + path(x)
        return out


@dataclass
class SIIRConfig:
    dim: int = 512
    post_dim: int = 1024
    rbt_paths: int = 4
    max_refinement_steps: int = 50


class SIIRModule(nn.Module):
    """Refine heterogeneous ArcFace/FaceNet channels into one identity code."""

    def __init__(self, config: SIIRConfig | None = None):
        super().__init__()
        self.config = config or SIIRConfig()
        c = self.config
        self.rbt1 = RBTBlock(c.dim, c.dim, c.rbt_paths)
        self.rbt2 = RBTBlock(c.dim, c.dim, c.rbt_paths)
        self.post_rbt = RBTBlock(c.post_dim, c.post_dim, c.rbt_paths)
        self.post_linear = nn.Linear(c.post_dim, c.dim)

    @staticmethod
    def harmonic_mean(d: Tensor, eps: float = 1e-12) -> Tensor:
        safe = d.clamp_min(eps)
        return d.shape[1] / (1.0 / safe).sum(dim=1, keepdim=True)

    def forward(self, e1: Tensor, e2: Tensor, return_debug: bool = False):
        if e1.ndim != 2 or e2.shape != e1.shape or e1.shape[1] != self.config.dim:
            raise ValueError(f"Expected paired tensors (B,{self.config.dim}), got {e1.shape}, {e2.shape}")

        z1, z2 = e1, e2
        r1, r2 = torch.zeros_like(e1), torch.zeros_like(e2)
        previous = torch.zeros_like(e1, dtype=torch.bool)
        masks = []

        for step in range(self.config.max_refinement_steps):
            z1, z2 = self.rbt1(z1), self.rbt2(z2)
            disagreement = (z1.abs() - z2.abs()).abs()
            mask = disagreement > self.harmonic_mean(disagreement)
            homogeneous = (~mask).to(z1.dtype)
            heterogeneous = mask.to(z1.dtype)
            r1 = r1 + homogeneous * z1
            r2 = r2 + homogeneous * z2
            z1, z2 = heterogeneous * z1, heterogeneous * z2
            masks.append(mask)
            if torch.equal(mask, previous):
                break
            previous = mask

        identity = self.post_linear(self.post_rbt(torch.cat([r1, r2], dim=1)))
        debug = {
            "z1_last": z1,
            "z2_last": z2,
            "R1": r1,
            "R2": r2,
            "steps_used": torch.full((e1.shape[0],), len(masks), device=e1.device, dtype=torch.long),
            "last_mask": masks[-1].to(identity.dtype),
        }
        return (identity, debug) if return_debug else (identity, debug)


class EmbeddingPairDataset(Dataset):
    def __init__(self, arcface: np.ndarray, facenet: np.ndarray, labels: np.ndarray | None = None):
        if arcface.ndim != 2 or facenet.ndim != 2 or arcface.shape != facenet.shape or arcface.shape[1] != 512:
            raise ValueError("ArcFace and FaceNet arrays must both have shape (N,512)")
        if labels is None:
            labels = np.arange(len(arcface), dtype=np.int64)
        if labels.ndim != 1 or len(labels) != len(arcface):
            raise ValueError("labels must have shape (N,)")
        self.arcface = arcface.astype(np.float32, copy=False)
        self.facenet = facenet.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            l2_normalize(torch.from_numpy(self.arcface[index]), dim=0),
            l2_normalize(torch.from_numpy(self.facenet[index]), dim=0),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class LinearHead(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def load_head(path: str | Path, device: torch.device) -> LinearHead:
    checkpoint = torch.load(str(path), map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        head = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = {k.replace("module.", "", 1): v for k, v in checkpoint["state_dict"].items()}
        dim = int(checkpoint.get("feat_dim", state.get("fc.weight", state.get("weight")).shape[1]))
        classes = int(checkpoint.get("num_classes", state.get("fc.weight", state.get("weight")).shape[0]))
        head = LinearHead(dim, classes)
        if "weight" in state and "fc.weight" not in state:
            state = {"fc.weight": state["weight"], "fc.bias": state.get("bias")}
            state = {k: v for k, v in state.items() if v is not None}
        head.load_state_dict(state, strict=True)
    elif isinstance(checkpoint, dict) and "fc.weight" in checkpoint:
        w = checkpoint["fc.weight"]
        head = LinearHead(w.shape[1], w.shape[0])
        head.load_state_dict(checkpoint, strict=True)
    else:
        raise ValueError(f"Unsupported classifier-head checkpoint: {path}")
    head.to(device).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def batch_mean_cov(x: Tensor) -> Tuple[Tensor, Tensor]:
    if x.shape[0] < 2:
        raise ValueError("SIIR covariance requires batch_size >= 2")
    mean = x.mean(dim=0)
    centered = x - mean
    return mean, centered.T @ centered / (x.shape[0] - 1)


def statistic_loss(z1: Tensor, z2: Tensor) -> Tensor:
    mu1, cov1 = batch_mean_cov(z1)
    mu2, cov2 = batch_mean_cov(z2)
    return (mu1 - mu2).pow(2).sum() + (cov1 - cov2).pow(2).sum()


def contrastive_loss(z1: Tensor, z2: Tensor) -> Tensor:
    logits = l2_normalize(z1, 1) @ l2_normalize(z2, 1).T
    return -F.log_softmax(logits, dim=1).diag().mean()


def classification_loss(identity: Tensor, labels: Tensor, head1: nn.Module, head2: nn.Module) -> Tensor:
    return 0.5 * F.cross_entropy(head1(identity), labels) + 0.5 * F.cross_entropy(head2(identity), labels)


def train_siir(
    arcface: np.ndarray,
    facenet: np.ndarray,
    labels: np.ndarray,
    arcface_head: nn.Module,
    facenet_head: nn.Module,
    out_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.05,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
    seed: int = 42,
) -> Path:
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    dataset = EmbeddingPairDataset(arcface, facenet, labels)
    indices = rng.permutation(len(dataset))
    n_val = int(len(dataset) * val_ratio)
    val_indices, train_indices = indices[:n_val], indices[n_val:]
    train_loader = DataLoader(Subset(dataset, train_indices.tolist()), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(dataset, val_indices.tolist()), batch_size=batch_size, shuffle=False, drop_last=True)

    config = SIIRConfig()
    model = SIIRModule(config).to(device)
    arcface_head, facenet_head = arcface_head.to(device).eval(), facenet_head.to(device).eval()
    for p in list(arcface_head.parameters()) + list(facenet_head.parameters()):
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path, last_path = output / "siir_module_best.pt", output / "siir_module_last.pt"
    best = math.inf

    def run(loader: DataLoader, train: bool) -> dict[str, float]:
        model.train(train)
        totals = {"L_gallery": 0.0, "L_statistic": 0.0, "L_contrast": 0.0, "L_classification": 0.0}
        count = 0
        for e1, e2, y in loader:
            e1, e2, y = e1.to(device), e2.to(device), y.to(device)
            identity, debug = model(e1, e2, return_debug=True)
            ls = statistic_loss(debug["z1_last"], debug["z2_last"])
            lc = contrastive_loss(debug["z1_last"], debug["z2_last"])
            lcls = classification_loss(identity, y, arcface_head, facenet_head)
            loss = ls + lc + lcls
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            for key, value in [("L_gallery", loss), ("L_statistic", ls), ("L_contrast", lc), ("L_classification", lcls)]:
                totals[key] += float(value.detach())
            count += 1
        return {key: value / max(1, count) for key, value in totals.items()}

    for epoch in range(1, epochs + 1):
        train_metrics = run(train_loader, True)
        val_metrics = run(val_loader, False) if n_val >= 2 else train_metrics
        payload = {
            "cfg": asdict(config),
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "val_L_gallery": val_metrics["L_gallery"],
        }
        torch.save(payload, last_path)
        if val_metrics["L_gallery"] < best:
            best = val_metrics["L_gallery"]
            torch.save(payload, best_path)
        print(
            f"[SIIR] epoch={epoch:03d} train={train_metrics['L_gallery']:.5f} "
            f"val={val_metrics['L_gallery']:.5f}"
        )
    return best_path


@torch.no_grad()
def encode_siir(model: SIIRModule, arcface: np.ndarray, facenet: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    dataset = EmbeddingPairDataset(arcface, facenet)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval().to(device)
    outputs = []
    for e1, e2, _ in loader:
        identity, _ = model(e1.to(device), e2.to(device))
        outputs.append(l2_normalize(identity, dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def load_siir(path: str | Path, device: torch.device) -> SIIRModule:
    checkpoint = torch.load(str(path), map_location="cpu")
    config = SIIRConfig(**checkpoint.get("cfg", {}))
    model = SIIRModule(config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model

