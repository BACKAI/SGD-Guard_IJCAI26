# siir_train.py : semantic iterative identity refinement (train only)

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# RBTBlock(residual bottleneck transformation)

class RBTBlock(nn.Module):
    """
    The RBT block introduced in <Unified Representation Learning for Cross Model Compatibility>
    """
    def __init__(self, in_planes, out_planes, num_paths=4):
        super().__init__()
        if not isinstance(num_paths, int) or not 0 <= num_paths <= 4:
            raise ValueError(f'num_paths: {num_paths}')
        self.num_paths = num_paths
        if self.num_paths == 0:
            print('No need to construct trans.path since num_paths is 0.')
        for i in range(self.num_paths):
            print(f'Construct trans.path{i+1}: {in_planes} -> {out_planes}')
            setattr(self, f'path{i+1}', self._make_onepath(in_planes, out_planes))

    def _make_onepath(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Linear(in_planes, 16, bias=False),
            nn.BatchNorm1d(16, eps=2e-05, momentum=0.9),
            nn.PReLU(16),
            nn.Linear(16, 16, bias=False),
            nn.BatchNorm1d(16, eps=2e-05, momentum=0.9),
            nn.PReLU(16),
            nn.Linear(16, out_planes, bias=False),
            nn.BatchNorm1d(out_planes, eps=2e-05, momentum=0.9),
            nn.PReLU(out_planes),
        )

    def forward(self, feat):
        out = feat
        for i in range(self.num_paths):
            out = out + getattr(self, f'path{i+1}')(feat)
        return out


# Utils

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


# Dataset

class EmbeddingCacheDataset(Dataset):
    def __init__(self, arc_emb: np.ndarray, face_emb: np.ndarray, labels: np.ndarray, force_l2norm: bool = True):
        assert arc_emb.ndim == 2 and arc_emb.shape[1] == 512
        assert face_emb.ndim == 2 and face_emb.shape[1] == 512
        assert arc_emb.shape[0] == face_emb.shape[0]
        assert labels.ndim == 1 and labels.shape[0] == arc_emb.shape[0]

        self.arc = arc_emb.astype(np.float32, copy=False)
        self.face = face_emb.astype(np.float32, copy=False)
        self.y = labels.astype(np.int64, copy=False)
        self.force_l2norm = force_l2norm

    def __len__(self):
        return self.arc.shape[0]

    def __getitem__(self, idx):
        e1 = torch.from_numpy(self.arc[idx])
        e2 = torch.from_numpy(self.face[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        if self.force_l2norm:
            e1 = l2_normalize(e1, dim=0)
            e2 = l2_normalize(e2, dim=0)
        return e1, e2, y


# Frozen classifier head loader

class LinearHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes, bias=True)

    def forward(self, x):
        return self.fc(x)

def load_head_checkpoint(path: str, device: torch.device) -> LinearHead:
    ckpt = torch.load(path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    # supported formats
    if "state_dict" in ckpt and ("feat_dim" in ckpt) and ("num_classes" in ckpt):
        feat_dim = int(ckpt["feat_dim"])
        num_classes = int(ckpt["num_classes"])
        head = LinearHead(feat_dim, num_classes)
        sd = ckpt["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

        # nn.Linear only saved as weight/bias
        if ("weight" in sd) and ("fc.weight" not in sd):
            new_sd = {"fc.weight": sd["weight"]}
            if "bias" in sd:
                new_sd["fc.bias"] = sd["bias"]
            sd = new_sd
        head.load_state_dict(sd, strict=True)

    elif "fc.weight" in ckpt:
        w = ckpt["fc.weight"]
        num_classes, feat_dim = w.shape
        head = LinearHead(feat_dim, num_classes)
        head.load_state_dict(ckpt, strict=True)

    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
        w = sd.get("fc.weight", None)
        if w is None:
            raise ValueError(f"dict['model'] missing fc.weight: {path}")
        num_classes, feat_dim = w.shape
        head = LinearHead(feat_dim, num_classes)
        head.load_state_dict(sd, strict=True)

    else:
        raise ValueError(f"Unsupported head checkpoint format: keys={list(ckpt.keys())[:20]}")

    head.eval().to(device)
    for p in head.parameters():
        p.requires_grad_(False)
    return head


# SIIR module (paper-aligned)

@dataclass
class SIIRConfig:
    dim: int = 512
    post_dim: int = 1024
    rbt_paths: int = 4
    safety_cap_iters: int = 50   # implementation safety only

class SIIRModule(nn.Module):
    def __init__(self, cfg: SIIRConfig):
        super().__init__()
        self.cfg = cfg

        self.rbt1 = RBTBlock(cfg.dim, cfg.dim, num_paths=cfg.rbt_paths)
        self.rbt2 = RBTBlock(cfg.dim, cfg.dim, num_paths=cfg.rbt_paths)

        self.post_rbt = RBTBlock(cfg.post_dim, cfg.post_dim, num_paths=cfg.rbt_paths)
        self.post_linear = nn.Linear(cfg.post_dim, cfg.dim, bias=True)

    @staticmethod
    def _harmonic_mean_threshold(d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        d: (B,512) >= 0
        thr_b = H(d_b) = 512 / sum_c (1/(d_bc + eps))
        returns (B,1)
        """
        d_safe = d.clamp_min(eps)
        inv = 1.0 / d_safe
        C = d.shape[1]
        thr = C / inv.sum(dim=1, keepdim=True)
        return thr

    def forward(self, e1: torch.Tensor, e2: torch.Tensor, return_debug: bool = False):
        B, D = e1.shape
        assert D == self.cfg.dim and e2.shape == e1.shape

        z1 = e1
        z2 = e2
        R1 = torch.zeros_like(z1)
        R2 = torch.zeros_like(z2)

        # m^0 = 0
        m_prev = torch.zeros((B, D), device=z1.device, dtype=torch.bool)
        steps_used = 0

        for _ in range(self.cfg.safety_cap_iters):
            steps_used += 1
            z1 = self.rbt1(z1)
            z2 = self.rbt2(z2)

            d = (z1.abs() - z2.abs()).abs()          # (B,512)
            thr = self._harmonic_mean_threshold(d)   # (B,1)
            m = (d > thr)                            # bool

            inv_m = (~m).to(z1.dtype)
            m_f = m.to(z1.dtype)

            R1 = R1 + inv_m * z1
            R2 = R2 + inv_m * z2

            z1 = m_f * z1
            z2 = m_f * z2

            if torch.equal(m, m_prev):
                break
            m_prev = m

        cat = torch.cat([R1, R2], dim=1)  # (B,1024)
        cat = self.post_rbt(cat)
        I = self.post_linear(cat)         # (B,512)

        dbg = {"z1_last": z1, "z2_last": z2}
        if return_debug:
            dbg.update({
                "R1": R1, "R2": R2,
                "steps_used": torch.tensor([steps_used], device=I.device, dtype=torch.long),
                "last_mask": m.to(I.dtype),
            })
        return I, dbg


# Losses

def batch_mean_cov(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, D = x.shape
    if B < 2:
        raise ValueError("Batch size must be >= 2 to compute covariance.")
    mu = x.mean(dim=0)
    xc = x - mu
    cov = (xc.t() @ xc) / (B - 1)
    return mu, cov

def L_statistic(z1_last: torch.Tensor, z2_last: torch.Tensor) -> torch.Tensor:
    mu1, cov1 = batch_mean_cov(z1_last)
    mu2, cov2 = batch_mean_cov(z2_last)
    return (mu1 - mu2).pow(2).sum() + (cov1 - cov2).pow(2).sum()

def L_contrast(z1_last: torch.Tensor, z2_last: torch.Tensor) -> torch.Tensor:
    a = l2_normalize(z1_last, dim=1)
    b = l2_normalize(z2_last, dim=1)
    sim = a @ b.t()  # (B,B)
    logp = F.log_softmax(sim, dim=1)
    return -torch.diag(logp).mean()

def L_classification(I: torch.Tensor, y: torch.Tensor, head1: nn.Module, head2: nn.Module) -> torch.Tensor:
    logits1 = head1(I)
    logits2 = head2(I)
    return 0.5 * F.cross_entropy(logits1, y) + 0.5 * F.cross_entropy(logits2, y)


# Eval / Train

@torch.no_grad()
def eval_epoch(model: SIIRModule, head1: nn.Module, head2: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    totals = {"L_gallery": 0.0, "L_statistic": 0.0, "L_contrast": 0.0, "L_classification": 0.0}
    n = 0

    for e1, e2, y in loader:
        e1 = e1.to(device)
        e2 = e2.to(device)
        y = y.to(device)

        I, dbg = model(e1, e2, return_debug=False)
        z1_last = dbg["z1_last"]
        z2_last = dbg["z2_last"]

        ls = L_statistic(z1_last, z2_last)
        lc = L_contrast(z1_last, z2_last)
        lcls = L_classification(I, y, head1, head2)
        L_gallery_val = ls + lc + lcls

        totals["L_gallery"] += L_gallery_val.item()
        totals["L_statistic"] += ls.item()
        totals["L_contrast"] += lc.item()
        totals["L_classification"] += lcls.item()
        n += 1

    if n == 0:
        return {k: math.nan for k in totals}
    return {k: v / n for k, v in totals.items()}

def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Device] {device}")

    arc = np.load(args.arc_emb)
    face = np.load(args.face_emb)
    y = np.load(args.labels)

    assert arc.shape == face.shape and arc.ndim == 2 and arc.shape[1] == 512
    assert y.ndim == 1 and y.shape[0] == arc.shape[0]

    head1 = load_head_checkpoint(args.arc_head, device=device)
    head2 = load_head_checkpoint(args.face_head, device=device)

    K1, K2 = head1.fc.out_features, head2.fc.out_features
    assert K1 == K2, f"Head class mismatch: arc={K1}, face={K2}"
    y_min, y_max = int(y.min()), int(y.max())
    assert y_min >= 0 and y_max < K1, f"labels range [{y_min},{y_max}] must be within [0,{K1-1}]"

    ds = EmbeddingCacheDataset(arc, face, y, force_l2norm=not args.no_input_l2)
    N = len(ds)
    idx = np.arange(N)
    np.random.shuffle(idx)

    n_val = int(N * args.val_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    tr_ds = torch.utils.data.Subset(ds, tr_idx.tolist())
    va_ds = torch.utils.data.Subset(ds, val_idx.tolist())

    # drop_last=True to avoid B==1 (covariance)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

    cfg = SIIRConfig()
    model = SIIRModule(cfg).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "siir_module_best.pt")
    last_path = os.path.join(args.out_dir, "siir_module_last.pt")

    best_val = float("inf")
    print("[Train] start")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"L_gallery": 0.0, "L_statistic": 0.0, "L_contrast": 0.0, "L_classification": 0.0}
        nb = 0

        for e1, e2, yb in tr_loader:
            e1 = e1.to(device)
            e2 = e2.to(device)
            yb = yb.to(device)

            I, dbg = model(e1, e2, return_debug=False)
            z1_last = dbg["z1_last"]
            z2_last = dbg["z2_last"]

            ls = L_statistic(z1_last, z2_last)
            lc = L_contrast(z1_last, z2_last)
            lcls = L_classification(I, yb, head1, head2)
            L_gallery_val = ls + lc + lcls

            optim.zero_grad(set_to_none=True)
            L_gallery_val.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            running["L_gallery"] += L_gallery_val.item()
            running["L_statistic"] += ls.item()
            running["L_contrast"] += lc.item()
            running["L_classification"] += lcls.item()
            nb += 1

        for k in running:
            running[k] /= max(1, nb)

        val = eval_epoch(model, head1, head2, va_loader, device)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train L_gallery={running['L_gallery']:.4f} "
            f"L_statistic={running['L_statistic']:.4f} "
            f"L_contrast={running['L_contrast']:.4f} "
            f"L_classification={running['L_classification']:.4f} | "
            f"val L_gallery={val['L_gallery']:.4f}"
        )

        torch.save(
            {"cfg": vars(cfg), "state_dict": model.state_dict(), "epoch": epoch, "val_L_gallery": val["L_gallery"]},
            last_path
        )

        if val["L_gallery"] < best_val:
            best_val = val["L_gallery"]
            torch.save(
                {"cfg": vars(cfg), "state_dict": model.state_dict(), "epoch": epoch, "val_L_gallery": val["L_gallery"]},
                best_path
            )
            print(f"  -> saved BEST: {best_path} (val_L_gallery={best_val:.4f})")

    print("[Train] done")
    print(f"[Output] best={best_path}")
    print(f"[Output] last={last_path}")


# CLI (train only)

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--arc_emb", type=str, required=True)
    p.add_argument("--face_emb", type=str, required=True)
    p.add_argument("--labels", type=str, required=True)
    p.add_argument("--arc_head", type=str, required=True)
    p.add_argument("--face_head", type=str, required=True)

    p.add_argument("--out_dir", type=str, default="./outputs_siir")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_ratio", type=float, default=0.05)

    p.add_argument("--no_input_l2", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p

def main():
    args = build_parser().parse_args()
    train(args)

if __name__ == "__main__":
    main()
