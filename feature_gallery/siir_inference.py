# siir_inference.py : semantic iterative identity refinement (infer only)

import os
import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
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

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


# Dataset 

class EmbeddingPairDataset(Dataset):
    def __init__(self, arc_emb: np.ndarray, face_emb: np.ndarray, force_l2norm: bool = True):
        assert arc_emb.ndim == 2 and arc_emb.shape[1] == 512
        assert face_emb.ndim == 2 and face_emb.shape[1] == 512
        assert arc_emb.shape[0] == face_emb.shape[0]
        self.arc = arc_emb.astype(np.float32, copy=False)
        self.face = face_emb.astype(np.float32, copy=False)
        self.force_l2norm = force_l2norm

    def __len__(self):
        return self.arc.shape[0]

    def __getitem__(self, idx):
        e1 = torch.from_numpy(self.arc[idx])
        e2 = torch.from_numpy(self.face[idx])
        if self.force_l2norm:
            e1 = l2_normalize(e1, dim=0)
            e2 = l2_normalize(e2, dim=0)
        return e1, e2


# SIIR module(semantic iterative identity refinement) 

@dataclass
class SIIRConfig:
    dim: int = 512
    post_dim: int = 1024
    rbt_paths: int = 4
    safety_cap_iters: int = 50  

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
        d_safe = d.clamp_min(eps)
        inv = 1.0 / d_safe
        C = d.shape[1]
        thr = C / inv.sum(dim=1, keepdim=True)   # (B,1)
        return thr

    def forward(self, e1: torch.Tensor, e2: torch.Tensor, return_debug: bool = False):
        B, D = e1.shape
        assert D == self.cfg.dim and e2.shape == e1.shape

        z1 = e1
        z2 = e2
        R1 = torch.zeros_like(z1)
        R2 = torch.zeros_like(z2)

        m_prev = torch.zeros((B, D), device=z1.device, dtype=torch.bool)
        steps_used = 0

        for _ in range(self.cfg.safety_cap_iters):
            steps_used += 1
            z1 = self.rbt1(z1)
            z2 = self.rbt2(z2)

            d = (z1.abs() - z2.abs()).abs()
            thr = self._harmonic_mean_threshold(d)
            m = (d > thr)

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

        dbg = {}
        if return_debug:
            dbg = {
                "R1": R1,
                "R2": R2,
                "steps_used": torch.tensor([steps_used], device=I.device, dtype=torch.long),
                "last_mask": m.to(I.dtype),
            }
        return I, dbg


# Inference

@torch.no_grad()
def infer(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Device] {device}")

    arc = np.load(args.arc_emb)
    face = np.load(args.face_emb)
    assert arc.shape == face.shape and arc.ndim == 2 and arc.shape[1] == 512

    ckpt = torch.load(args.siir_ckpt, map_location="cpu")
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise ValueError("siir_ckpt must contain 'cfg' and 'state_dict'")

    cfg = SIIRConfig(**cfg_dict)
    model = SIIRModule(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    ds = EmbeddingPairDataset(arc, face, force_l2norm=not args.no_input_l2)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    I_all = []
    R1_all = [] if args.save_r12 else None
    R2_all = [] if args.save_r12 else None

    for e1, e2 in loader:
        e1 = e1.to(device)
        e2 = e2.to(device)
        I, dbg = model(e1, e2, return_debug=args.save_r12)
        I_all.append(I.detach().cpu().numpy().astype(np.float32))
        if args.save_r12:
            R1_all.append(dbg["R1"].detach().cpu().numpy().astype(np.float32))
            R2_all.append(dbg["R2"].detach().cpu().numpy().astype(np.float32))

    I_all = np.concatenate(I_all, axis=0)
    os.makedirs(args.out_dir, exist_ok=True)

    out_I = os.path.join(args.out_dir, "I.npy")
    np.save(out_I, I_all)
    print(f"[Saved] {out_I}  shape={I_all.shape}")

    if args.save_r12:
        R1_all = np.concatenate(R1_all, axis=0)
        R2_all = np.concatenate(R2_all, axis=0)
        out_r1 = os.path.join(args.out_dir, "R1.npy")
        out_r2 = os.path.join(args.out_dir, "R2.npy")
        np.save(out_r1, R1_all)
        np.save(out_r2, R2_all)
        print(f"[Saved] {out_r1} shape={R1_all.shape}")
        print(f"[Saved] {out_r2} shape={R2_all.shape}")


# CLI (infer only)=

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--arc_emb", type=str, required=True)
    p.add_argument("--face_emb", type=str, required=True)
    p.add_argument("--siir_ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./gallery_out")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_r12", action="store_true")
    p.add_argument("--no_input_l2", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p

def main():
    args = build_parser().parse_args()
    infer(args)

if __name__ == "__main__":
    main()
