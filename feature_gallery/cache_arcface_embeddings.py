#cache_arcface_embeddings.py

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, UnidentifiedImageError


# Normalization constants matching the original pipeline: (x - 0.5) / 0.5
PIX_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
PIX_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


def read_lines(txt_path: Path):
    """Read non-empty lines from a text file (one path per line)."""
    with txt_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def resolve_path(root: Path, p: str) -> Path:
    """Resolve a path string relative to root_dir unless it is already absolute."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (root / pp).resolve()


def load_arcface_model_like_your_code(ckpt_path: Path, device: torch.device, models_root: str = "") -> nn.Module:
    """
    Load an ArcFace model object from a checkpoint in a way compatible with legacy checkpoints.

    - If the checkpoint depends on custom modules (e.g., "models"), pass the repo root via --models_root.
    - Supports checkpoints that store an nn.Module directly or store it under the key "model".
    """
    if models_root:
        mr = str(Path(models_root).expanduser().resolve())
        if mr not in sys.path:
            sys.path.insert(0, mr)

    try:
        arcface_obj = torch.load(str(ckpt_path), map_location=device)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"[CKPT LOAD ERROR] {e}\n"
            f"This checkpoint depends on custom modules that are not importable.\n"
            f"Provide --models_root pointing to the repository root containing the required modules.\n"
        )

    if isinstance(arcface_obj, nn.Module):
        model = arcface_obj
    else:
        maybe_model = arcface_obj.get("model", None) if isinstance(arcface_obj, dict) else None
        if isinstance(maybe_model, nn.Module):
            model = maybe_model
        else:
            model = arcface_obj

    if not isinstance(model, nn.Module):
        raise RuntimeError(
            "[CKPT LOAD ERROR] Failed to find an nn.Module in the checkpoint.\n"
            "This checkpoint may contain only a state_dict or be in an unsupported format.\n"
        )

    # Ensure inference mode and disable gradients
    setattr(model, "fp16", False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_feature_img(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """
    Extract ArcFace features.

    Args:
        image: Tensor of shape (B, 3, 256, 256) in [0, 1].

    Returns:
        Feature tensor of shape (B, 512) (expected).
    """
    img = F.interpolate(image, size=(112, 112), mode="bilinear", align_corners=False)
    mean = PIX_MEAN.to(img.device)
    std  = PIX_STD.to(img.device)
    img = (img - mean) / std
    feat = model(img)
    if isinstance(feat, (tuple, list)):
        feat = feat[0]
    return feat


@torch.no_grad()
def run_batch(model: nn.Module, device: torch.device, batch_tensors, l2_norm: bool) -> np.ndarray:
    """Run a batch forward pass and return features as a float32 NumPy array."""
    x = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)   # (B,3,256,256)
    feat = extract_feature_img(model, x)                                  # (B,512) expected
    if l2_norm:
        feat = F.normalize(feat, dim=1)
    return feat.detach().float().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True, help="Root directory used to resolve relative paths in files_txt.")
    ap.add_argument("--files_txt", type=str, required=True, help="Text file listing images (one per line) in a fixed order.")
    ap.add_argument("--out_npy", type=str, required=True, help="Output path for the stacked embedding array (N,512).")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to an ArcFace checkpoint storing an nn.Module (or a dict with key 'model').")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", type=str, default="", help="cuda / cpu / ''(auto)")
    ap.add_argument("--models_root", type=str, default="",
                    help="If checkpoint loading requires custom modules, provide the repository root containing them.")
    ap.add_argument("--l2_norm", action="store_true", help="Apply L2 normalization to embeddings.")
    ap.add_argument("--strict_images", action="store_true",
                    help="Stop immediately if any image fails to load (recommended to prevent index/label misalignment).")
    args = ap.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    files_txt = Path(args.files_txt).expanduser().resolve()
    out_npy = Path(args.out_npy).expanduser().resolve()
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing consistent with the original pipeline
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),  # [0,1]
    ])

    model = load_arcface_model_like_your_code(Path(args.ckpt).expanduser().resolve(), device, args.models_root)

    rel_paths = read_lines(files_txt)
    abs_paths = [resolve_path(root_dir, p) for p in rel_paths]
    N = len(abs_paths)

    print(f"[INFO] device={device} | N={N} | batch={args.batch_size}")
    print(f"[INFO] out={out_npy}")
    print(f"[INFO] L2_norm={'ON' if args.l2_norm else 'OFF'} | strict_images={'ON' if args.strict_images else 'OFF'}")

    out = np.zeros((N, 512), dtype=np.float32)

    i = 0
    while i < N:
        j = min(N, i + args.batch_size)
        batch = []
        for t in range(i, j):
            p = abs_paths[t]
            try:
                img = Image.open(p).convert("RGB")
                batch.append(data_transforms(img))
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                msg = f"[ERROR] failed to load index={t}: {p} -> {e}"
                # Default behavior is to stop to avoid any ordering/label mismatch.
                if args.strict_images:
                    raise RuntimeError(msg)
                else:
                    raise RuntimeError(msg)

        emb = run_batch(model, device, batch, l2_norm=args.l2_norm)  # (B,512)
        if emb.shape != (j - i, 512):
            raise RuntimeError(f"[ERROR] unexpected embedding shape: {emb.shape} (expected {(j - i, 512)})")

        out[i:j] = emb

        if (i // args.batch_size) % 10 == 0:
            print(f"[PROGRESS] {j}/{N}")

        i = j

    np.save(out_npy, out)
    print("[DONE] saved:", out_npy)
    print("[DONE] shape:", out.shape, out.dtype)


if __name__ == "__main__":
    main()
