#cache_facenet_embeddings.py

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1

# file list + path resolution
def read_lines(p: Path):
    """Read non-empty lines from a text file."""
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def resolve_path(root: Path, rel: str) -> Path:
    """Resolve a relative path w.r.t. root_dir (supports absolute paths as-is)."""
    rp = Path(rel)
    return rp if rp.is_absolute() else (root / rp).resolve()

# Inference: batched embedding extraction
@torch.no_grad()
def run_batch(model, device, batch, amp, l2_norm):
    """
    Run a forward pass on a stacked batch tensor and return embeddings as float32 numpy array.
    """
    x = torch.stack(batch, 0).to(device, non_blocking=True)
    if amp and device.type == "cuda":
        with torch.cuda.amp.autocast(True):
            e = model(x)
    else:
        e = model(x)
    if l2_norm:
        e = F.normalize(e, dim=1)
    return e.detach().float().cpu().numpy().astype(np.float32)


def main():
    # Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--files_txt", type=str, required=True)
    ap.add_argument("--out_npy", type=str, required=True, help="Output path for (N,512) .npy")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--no_l2_norm", action="store_true")
    args = ap.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    files_txt = Path(args.files_txt).expanduser().resolve()
    out_npy = Path(args.out_npy).expanduser().resolve()
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing pipeline 
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Model-> FaceNet (InceptionResnetV1)
    model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device).eval()

    # Load file list and allocate output array
    rels = read_lines(files_txt)
    paths = [resolve_path(root, r) for r in rels]
    N = len(paths)

    print(f"[INFO] FaceNet device={device} | N={N} | batch={args.batch_size}")
    out = np.zeros((N, 512), dtype=np.float32)

    # Main loop: load images -> preprocess -> infer -> save into array
    i = 0
    while i < N:
        j = min(N, i + args.batch_size)
        batch = []
        for t in range(i, j):
            p = paths[t]
            with Image.open(p) as im:
                img = im.convert("RGB")
            batch.append(transform(img))

        emb = run_batch(model, device, batch, args.amp, (not args.no_l2_norm))
        out[i:j] = emb

        if (i // args.batch_size) % 10 == 0:
            print(f"[PROGRESS] {j}/{N}")
        i = j


    # Save output embeddings
    np.save(out_npy, out)
    print(f"[DONE] saved: {out_npy} | shape={out.shape}")


if __name__ == "__main__":
    main()
