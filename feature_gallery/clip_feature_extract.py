# clip_feature_extract.py

import os
import glob
import argparse

import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
import clip  # pip install git+https://github.com/openai/CLIP.git


# Model loading: FaRL weights on CLIP ViT-B/16
def load_farl_clip(farl_path: str, device: str = "cuda"):
    """
    Load FaRL weights(FaRL-Base-Patch16-LAIONFace20M-ep64) into a CLIP ViT-B/16 backbone for image feature extraction.
    """
    model, preprocess = clip.load("ViT-B/16", device="cpu")
    state = torch.load(farl_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model, preprocess

# Input discovery: collect image paths
def sort_key(path: str):
    base = os.path.splitext(os.path.basename(path))[0]
    return (0, int(base)) if base.isdigit() else (1, base)


def collect_images(input_dir: str, recursive: bool = False):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths = []
    if recursive:
        for ext in exts:
            paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    else:
        for ext in exts:
            paths.extend(glob.glob(os.path.join(input_dir, ext)))
    return sorted(paths, key=sort_key)


# Main: extract and save CLIP features
def main():
    ap = argparse.ArgumentParser("FaRL-CLIP feature extractor (stacked cache)")
    ap.add_argument("--farl_path", type=str, required=True)
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--recursive", action="store_true")

    # Output: stacked features + corresponding filenames
    ap.add_argument("--out_clip_npy", type=str, default="clip_emb.npy")
    ap.add_argument("--out_names_npy", type=str, default="clip_names.npy")

    # Optional: also save per-image feature as <basename>.npy
    ap.add_argument("--save_per_file", action="store_true")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    print("[Load] FaRL-CLIP...")
    model, preprocess = load_farl_clip(args.farl_path, device=device)

    image_paths = collect_images(args.input_dir, recursive=args.recursive)
    print(f"[Found] {len(image_paths)} images in input_dir")

    feats = []
    names = []
    skipped = []

    # Feature extraction loop
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            base = os.path.splitext(os.path.basename(img_path))[0]

            try:
                img = Image.open(img_path).convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                print(f"[SKIP] {img_path} ({e})")
                skipped.append(img_path)
                continue

            x = preprocess(img).unsqueeze(0).to(device)  # (1,3,224,224)
            f = model.encode_image(x)                    # (1,512)
            f = f / f.norm(dim=-1, keepdim=True)         # L2 normalization
            f = f.squeeze(0).detach().cpu().numpy().astype(np.float32)

            feats.append(f)
            names.append(f"{base}.npy")

            if args.save_per_file:
                np.save(os.path.join(args.out_dir, f"{base}.npy"), f)

            if (i + 1) % 200 == 0 or (i + 1) == len(image_paths):
                print(f"[{i+1}/{len(image_paths)}] processed")

    # Save stacked cache: (N,512) float32
    feats = np.stack(feats, axis=0).astype(np.float32)

    maxlen = max(len(n) for n in names) if names else 1
    names_arr = np.array(names, dtype=f"S{maxlen}")

    out_clip = os.path.join(args.out_dir, args.out_clip_npy)
    out_names = os.path.join(args.out_dir, args.out_names_npy)
    np.save(out_clip, feats)
    np.save(out_names, names_arr)

    print("\n[DONE]")
    print(f"  clip_emb : {out_clip}  shape={feats.shape} dtype={feats.dtype}")
    print(f"  names    : {out_names} shape={names_arr.shape} dtype={names_arr.dtype}")
    print(f"  skipped  : {len(skipped)}")
    if skipped:
        print("  skipped examples:")
        for p in skipped[:10]:
            print("   -", p)


if __name__ == "__main__":
    main()
