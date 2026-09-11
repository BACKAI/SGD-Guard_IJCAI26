#!/usr/bin/env python3
"""Evaluate original/protected pairs with the source metrics from Section 4."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size:
            image = image.resize(size, Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.float32) / 255.0


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = np.mean((x - y) ** 2)
    return float(10 * math.log10(1.0 / max(mse, 1e-12)))


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    # Dataset-level global SSIM; use a dedicated skimage implementation if
    # exact windowed SSIM is needed for a benchmark reproduction.
    c1, c2 = 0.01**2, 0.03**2
    mx, my = x.mean(), y.mean()
    vx, vy = ((x - mx) ** 2).mean(), ((y - my) ** 2).mean()
    cov = ((x - mx) * (y - my)).mean()
    return float((2 * mx * my + c1) * (2 * cov + c2) / ((mx * mx + my * my + c1) * (vx + vy + c2)))


def main() -> None:
    parser = argparse.ArgumentParser(description="SGD-Guard source-image evaluation")
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--protected-dir", required=True)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    original_root, protected_root = Path(args.original_dir), Path(args.protected_dir)
    originals = sorted(p for p in original_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    rows = []
    for original in originals:
        relative = original.relative_to(original_root)
        protected = protected_root / relative
        if not protected.exists():
            protected = protected.with_suffix(".png")
        if not protected.exists():
            print(f"[Skip] no protected pair for {relative}")
            continue
        x, y = read(original), read(protected, (read(original).shape[1], read(original).shape[0]))
        rows.append({"path": str(relative), "PSNR": psnr(x, y), "SSIM": ssim(x, y)})
    if not rows:
        raise RuntimeError("No matching original/protected pairs")
    summary = {key: float(np.mean([row[key] for row in rows])) for key in ("PSNR", "SSIM")}
    print(f"[Summary] n={len(rows)} PSNR={summary['PSNR']:.4f} SSIM={summary['SSIM']:.4f}")
    if args.json_out:
        import json
        Path(args.json_out).write_text(json.dumps({"summary": summary, "images": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

