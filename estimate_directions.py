#!/usr/bin/env python3
"""Estimate offline semantic-direction EOT vectors (Eq. 7-8)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sgd_guard.attack import joint_features, save_directions
from sgd_guard.augmentations import DEFAULT_TRANSFORMS, TransformConfig, apply_transform
from sgd_guard.data import load_image, scan_images
from sgd_guard.models import load_arcface, load_clip, load_facenet


def main() -> None:
    parser = argparse.ArgumentParser(description="SGD-Guard offline SD-EOT direction estimation")
    parser.add_argument("--data-root", required=True, help="FFHQ root used for the expectation in Eq. (8)")
    parser.add_argument("--arcface-ckpt", required=True)
    parser.add_argument("--farl-ckpt", required=True)
    parser.add_argument("--out", default="./pretrained/transform_directions.npz")
    parser.add_argument("--models-root", default="")
    parser.add_argument("--identity-encoder", choices=["arcface", "facenet"], default="arcface")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=None, help="Use all FFHQ images by default")
    parser.add_argument("--device", default="")
    parser.add_argument("--allow-base-clip", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = scan_images(args.data_root, one_per_identity=False, max_images=args.max_images)
    clip_model = load_clip(args.farl_ckpt, device, args.allow_base_clip)
    if args.identity_encoder == "arcface":
        identity_model = load_arcface(args.arcface_ckpt, device, args.models_root or None)
    else:
        identity_model = load_facenet(device)
    config = TransformConfig()
    vectors = []
    names = tuple(DEFAULT_TRANSFORMS)

    for name in names:
        total = None
        count = 0
        for start in range(0, len(records), args.batch_size):
            images = torch.stack([load_image(record.path, 256) for record in records[start:start + args.batch_size]]).to(device)
            with torch.no_grad():
                _, _, base = joint_features(clip_model, identity_model, images, args.identity_encoder)
                transformed = apply_transform(images, name, config)
                _, _, shifted = joint_features(clip_model, identity_model, transformed, args.identity_encoder)
                difference = shifted - base
            batch_sum = difference.sum(dim=0)
            total = batch_sum if total is None else total + batch_sum
            count += len(images)
        vectors.append(total / max(1, count))
        print(f"[Direction] {name}: {count} images")

    save_directions(Path(args.out).expanduser().resolve(), names, torch.stack(vectors))
    print(f"[Done] saved={args.out} shape={(len(names), 1024)}")


if __name__ == "__main__":
    main()

