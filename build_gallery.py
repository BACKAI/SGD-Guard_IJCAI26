#!/usr/bin/env python3
"""Build the offline SIIR + FaRL feature gallery (Algorithm 1)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sgd_guard.data import load_image, scan_images
from sgd_guard.gallery import save_gallery
from sgd_guard.models import arcface_features, clip_image_features, load_arcface, load_clip, load_facenet, facenet_features
from sgd_guard.siir import LinearHead, encode_siir, train_siir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def encode_records(records, model, encoder: str, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(records), batch_size):
        batch = torch.stack([load_image(record.path, 256) for record in records[start:start + batch_size]]).to(device)
        if encoder == "arcface":
            output = arcface_features(model, batch)
        elif encoder == "facenet":
            output = facenet_features(model, batch)
        elif encoder == "clip":
            output = clip_image_features(model, batch)
        else:
            raise ValueError(encoder)
        outputs.append(output.cpu().numpy().astype(np.float32))
        if start == 0 or start + batch_size >= len(records):
            print(f"[Encode:{encoder}] {min(start + batch_size, len(records))}/{len(records)}")
    return np.concatenate(outputs, axis=0)


def train_identity_head(features: np.ndarray, labels: np.ndarray, path: Path, device: torch.device, epochs: int, batch_size: int, lr: float) -> None:
    model = LinearHead(features.shape[1], int(labels.max()) + 1).to(device)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print(f"[Head] {path.stem} epoch={epoch:03d} loss={total / max(1, len(loader)):.5f}")
    torch.save({"feat_dim": features.shape[1], "num_classes": int(labels.max()) + 1, "state_dict": model.state_dict()}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="SGD-Guard offline generalized feature gallery")
    parser.add_argument("--gallery-root", required=True, help="FaceFolder root; one direct subdirectory is one identity")
    parser.add_argument("--arcface-ckpt", required=True, help="Serialized ArcFace nn.Module checkpoint")
    parser.add_argument("--farl-ckpt", required=True, help="FaRL ViT-B/16 checkpoint")
    parser.add_argument("--out-dir", default="./pretrained/gallery")
    parser.add_argument("--models-root", default="", help="Old repository root needed by custom ArcFace checkpoints")
    parser.add_argument("--max-identities", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--siir-epochs", type=int, default=10)
    parser.add_argument("--siir-batch-size", type=int, default=64)
    parser.add_argument("--siir-lr", type=float, default=1e-3)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-base-clip", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = scan_images(args.gallery_root, one_per_identity=True, max_images=args.max_identities)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"[Gallery] device={device} identities={len(records)} root={args.gallery_root}")

    arc_model = load_arcface(args.arcface_ckpt, device, args.models_root or None)
    face_model = load_facenet(device)
    clip_model = load_clip(args.farl_ckpt, device, args.allow_base_clip)
    arc = encode_records(records, arc_model, "arcface", device, args.batch_size)
    face = encode_records(records, face_model, "facenet", device, args.batch_size)
    clip = encode_records(records, clip_model, "clip", device, args.batch_size)
    np.save(output / "arcface.npy", arc)
    np.save(output / "facenet.npy", face)
    np.save(output / "labels.npy", labels)

    arc_head_path, face_head_path = output / "arcface_head.pt", output / "facenet_head.pt"
    train_identity_head(arc, labels, arc_head_path, device, args.head_epochs, args.batch_size, 0.1)
    train_identity_head(face, labels, face_head_path, device, args.head_epochs, args.batch_size, 0.1)
    arc_head = torch.load(arc_head_path, map_location="cpu")
    face_head = torch.load(face_head_path, map_location="cpu")
    arc_head_model, face_head_model = LinearHead(512, len(records)), LinearHead(512, len(records))
    arc_head_model.load_state_dict(arc_head["state_dict"])
    face_head_model.load_state_dict(face_head["state_dict"])
    best_siir = train_siir(
        arc, face, labels, arc_head_model, face_head_model, output,
        epochs=args.siir_epochs, batch_size=args.siir_batch_size, lr=args.siir_lr,
        device=device, seed=args.seed,
    )
    from sgd_guard.siir import load_siir
    siir = load_siir(best_siir, device)
    identity = encode_siir(siir, arc, face, args.batch_size, device)
    gallery_path = output / "gallery.h5"
    save_gallery(gallery_path, clip, identity, labels, [record.relative_path for record in records])
    print(f"[Done] gallery={gallery_path} shape={(len(records), 512)}")


if __name__ == "__main__":
    main()

