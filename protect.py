#!/usr/bin/env python3
"""Protect images with the online procedure in Algorithm 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sgd_guard.attack import DirectionBank, sd_attack_loss, sr_jpeg_loss, joint_features
from sgd_guard.augmentations import TransformConfig, apply_transform
from sgd_guard.data import ImageRecord, load_image, save_image, scan_images
from sgd_guard.diffusion import DiffusionPurifier, IdentityPurifier
from sgd_guard.gallery import GalleryBank, build_anchors, load_projection
from sgd_guard.models import arcface_features, clip_image_features, load_arcface, load_clip, load_facenet


def _records(input_path: Path) -> list[ImageRecord]:
    if input_path.is_file():
        return [ImageRecord(input_path, 0, input_path.name)]
    return scan_images(input_path, one_per_identity=False)


def protect_image(
    image: torch.Tensor,
    clip_model,
    identity_model,
    gallery: GalleryBank,
    anchors,
    directions: DirectionBank,
    purifier,
    args,
    clip_projection,
    identity_projection,
) -> tuple[torch.Tensor, dict[str, float]]:
    delta = torch.zeros_like(image)
    transform_config = TransformConfig()
    qualities = tuple(int(q) for q in args.jpeg_qualities.split(",") if q.strip())
    if not qualities:
        raise ValueError("--jpeg-qualities must contain at least one quality factor")

    last_metrics: dict[str, float] = {}
    for step in range(args.iterations):
        adversarial = (image + delta).clamp(0, 1).detach().requires_grad_(True)
        purified = purifier.forward_ste(adversarial) if not args.disable_purifier else adversarial
        l_sd, sd_debug = sd_attack_loss(
            clip_model, identity_model, purified, gallery, args.identity_encoder, anchors,
            clip_projection, identity_projection,
        )
        l_eot = torch.zeros((), device=image.device)
        if not args.disable_sd_eot:
            direction = -torch.autograd.grad(l_sd, sd_debug["current_joint"], retain_graph=True)[0].detach()
            direction = F.normalize(direction, dim=1)
            scores = 1.0 - direction @ directions.vectors.T
            weights = F.softmax(scores, dim=1).detach()
            per_transform = []
            for name in directions.names:
                transformed = apply_transform(purified, name, transform_config)
                value, _ = sd_attack_loss(
                    clip_model, identity_model, transformed, gallery, args.identity_encoder, anchors,
                    clip_projection, identity_projection,
                )
                per_transform.append(value)
            l_eot = (weights * torch.stack(per_transform)).sum(dim=1).mean()

        l_jpeg = torch.zeros((), device=image.device)
        if not args.disable_sr_jpeg:
            l_jpeg, _ = sr_jpeg_loss(
                clip_model, identity_model, purified, gallery, anchors, qualities,
                args.identity_encoder, clip_projection, identity_projection,
            )
        total = l_sd + l_eot + l_jpeg
        gradient = torch.autograd.grad(total, adversarial)[0]
        delta = (delta - args.step_size * gradient.sign()).clamp(-args.epsilon, args.epsilon)
        delta = (image + delta).clamp(0, 1) - image
        last_metrics = {
            "iteration": float(step + 1),
            "L_SD_attack": float(l_sd.detach()),
            "L_SD_EOT": float(l_eot.detach()),
            "L_SR_JPEG": float(l_jpeg.detach()),
            "L_total": float(total.detach()),
            "delta_linf": float(delta.detach().abs().max()),
        }
        print(
            f"[Protect] iter={step + 1:02d}/{args.iterations} "
            f"SD={last_metrics['L_SD_attack']:.5f} EOT={last_metrics['L_SD_EOT']:.5f} "
            f"JPEG={last_metrics['L_SR_JPEG']:.5f}"
        )
    return (image + delta).clamp(0, 1).detach(), last_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="SGD-Guard proactive face-swapping defense")
    parser.add_argument("--input", required=True, help="One image or an ImageFolder-style directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gallery", required=True, help="Offline gallery.h5 from build_gallery.py")
    parser.add_argument("--directions", required=True, help="Offline transform_directions.npz from estimate_directions.py")
    parser.add_argument("--arcface-ckpt", required=True)
    parser.add_argument("--farl-ckpt", required=True)
    parser.add_argument("--lora-path", default="", help="LCM-LoRA directory or .safetensors file")
    parser.add_argument("--models-root", default="")
    parser.add_argument("--projection-ckpt", default=None)
    parser.add_argument("--identity-encoder", choices=["arcface", "facenet"], default="arcface")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--purifier-steps", type=int, default=4)
    parser.add_argument("--purifier-strength", type=float, default=0.4)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--step-size", type=float, default=2.0 / 255.0, help="Pixel step in [0,1] units")
    parser.add_argument("--epsilon", type=float, default=3.0 / 255.0, help="L-infinity pixel budget in [0,1] units")
    parser.add_argument("--retrieval-k", type=int, default=10)
    parser.add_argument("--jpeg-qualities", default="30,50,70")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-purifier", action="store_true")
    parser.add_argument("--disable-sd-eot", action="store_true")
    parser.add_argument("--disable-sr-jpeg", action="store_true")
    parser.add_argument("--allow-base-clip", action="store_true")
    parser.add_argument("--save-tensors", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    records = _records(input_path)
    gallery = GalleryBank.load(args.gallery, device)
    clip_model = load_clip(args.farl_ckpt, device, args.allow_base_clip)
    identity_model = load_arcface(args.arcface_ckpt, device, args.models_root or None) if args.identity_encoder == "arcface" else load_facenet(device)
    anchors = build_anchors(gallery, clip_model, device, args.retrieval_k)
    directions = DirectionBank.load(args.directions, device)
    clip_projection, identity_projection = load_projection(args.projection_ckpt, device)
    if args.disable_purifier:
        purifier = IdentityPurifier()
    else:
        if not args.lora_path:
            raise ValueError("--lora-path is required unless --disable-purifier is set")
        purifier = DiffusionPurifier(args.model_id, args.lora_path, device, args.purifier_steps, args.purifier_strength, args.seed)

    metrics = {}
    for index, record in enumerate(records):
        image = load_image(record.path, args.image_size).unsqueeze(0).to(device)
        protected, metric = protect_image(
            image, clip_model, identity_model, gallery, anchors, directions, purifier, args,
            clip_projection, identity_projection,
        )
        relative = Path(record.relative_path)
        destination = output_dir / (record.path.name if input_path.is_file() else relative)
        destination = destination.with_suffix(".png")
        save_image(protected, destination)
        metrics[str(relative)] = metric
        if args.save_tensors:
            torch.save({"x": image.cpu(), "x_adv": protected.cpu(), "metrics": metric}, destination.with_suffix(".pt"))
        print(f"[Saved] {index + 1}/{len(records)} {destination}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

