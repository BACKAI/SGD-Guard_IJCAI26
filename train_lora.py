#!/usr/bin/env python3
"""Train the SD1.5 LCM-LoRA purifier used by SGD-Guard.

This is a compact, single-GPU implementation of the teacher-consistency
training used in the released research code.  It trains only UNet LoRA
parameters; VAE, text encoder, and teacher UNet remain frozen.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sgd_guard.data import ImageRecord, load_image, scan_images


class FaceDataset(Dataset):
    def __init__(self, roots: list[str], resolution: int, max_samples: int | None = None):
        records = []
        for root in roots:
            records.extend(scan_images(root, one_per_identity=False))
        if max_samples is not None:
            records = records[:max_samples]
        self.records, self.resolution = records, resolution

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return load_image(self.records[index].path, self.resolution)


def _collate(batch):
    return torch.stack(batch, dim=0)


def _pred_x0(noisy, pred_noise, timesteps, alphas):
    alpha = alphas[timesteps].view(-1, 1, 1, 1)
    return (noisy - (1 - alpha).clamp_min(0).sqrt() * pred_noise) / alpha.clamp_min(1e-5).sqrt()


def main() -> None:
    parser = argparse.ArgumentParser(description="SGD-Guard one-step purifier LoRA training")
    parser.add_argument("--data-roots", required=True, help="Comma-separated CelebA-HQ and VGGFace2-HQ image roots")
    parser.add_argument("--pretrained-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--min-timestep", type=int, default=400)
    parser.add_argument("--max-timestep", type=int, default=800)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=4.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["no", "fp16", "bf16"], default="no")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.amp != "no":
        raise ValueError("--amp fp16/bf16 requires CUDA")
    dtype = torch.float32 if args.amp == "no" else (torch.float16 if args.amp == "fp16" else torch.bfloat16)

    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoTokenizer, CLIPTextModel

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = [root.strip() for root in args.data_roots.split(",") if root.strip()]
    dataset = FaceDataset(roots, args.resolution, args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, collate_fn=_collate, pin_memory=True)
    if len(loader) == 0:
        raise ValueError("No complete training batch; reduce --batch-size or provide more images")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder").to(device, dtype=dtype).eval()
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(device, dtype=dtype).eval()
    teacher = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet").to(device, dtype=dtype).eval()
    student = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet").to(device)
    for frozen in (text_encoder, vae, teacher):
        frozen.requires_grad_(False)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2", "time_emb_proj"],
    )
    student = get_peft_model(student, lora_config).to(device)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    alphas = noise_scheduler.alphas_cumprod.to(device)
    prompt_ids = tokenizer([""] * args.batch_size, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        prompt_embeds = text_encoder(prompt_ids)[0]
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp == "fp16")
    steps = 0
    print(f"[LoRA] device={device} samples={len(dataset)} batches/epoch={len(loader)} trainable={sum(p.numel() for p in trainable):,}")

    for epoch in range(args.epochs):
        student.train()
        for images in loader:
            images = images.to(device, non_blocking=True) * 2 - 1
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=args.amp != "no"):
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
                timesteps = torch.randint(args.min_timestep, args.max_timestep + 1, (latents.shape[0],), device=device).long()
                noise = torch.randn_like(latents)
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)
                prediction = student(noisy, timesteps, encoder_hidden_states=prompt_embeds[:latents.shape[0]]).sample
                with torch.no_grad():
                    teacher_prediction = teacher(noisy, timesteps, encoder_hidden_states=prompt_embeds[:latents.shape[0]]).sample
                clean_prediction = _pred_x0(noisy.float(), prediction.float(), timesteps, alphas)
                loss = F.mse_loss(prediction.float(), teacher_prediction.float()) + args.reconstruction_weight * F.mse_loss(clean_prediction, latents.float())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            if steps == 1 or steps % 50 == 0:
                print(f"[LoRA] step={steps} epoch={epoch + 1} loss={float(loss.detach()):.6f}")
            if args.max_steps is not None and steps >= args.max_steps:
                break
        if args.max_steps is not None and steps >= args.max_steps:
            break

    lora_state = get_peft_model_state_dict(student, adapter_name="default")
    StableDiffusionPipeline.save_lora_weights(str(output / "unet_lora"), unet_lora_layers=lora_state)
    torch.save({"state_dict": lora_state, "model_id": args.pretrained_model, "steps": steps}, output / "adapter_model.pt")
    (output / "training_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"[Done] LoRA weights saved under {output / 'unet_lora'}")


if __name__ == "__main__":
    main()

