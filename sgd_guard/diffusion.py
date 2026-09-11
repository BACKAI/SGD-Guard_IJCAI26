"""One purifier invocation per attack iteration, backed by an LCM-LoRA SD1.5 model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image


class IdentityPurifier(nn.Module):
    def forward_ste(self, images: Tensor) -> Tensor:
        return images


class DiffusionPurifier(nn.Module):
    def __init__(self, model_id: str, lora_path: str | Path, device: torch.device, steps: int = 4, strength: float = 0.4, seed: int | None = 42):
        super().__init__()
        from diffusers import LCMScheduler, StableDiffusionImg2ImgPipeline
        from diffusers.utils import logging
        logging.set_verbosity_error()
        self.device, self.steps, self.strength = device, steps, strength
        self.generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, safety_checker=None, requires_safety_checker=False, torch_dtype=torch.float32).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        lora = Path(lora_path).expanduser().resolve()
        if lora.is_file():
            self.pipe.load_lora_weights(str(lora.parent), weight_name=lora.name)
        elif lora.is_dir():
            try:
                self.pipe.load_lora_weights(str(lora))
            except Exception:
                from peft import PeftModel
                self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, str(lora)).merge_and_unload()
        else:
            raise FileNotFoundError(f"LoRA purifier path does not exist: {lora}")
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.vae.to(dtype=torch.float32)

    @torch.no_grad()
    def purify(self, images: Tensor) -> Tensor:
        original_size = images.shape[-2:]
        pil_images = [to_pil_image(image).resize((512, 512)) for image in images.detach().float().cpu().clamp(0, 1)]
        result = self.pipe(prompt=[""] * len(pil_images), image=pil_images, num_inference_steps=self.steps, strength=self.strength, guidance_scale=1.0, output_type="latent", generator=self.generator, return_dict=True)
        latents = result.images
        decoded = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        return F.interpolate(decoded, original_size, mode="bicubic", align_corners=False, antialias=True)

    def forward_ste(self, images: Tensor) -> Tensor:
        """Forward diffusion output with identity/BPDA gradient to the pixels."""
        purified = self.purify(images)
        return images + (purified - images).detach()

