"""Per-panel REPAIR only (PLAN C3): one masked region, never construction.

A feathered box in panel coordinates -> SDXL inpaint pipeline -> the same
panel with that region redrawn under the panel's own prompt.
"""
from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLInpaintPipeline
from transformers import CLIPTokenizer

from comgen import paths
from comgen import tags as tags_mod


def make_mask(image: Image.Image, box_norm: Sequence[float], feather_px: float | None = None) -> Image.Image:
    """White = repaint, black = keep. box_norm = (x, y, w, h) in 0..1.

    Edges are feathered with a Gaussian blur so the repair dissolves into
    the surrounding art instead of showing a hard rectangle.
    """
    x0, y0 = image.size[0] * box_norm[0], image.size[1] * box_norm[1]
    x1, y1 = image.size[0] * (box_norm[0] + box_norm[2]), image.size[1] * (box_norm[1] + box_norm[3])
    if feather_px is None:
        feather_px = max(6.0, min(image.size) * 0.015)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle([int(x0), int(y0), int(x1), int(y1)], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=feather_px))


def load_inpaint_pipeline() -> StableDiffusionXLInpaintPipeline:
    """Same base (Pony V6 XL + fp16 VAE), inpaint pipeline variant."""
    vae = AutoencoderKL.from_single_file(str(paths.VAE_MODEL), torch_dtype=torch.float16)
    tokenizer = CLIPTokenizer.from_pretrained(str(paths.TOKENIZER_DIR))
    pipe = StableDiffusionXLInpaintPipeline.from_single_file(
        str(paths.PONY_MODEL),
        vae=vae,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.vae.enable_tiling()
    return pipe


def repair(
    pipe: StableDiffusionXLInpaintPipeline,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    strength: float = 0.6,
    steps: int = paths.DEFAULT_STEPS,
    cfg: float = paths.DEFAULT_CFG,
    negative_prompt: str = tags_mod.PANEL_NEGATIVE,
    seed: int | None = None,
) -> Image.Image:
    """Redraw the masked region under the panel's own prompt (PLAN C3 defaults)."""
    image = image.convert("RGB")
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=steps,
            guidance_scale=cfg,
            strength=strength,
            generator=gen,
        )
    return out.images[0]
