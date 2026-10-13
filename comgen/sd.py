"""SDXL (Pony V6 XL) pipeline loading, multi-LoRA helpers, and generation.

One concern: the diffusion side. Callers own when to load and when to
release (comgen.gpu.release_gpu). No stage logic lives here.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline
from transformers import CLIPTokenizer

from comgen import paths
from comgen import tags as tags_mod


def load_pipeline() -> StableDiffusionXLPipeline:
    """Pony V6 XL + fp16 SDXL VAE + Euler scheduler, on cuda, VAE tiling on.

    Mirrors the proven loading order (single-file ckpt, explicit tokenizer,
    explicit VAE) so behavior is identical to the known-good setup.
    """
    vae = AutoencoderKL.from_single_file(
        str(paths.VAE_MODEL), torch_dtype=torch.float16)
    tokenizer = CLIPTokenizer.from_pretrained(str(paths.TOKENIZER_DIR))
    pipe = StableDiffusionXLPipeline.from_single_file(
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


def load_loras(pipe: StableDiffusionXLPipeline, adapters: dict[str, Path]) -> None:
    """Attach one LoRA per entry (adapter name -> file path).

    Nothing is activated yet — pick panels per panel via set_active_adapters.
    """
    for name, path in adapters.items():
        pipe.load_lora_weights(str(path), adapter_name=name)


def set_active_adapters(pipe: StableDiffusionXLPipeline, names: list[str]) -> None:
    """Enable exactly the listed adapters (weight 1.0); [] disables all.

    Empty list = location-only panels with no characters forced in.
    Note: pipe.set_adapters([]) crashes in diffusers (empty weight list is
    never built); disable_lora() is the proper API for 'clear all'.
    """
    if not names:
        pipe.disable_lora()
        return
    pipe.set_adapters(names)


def generate(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    width: int,
    height: int,
    steps: int = paths.DEFAULT_STEPS,
    cfg: float = paths.DEFAULT_CFG,
    negative_prompt: str = tags_mod.NEGATIVE,
    seed: int | None = None,
) -> Image.Image:
    """One full-canvas forward pass (the no-compositing rule)."""
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=gen,
        )
    return out.images[0]


def panel_size(orientation: str) -> tuple[int, int]:
    """Native panel ratio per PLAN C2: (width, height)."""
    return paths.PANEL_PORTRAIT if orientation == "portrait" else paths.PANEL_LANDSCAPE


def upscale(img: Image.Image, factor: int = 2) -> Image.Image:
    """PIL LANCZOS resize — the only allowed upscaler for the test (PLAN C4)."""
    w, h = img.size
    return img.resize((w * factor, h * factor), Image.LANCZOS)
