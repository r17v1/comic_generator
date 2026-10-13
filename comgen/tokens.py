"""CLIP token counting (both SDXL encoders) for prompt budget guarantees.

SDXL truncates prompts at 77 tokens on BOTH encoders (CLIP-L and bigG), and
the two tokenizers split words differently — any hard budget must hold on
BOTH. `transformers` is imported lazily so CPU-only code paths don't pay
the import cost (same cache-resolved bigG dir as promptlab.py).

Budget rule (user, 2026-10-12): a character's per-panel identity is
`name + at most ONE anchor entry` and must stay under 7 tokens (3
characters x >7 would burn 21+ tokens identifying people). So:

    trigger (1) + comma (1) + entry (<= MAX_IDENTIFIER_TOKENS) <= 6 < 7

LoRA TRAINING uses MORE (a rotating pair of entries per ref image, so the
model binds identity to the trigger alone and never expects a fixed full
set) — only the per-panel single-entry cap is enforced here.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from comgen import paths

MAX_IDENTIFIER_TOKENS = 4

_HF_BIGG = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"


def _bigg_tokenizer_dir() -> Path:
    """Resolve the bigG tokenizer snapshot dir straight from the HF cache
    (no network, no huggingface_hub — try_to_load_from_cache is flaky offline)."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    repo = hub / f"models--{_HF_BIGG.replace('/', '--')}"
    if not repo.exists():
        raise FileNotFoundError(f"bigG tokenizer not found in HF cache ({repo})")
    main = (repo / "refs" / "main").read_text().strip()
    return repo / "snapshots" / main


@lru_cache(maxsize=1)
def _tokenizers() -> tuple:
    from transformers import CLIPTokenizer, CLIPTokenizerFast
    clip_l = CLIPTokenizer.from_pretrained(str(paths.TOKENIZER_DIR))
    big_g = CLIPTokenizerFast.from_pretrained(str(_bigg_tokenizer_dir()))
    return clip_l, big_g


@lru_cache(maxsize=512)
def count_both(text: str) -> tuple[int, int]:
    """Un-padded token count of `text` on (CLIP-L, bigG)."""
    clip_l, big_g = _tokenizers()
    return (
        len(clip_l(text, add_special_tokens=False)["input_ids"]),
        len(big_g(text, add_special_tokens=False)["input_ids"]),
    )


def check_identifiers(anchors: list[str]) -> None:
    """Raise ValueError when any anchor entry exceeds MAX_IDENTIFIER_TOKENS
    on either encoder. Called by the stage-1 core validator before anything
    is written; llm_json feeds the error back to the model on its retry."""
    for a in anchors:
        n_clip, n_bigg = count_both(a)
        if n_clip > MAX_IDENTIFIER_TOKENS or n_bigg > MAX_IDENTIFIER_TOKENS:
            raise ValueError(
                f"anchor entry '{a}' is {n_clip} CLIP-L / {n_bigg} bigG tokens — "
                f"max is {MAX_IDENTIFIER_TOKENS} on both: the trigger name is the "
                f"main identifier, a panel shows name + at most one entry, and "
                f"that must stay under 7 tokens (use 1-3 common words)"
            )
