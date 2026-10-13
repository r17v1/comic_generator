"""GPU hygiene.

Rule for the whole pipeline (24 GB card, one job at a time):
after every heavy phase (LLM / inference / training), release the
object in the caller (`del llm`) and call `release_gpu()` before
loading the next heavy model.
"""
from __future__ import annotations

import gc

import torch


def release_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
