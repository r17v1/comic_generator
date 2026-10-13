"""Project paths and run-folder helpers. Import-safe: no heavy deps."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- models (kept untracked by design) ---
MODELS_DIR = PROJECT_ROOT / "models"
LLM_MODEL = MODELS_DIR / "llm" / "qwen3.6-27b-uncensored-heretic-v2-q4_k_m.gguf"
IMAGE_DIR = MODELS_DIR / "image"
PONY_MODEL = IMAGE_DIR / "pony-diffusion-v6-xl.safetensors"
VAE_MODEL = IMAGE_DIR / "sdxl-vae.safetensors"
TOKENIZER_DIR = IMAGE_DIR / "tokenizer"

# --- interpreters / tools ---
APP_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
KOHYA_PYTHON = PROJECT_ROOT / ".venv-kohya" / "bin" / "python"
KOHYA_SCRIPTS = PROJECT_ROOT / "tools" / "sd-scripts"
KOHYA_TRAIN = KOHYA_SCRIPTS / "sdxl_train_network.py"

# --- output ---
TESTRUNS_DIR = PROJECT_ROOT / "testruns"

# --- rendering defaults (locked: PLAN §4) ---
REF_SIZE = 1024                 # square ref renders (kohya 1024 bucket, no bucketing)
PANEL_PORTRAIT = (896, 1152)    # (width, height)
PANEL_LANDSCAPE = (1152, 896)
DEFAULT_STEPS = 30
DEFAULT_CFG = 6.0


def new_run_dir() -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = TESTRUNS_DIR / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir_for(arg: str | None) -> Path:
    """Resolve the run folder: explicit path if given, else a fresh timestamped one."""
    if arg:
        d = Path(arg).expanduser().resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d
    return new_run_dir()
