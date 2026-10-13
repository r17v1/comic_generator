"""Kohya sd-scripts LoRA training (stage 2 machinery).

Kohya runs in its OWN venv (paths.KOHYA_PYTHON); this module only builds
its inputs (dataset config, CLI args), streams its log into the run folder,
and then promotes the finished checkpoint to `loras/model_<char>.safetensors`.

Flags verified against kohya sd-scripts main @ 690ea7f (2026-09): entry point
`sdxl_train_network.py`, `--pretrained_model_name_or_path`, `--vae`,
`--dataset_config`, `--network_module networks.lora`, `--network_dim/-alpha`,
`--max_train_steps` (loop is steps-driven; `--stop_at_max_steps` no longer exists).
Dataset TOML: `[general]` + `[[datasets]]` with `subsets = [{image_dir, ...}]`.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from comgen import io_utils, paths

# --- knobs isolated for easy flipping after the first real kohya run ---
VAE_FLAG = "--vae"         # verified: library/args.py L332
DEFAULT_REPEATS = 4        # 24 refs x 4 = 96 imgs/epoch; ~10.4 epochs @ 1000 steps
SEED = 42

LOREX = re.compile(r"(\d+)\.safetensors$")  # trailing step digits: "-200." and "-step00000200." both match


def ready() -> tuple[bool, list[str]]:
    """Is the kohya stack installed? (checklist: stage 2 exits with a hint if not)"""
    missing = []
    if not paths.KOHYA_PYTHON.exists():
        missing.append(str(paths.KOHYA_PYTHON))
    if not paths.KOHYA_TRAIN.exists():
        missing.append(str(paths.KOHYA_TRAIN))
    return (not missing, missing)


def setup_hint() -> str:
    """Copy-paste block for the missing parts (CPU-only, no idle GPU needed)."""
    return "\n".join([
        "Kohya training stack is missing. Run from the project root:",
        "  cd " + str(paths.PROJECT_ROOT),
        "  python3 -m venv .venv-kohya",
        "  .venv-kohya/bin/pip install -U pip",
        "  git clone https://github.com/kohya-ss/sd-scripts tools/sd-scripts",
        "  cd tools/sd-scripts   # requirements.txt ends with '-e .', must resolve to the repo",
        "  ../../.venv-kohya/bin/pip install -r requirements.txt",
        "  # if you get a CUDA sm_120 / arch error afterwards, install a Blackwell wheel first:",
        "  # .venv-kohya/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128",
    ])


def dataset_config_toml(refs_dir: Path) -> str:
    """Kohya dataset config (2026 layout) for one character's refs subfolder.
    Schema: [general] dataset-level params + [[datasets]] each with a subsets
    array; DB subset requires `image_dir`; captions are our sibling .txt files."""
    quoted = str(refs_dir).replace("\\", "\\\\").replace('"', '\\"')
    return (
        "[general]\n"
        "enable_bucket = true\n"
        "batch_size = 1\n"
        "resolution = 1024\n"
        # OFF: kohya forbids shuffle_caption when caching TE outputs (VRAM win).
        # Captions already vary per image (trigger + a rotated pair of the
        # character's anchor entries), so the trigger is never fused to one
        # fixed full set — a deterministic rotation beats shuffling anyway.
        "shuffle_caption = false\n"
        "keep_tokens = 1\n"
        "\n"
        "[[datasets]]\n"
        f'subsets = [ {{ image_dir = "{quoted}", num_repeats = {DEFAULT_REPEATS}, '
        f'caption_extension = ".txt" }} ]\n'
    )


def write_dataset_config(run_dir: Path, char_id: str, refs_dir: Path) -> Path:
    """ALWAYS overwritten: this is generated input, not a trained artifact
    (checkpoints/LORAs remain write-once). A stale config would otherwise
    silently keep old settings like shuffle_caption.
    image_dir MUST be absolute: kohya runs with cwd=tools/sd-scripts, so a
    relative dir would resolve to nothing and the dataset comes up empty."""
    path = (run_dir / "train" / f"dataset_{char_id}.toml").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dataset_config_toml(Path(refs_dir).resolve()), encoding="utf-8")
    return path


def build_train_args(
    run_dir: Path,
    char_id: str,
    config_toml: Path,
    steps: int = 1000,
    dim: int = 32,
    alpha: int = 16,
    lr: float = 1e-4,
    save_every: int = 100,
) -> list[str]:
    """sdxl_train_network.py args (verified against kohya 2026-09); everything
    on-disk, no downloads (PLAN locked values). Loop is exactly `steps` long."""
    return [
        str(paths.KOHYA_TRAIN),
        "--pretrained_model_name_or_path", str(paths.PONY_MODEL),
        VAE_FLAG, str(paths.VAE_MODEL),
        "--dataset_config", str(config_toml),
        "--output_dir", str(run_dir / "train" / "out"),
        "--output_name", char_id,
        "--network_module", "networks.lora",
        "--network_dim", str(dim),
        "--network_alpha", str(alpha),
        "--network_train_unet_only",   # kohya forbids TE training + TE caching; identity lives in UNet
        "--learning_rate", str(lr),
        "--max_train_steps", str(steps),
        "--save_every_n_steps", str(save_every),
        "--train_batch_size", "1",
        "--mixed_precision", "bf16",
        # --- VRAM levers for the 24GB 5090 (batch is already 1) ---
        "--sdpa",                            # memory-efficient attention (PyTorch 2.14 ok)
        "--gradient_checkpointing",          # ~70% less activation memory, modest slowdown
        "--cache_latents",                   # VAE latents precomputed once, held in main RAM
        "--cache_text_encoder_outputs",      # same for both TEs (freed from hot path)
        "--seed", str(SEED),
        "--save_state",  # keep optimizer state so a rerun can continue
    ]


def train_character(run_dir: Path, char_id: str, cfg_path: Path, args: list[str], log) -> int:
    """Run one training job; stream stdout to `log` and the console. Returns exit code."""
    proc = subprocess.Popen(
        [str(paths.KOHYA_PYTHON), *args],
        cwd=str(paths.KOHYA_SCRIPTS),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip("\n"))
        print(line, end="")
    return proc.wait()


def find_best_lora(output_dir: Path, char_id: str) -> Path | None:
    """Highest-step checkpoint written for this character (any of kohya's naming schemes:
    `<name>-<step>.safetensors`, `<name>-step0000NNNN.safetensors` (this kohya version).
    The stepless final `<name>.safetensors` is ignored (a stepped one always exists beside it)."""
    best: Path | None = None
    best_step = -1
    if not output_dir.exists():
        return None
    for p in sorted(output_dir.glob(f"{char_id}*.safetensors")):
        m = LOREX.search(p.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best, best_step = p, step
    return best


def finalize_lora(run_dir: Path, char_id: str, log) -> Path | None:
    """Promote the best checkpoint to loras/model_<char>.safetensors (never clobbered)
    and record it in the cheap-to-rebuild loras/index.json."""
    src = find_best_lora(run_dir / "train" / "out", char_id)
    if src is None:
        return None
    dst = run_dir / "loras" / f"model_{char_id}.safetensors"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, dst)
    index_path = run_dir / "loras" / "index.json"
    index: dict = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    index[char_id] = {
        "file": dst.name,
        "source": str(src.name),
        "trigger": char_id,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    log(f"LoRA ready: {dst}  (from {src})")
    return dst


def run_status(run_dir: Path, char_id: str) -> str:
    """'done' | 'ready' | 'missing' — drives stage 2 skip logic.

    ready    = refs exist on disk, LoRA not yet trained (stage 2 should train it)
    missing  = no refs; stage 1 was not (fully) run for this character
    """
    if (run_dir / "loras" / f"model_{char_id}.safetensors").exists():
        return "done"
    refs = run_dir / "refs" / char_id
    if refs.is_dir() and any(refs.glob("*.png")):
        return "ready"
    return "missing"
