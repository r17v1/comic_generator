#!/usr/bin/env python3
"""Stage 2 — train one LoRA per character with kohya sd-scripts (its own venv).

Usage (from project root):
  .venv/bin/python stage2_train_loras.py --run testruns/20261012_120000
  .venv/bin/python stage2_train_loras.py --run testruns/20261012_120000 --steps 1000 --save-every 100

Skip logic: a character is skipped when `loras/model_<char>.safetensors`
already exists. To retrain one, delete that file first.
"""
from __future__ import annotations

import argparse
import sys

from comgen import io_utils, kohya, paths, runlog


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="run folder from stage 1")
    p.add_argument("--steps", type=int, default=1000,
                   help="training steps (default 1000 ~= 6.25 epochs over 160 imgs)")
    p.add_argument("--dim", type=int, default=32, help="LoRA rank")
    p.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    p.add_argument("--save-every", type=int, default=100, help="checkpoint every N steps")
    return p.parse_args()


def train_one(run_dir, cid: str, args, log) -> int:
    """One character: dataset config -> kohya subprocess -> promote checkpoint."""
    refs_dir = run_dir / "refs" / cid
    cfg_path = run_dir / "train" / f"dataset_{cid}.toml"
    if cfg_path.exists():
        log(f"train/dataset_{cid}.toml exists -> reusing it (never overwritten)")
    else:
        cfg_path = kohya.write_dataset_config(run_dir, cid, refs_dir)

    train_args = kohya.build_train_args(
        run_dir, cid, cfg_path,
        steps=args.steps, dim=args.dim, alpha=args.alpha,
        lr=args.lr, save_every=args.save_every)

    log(f"=== training {cid} (rank {args.dim} / alpha {args.alpha} / "
        f"{args.steps} steps / save every {args.save_every}) ===")
    rc = kohya.train_character(run_dir, cid, cfg_path, train_args, log)
    if rc != 0:
        log(f"ERROR: trainer for {cid} exited with code {rc}; see train/{cid} logs")
        return 2

    dst = kohya.finalize_lora(run_dir, cid, log)
    if dst is None:
        log(f"ERROR: no checkpoint found for {cid} in train/out/")
        return 3
    return 0


def main() -> int:
    args = parse_args()
    run_dir = paths.run_dir_for(args.run)
    log = runlog.RunLog(run_dir)
    log(f"stage 2 starting in {run_dir}")

    ok, missing = kohya.ready()
    if not ok:
        for m in missing:
            print(f"missing: {m}")
        print()
        print(kohya.setup_hint())
        return 1

    assets = io_utils.read_json(run_dir / "assets.json")
    failed = 0
    for char in assets["characters"]:
        cid = char["character_id"]
        status = kohya.run_status(run_dir, cid)
        if status == "done":
            log(f"{cid}: LoRA already trained -> skipping")
            continue
        if status == "missing":
            log(f"ERROR: {cid} has no refs in this run folder — run stage 1 first")
            failed = 1
            continue
        rc = train_one(run_dir, cid, args, log.log)
        if rc != 0:
            failed = 1
            break  # one GPU job at a time: stop on failure, user investigates

    if failed:
        return failed
    log("stage 2 complete. Next: stage3_render_panels.py --run " + run_dir.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
