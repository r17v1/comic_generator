#!/usr/bin/env python3
"""Stage 3 — render the 5 panels (base pipeline + all char LoRAs, one pass each).

Usage (from project root):
  .venv/bin/python stage3_render_panels.py --run testruns/20261012_120000
  .venv/bin/python ... --run ... --upscale2x
  .venv/bin/python ... --run ... --repair panel_4.png --box 0.35,0.55,0.25,0.2

Skip logic: a panel (png + prompt txt) that already exists is never re-rendered.
--repair writes panel_<n>_repaired.png beside the original; never clobbers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from comgen import gpu, inpaint, io_utils, paths, runlog, sd, tags


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="run folder from stages 1+2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=paths.DEFAULT_STEPS)
    p.add_argument("--cfg", type=float, default=paths.DEFAULT_CFG)
    p.add_argument("--upscale2x", action="store_true", help="2x LANCZOS pass per panel")
    p.add_argument("--repair", default=None,
                   help="panel file to repair, e.g. 'panel_4.png'")
    p.add_argument("--box", default=None,
                   help="normalized (x,y,w,h) 0..1, comma separated, for --repair")
    p.add_argument("--repair-strength", type=float, default=0.6)
    return p.parse_args()


def parse_box(value: str) -> tuple[float, float, float, float]:
    try:
        box = tuple(float(v) for v in value.split(","))
    except ValueError as e:
        raise SystemExit(f"--box must be x,y,w,h floats (got '{value}'): {e}") from None
    if len(box) != 4 or not all(0.0 <= v <= 1.0 for v in box):
        raise SystemExit(f"--box values must all be in 0..1 (got {box})")
    return box


def load_adapters(run_dir: Path, assets: dict) -> dict[str, Path]:
    adapters: dict[str, Path] = {}
    for char in assets["characters"]:
        cid = char["character_id"]
        path = run_dir / "loras" / f"model_{cid}.safetensors"
        adapters[cid] = path
    return adapters


def render_panels(run_dir: Path, assets: dict, panels: list, adapters: dict[str, Path],
                  args, log) -> None:
    pipe = sd.load_pipeline()
    sd.load_loras(pipe, adapters)
    try:
        for panel in panels:
            n = panel["panel_num"]
            png = run_dir / f"panel_{n}.png"
            txt = run_dir / f"panel_{n}_prompt.txt"
            if png.exists() and txt.exists():
                log(f"panel_{n}: already rendered -> skipping")
                continue
            prompt = tags.panel_prompt(panel, assets)
            active = [c["character_id"] for c in panel["characters"]]
            w, h = sd.panel_size(panel["orientation"])
            log(f"panel_{n}: {w}x{h} {panel['rating']} loras={active or '(none)'} "
                f"seed={args.seed + n*2}")
            if txt.exists() and not png.exists():
                prompt = io_utils.read_text(txt)  # resume uses the saved prompt
            else:
                io_utils.write_text_once(txt, prompt + "\n")
            # Every panel keeps the standard "nsfw" negative.
            neg = tags.PANEL_NEGATIVE
            sd.set_active_adapters(pipe, active)
            img = sd.generate(pipe, prompt, w, h, steps=args.steps, cfg=args.cfg,
                              negative_prompt=neg, seed=args.seed + n)
            img.save(png)
            log(f"panel_{n}: saved {png.name}")
    finally:
        del pipe
        gpu.release_gpu()


def repair_panel(run_dir: Path, assets: dict, args, log) -> None:
    """Load the inpaint pipeline (separate job), load same LoRAs, fix one box."""
    name = args.repair
    panel_file = run_dir / name
    if not panel_file.exists():
        raise SystemExit(f"--repair target not found: {panel_file}")
    box = parse_box(args.box or "0.25,0.25,0.5,0.5")

    # find the panel spec by filename to reuse its prompt + active LoRAs
    panels = io_utils.read_json(run_dir / "panel_spec.json")
    num = int(name.replace("panel_", "").replace(".png", ""))
    panel = next((p for p in panels if p["panel_num"] == num), None)
    if panel is None:
        raise SystemExit(f"panel_{num} not found in panel_spec.json")
    # prefer the prompt that was actually used to render this panel
    saved = run_dir / f"panel_{num}_prompt.txt"
    if saved.exists():
        prompt = io_utils.read_text(saved).strip()
    else:
        prompt = tags.panel_prompt(panel, assets)
    active = [c["character_id"] for c in panel["characters"]]

    out = run_dir / f"panel_{num}_repaired.png"
    if out.exists():
        log(f"{out.name} already exists -> refusing to overwrite (delete to re-run)")
        return

    adapters = load_adapters(run_dir, assets)
    log(f"loading inpaint pipeline for repair of {name}, box={box}")
    pipe = inpaint.load_inpaint_pipeline()
    sd.load_loras(pipe, adapters)
    try:
        from PIL import Image
        img = Image.open(panel_file).convert("RGB")
        mask = inpaint.make_mask(img, box)
        neg = tags.PANEL_NEGATIVE  # standard "nsfw" negative
        sd.set_active_adapters(pipe, active)
        fixed = inpaint.repair(pipe, img, mask, prompt, strength=args.repair_strength,
                               negative_prompt=neg, seed=args.seed + num)
        fixed.save(out)
        log(f"repair saved: {out.name}")
    finally:
        del pipe
        gpu.release_gpu()


def upscale_all(run_dir: Path, log) -> None:
    from PIL import Image
    for png in sorted(run_dir.glob("panel_*.png")):
        if "_repaired" in png.name or "_2x" in png.name:
            continue
        out = run_dir / png.name.replace(".png", "_2x.png")
        if out.exists():
            log(f"{out.name} already exists -> skipping")
            continue
        img = Image.open(png).convert("RGB")
        w, h = img.size
        s = sd.upscale(img, 2)
        s.save(out)
        log(f"upscaled: {out.name} ({w}x{h} -> {w * 2}x{h * 2})")


def main() -> int:
    args = parse_args()
    run_dir = paths.run_dir_for(args.run)
    log = runlog.RunLog(run_dir)
    log(f"stage 3 starting in {run_dir}")

    assets = io_utils.read_json(run_dir / "assets.json")
    panels = io_utils.read_json(run_dir / "panel_spec.json")
    if args.box and not args.repair:
        raise SystemExit("--box requires --repair <panel_N.png>")

    missing = [run_dir / "loras" / f"model_{c['character_id']}.safetensors"
               for c in assets["characters"]
               if not (run_dir / "loras" / f"model_{c['character_id']}.safetensors").exists()]
    if missing:
        for m in missing:
            print(f"missing LoRA: {m}")
        print("Run stage 2 first (or check the run folder).")
        return 1

    render_panels(run_dir, assets, panels, load_adapters(run_dir, assets), args, log.log)

    if args.repair:
        repair_panel(run_dir, assets, args, log.log)

    if args.upscale2x:
        upscale_all(run_dir, log.log)

    log("stage 3 complete. Check quality bar in CHECKLIST.md §5 against panel_*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
