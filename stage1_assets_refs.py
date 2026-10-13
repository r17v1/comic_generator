#!/usr/bin/env python3
"""Stage 1 — assets.json + panel_spec.json (LLM) then per-character ref sets (SDXL).

Usage (from project root):
  .venv/bin/python stage1_assets_refs.py --prompt-file test_prompt.txt
  .venv/bin/python stage1_assets_refs.py --run testruns/20261012_120000 --skip-llm

Never overwrites: assets.json / panel_spec.json / any existing ref image.
"""
from __future__ import annotations

import argparse
import faulthandler
import sys
from pathlib import Path

# Hard-crash forensics: on SIGSEGV/SIGBUS/SIGABRT, dump the Python (and C)
# stack of every thread to stderr *before* the process dies. Must be enabled
# before any native library (torch / llama-cpp) is imported.
faulthandler.enable()

from comgen import gpu, io_utils, llm, llm_prompts, paths, runlog, sd, schema, tags, tokens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default=None,
                   help="run folder (default: fresh testruns/<timestamp>/)")
    p.add_argument("--prompt-file", default=str(paths.PROJECT_ROOT / "test_prompt.txt"),
                   help="story prompt text file (default: test_prompt.txt)")
    p.add_argument("--seed", type=int, default=42,
                   help="base seed for ref renders (each view gets seed+i)")
    p.add_argument("--steps", type=int, default=paths.DEFAULT_STEPS,
                   help=f"ref generation steps (default {paths.DEFAULT_STEPS})")
    p.add_argument("--cfg", type=float, default=paths.DEFAULT_CFG,
                   help=f"ref generation CFG (default {paths.DEFAULT_CFG})")
    p.add_argument("--skip-llm", action="store_true",
                   help="reuse existing assets.json + panel_spec.json")
    p.add_argument("--skip-refs", action="store_true",
                   help="reuse existing ref images")
    return p.parse_args()


def _core_validate(data) -> dict:
    """schema.validate_assets + the anchor-entry token cap.

    A character's per-panel identity is trigger (1 token) + ONE rotating
    anchor entry; the budget is under 7 tokens (3 characters in one frame
    would otherwise burn 21+ on ID alone), so each entry must fit
    tokens.MAX_IDENTIFIER_TOKENS on BOTH SDXL encoders. The error message
    goes back to the LLM on its one retry (llm_json)."""
    assets = schema.validate_assets(data)
    for c in assets["characters"]:
        tokens.check_identifiers(c["anchors"])
    return assets


def phase_assets(run_dir: Path, story: str, log) -> dict:
    """LLM phase: core assets -> per-character ref groups (3 calls/char)
    -> assets.json (with ref_plan + 24 ref_views) -> 5-panel plan.

    Every step is written once and resume-able: assets_core.json and
    refplan_<char>.json are persisted right after validation, so a failed
    run only re-executes the LLM calls that never saved. assets.json is
    written only once EVERYTHING validated. Skipped entirely when the two
    final files already exist."""
    assets_path, spec_path = run_dir / "assets.json", run_dir / "panel_spec.json"
    if assets_path.exists() and spec_path.exists():
        log("assets.json + panel_spec.json already present -> LLM phase skipped")
        return io_utils.read_json(assets_path)

    log("loading LLM (~18 GB, GPU exclusive)...")
    model = llm.create_llm(str(paths.LLM_MODEL), log=log)

    # --- 1. core: identities + locations ------------------------------------
    core_path = run_dir / "assets_core.json"
    if core_path.exists():
        log("assets_core.json present -> reusing core (skipping core LLM call)")
        assets = io_utils.read_json(core_path)
    else:
        log("LLM core call starting (JSON: characters + locations)...")
        assets = llm.llm_json(model, llm_prompts.ASSETS_SYSTEM,
                              llm_prompts.ASSETS_USER.format(story=story),
                              _core_validate, log=log)
        io_utils.write_json_once(core_path, assets)
        log(f"core call OK; assets_core.json written: "
            + ", ".join(c["character_id"] for c in assets["characters"]))

    # --- 2. per character: 3 outfit groups x 8 poses -------------------------
    n_states = len(schema.REF_STATES)
    for idx, char in enumerate(assets["characters"]):
        cid = char["character_id"]
        plan_path = run_dir / f"refplan_{cid}.json"
        if plan_path.exists():
            log(f"refplan_{cid}.json present -> reusing 3 ref groups for {cid}")
            groups = io_utils.read_json(plan_path)
        else:
            groups = []
            for i in range(n_states):
                state = schema.REF_STATES[i][0]
                log(f"LLM ref-group call {cid} {i + 1}/{n_states} ({state})...")
                groups.append(llm.llm_json(
                    model, llm_prompts.REF_GROUP_SYSTEM,
                    llm_prompts.ref_group_user(char, i),
                    lambda d, i=i: schema.clean_ref_group(i, d), log=log))
            io_utils.write_json_once(plan_path, groups)
            log(f"refplan_{cid}.json written ({len(groups)} groups x 8 poses)")
        assets["characters"][idx] = schema.add_ref_plan(char, groups)

    io_utils.write_json_once(assets_path, assets)
    log("assets.json written (ref_plan + ref_views attached): "
        + ", ".join(f"{c['character_id']}={len(c['ref_views'])} refs"
                    for c in assets["characters"]))

    # --- 3. the 5-panel plan --------------------------------------------------
    char_ids = [c["character_id"] for c in assets["characters"]]
    loc_ids = [l["location_id"] for l in assets["locations"]]
    log("LLM panels call starting (JSON: 5-panel plan)...")
    panels = llm.llm_json(
        model, llm_prompts.PANELS_SYSTEM,
        llm_prompts.panels_user(story, char_ids, loc_ids),
        lambda data: schema.validate_panels(data, assets), log=log)
    io_utils.write_json_once(spec_path, panels)
    log(f"panel_spec.json written: {len(panels)} panels "
        + " / ".join(p["rating"] for p in panels))

    log("LLM phase done -> del model + release_gpu ...")
    del model
    gpu.release_gpu()
    log("GPU released; entering refs phase (SDXL)...")
    return assets


def phase_refs(run_dir: Path, assets: dict, args, log) -> int:
    """Render each character's ref set (1024x1024, base model, no LoRA)."""
    pipe = sd.load_pipeline()
    n_new = n_skip = 0
    try:
        for char in assets["characters"]:
            cid = char["character_id"]
            refs_dir = run_dir / "refs" / cid
            refs_dir.mkdir(parents=True, exist_ok=True)
            # Varyed captions: `trigger + a rotated pair of anchor entries`
            # per ref image. Over 24 refs each entry appears in most captions
            # but the FULL set never appears at once, so the LoRA binds the
            # identity to the trigger ALONE — inference then only needs
            # name + one entry (or none).
            anchors = char.get("anchors") or [
                a.strip() for a in char.get("anchor", "").split(",") if a.strip()]
            if not anchors:
                anchors = [char["lora_caption"].lower()]
            for i, view in enumerate(char["ref_views"], start=1):
                png = refs_dir / f"{i:02d}.png"
                txt = refs_dir / f"{i:02d}.txt"
                if png.exists():
                    n_skip += 1
                else:
                    prompt = tags.ref_prompt(char, view)
                    log(f"ref {cid}/{i:02d}: {view}  (seed={args.seed + i})")
                    img = sd.generate(pipe, prompt, paths.REF_SIZE, paths.REF_SIZE,
                                      steps=args.steps, cfg=args.cfg,
                                      negative_prompt=tags.REF_NEGATIVE,
                                      seed=args.seed + i)
                    img.save(png)
                    n_new += 1
                if not txt.exists():
                    a1 = anchors[(i - 1) % len(anchors)]
                    a2 = anchors[i % len(anchors)] if len(anchors) >= 2 else a1
                    txt.write_text(f"{cid}, {a1}, {a2}\n", encoding="utf-8")
            log(f"refs for {cid}: complete ({len(char['ref_views'])} images in refs/{cid}/)")
    finally:
        del pipe
        gpu.release_gpu()
    log(f"refs: {n_new} new, {n_skip} reused")
    return n_new + n_skip


def main() -> int:
    args = parse_args()
    run_dir = paths.run_dir_for(args.run)
    log = runlog.RunLog(run_dir)
    log(f"stage 1 starting in {run_dir}")

    if not args.skip_llm:
        prompt_file = Path(args.prompt_file)
        if not prompt_file.exists():
            log(f"ERROR: prompt file not found: {prompt_file}")
            return 1
        story = io_utils.read_text(prompt_file)
        phase_assets(run_dir, story, log.log)
    else:
        assets_path = run_dir / "assets.json"
        if not assets_path.exists():
            log("ERROR: --skip-llm but assets.json is missing in this run folder")
            return 1

    if args.skip_refs:
        log("refs phase skipped (--skip-refs)")
        return 0

    assets = io_utils.read_json(run_dir / "assets.json")
    phase_refs(run_dir, assets, args, log.log)

    log("stage 1 complete. Next: stage2_train_loras.py --run " + run_dir.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
