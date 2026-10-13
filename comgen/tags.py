"""Deterministic Pony-style prompt building (checklist 3.13 tag order).

No LLM, no GPU, no third-party deps: pure string functions over the
validated structures from comgen.schema. This is the ONLY place that
decides tag order; everywhere else just calls these.
"""
from __future__ import annotations

# --- standard Pony quality / score tags ---
QUALITY = "score_9, score_8_up, score_7_up"

# Documented default negative block for Pony V6 XL.
NEGATIVE = (
    "score_4, score_5, score_6, score_6_up, score_7_up, "
    "nsfw, bad anatomy, bad hands, extra limbs, extra digits, "
    "missing fingers, blurry, lowres, jpeg artifacts, "
    "watermark, signature, username, text, logo"
)

_PANEL_SUFFIX = "multiple views, split screen, panel frame, speech bubble, comic grid"

# Panels additionally: keep it one clean image, no comic machinery.
PANEL_NEGATIVE = NEGATIVE + f", {_PANEL_SUFFIX}"

# Refs additionally: exactly one isolated figure on a plain background.
# All refs are fully clothed, so the standard "nsfw" negative
# applies to every one of them (no per-view variants).
REF_NEGATIVE = NEGATIVE + ", multiple views, another person, couple, background details, cluttered background"


def count_anchor(genders: list[str]) -> str:
    """Booru-style count anchor: ['female','male'] -> '1girl, 1boy'."""
    girls = genders.count("female")
    boys = genders.count("male")
    parts = []
    if girls:
        parts.append("1girl" if girls == 1 else f"{girls}girls")
    if boys:
        parts.append("1boy" if boys == 1 else f"{boys}boys")
    return ", ".join(parts)


def ref_prompt(character: dict, view: str) -> str:
    """One reference-image prompt.

    Order: quality -> count anchor -> trigger + view -> appearance caption.
    The LoRA must learn exactly `trigger + lora_caption`, so the caption
    stays verbatim and the view phrase stays separate.
    """
    parts = [
        QUALITY,
        count_anchor([character["gender"]]),
        f"{character['character_id']}, {view.lower()}",
        character["lora_caption"].lower(),
    ]
    return ", ".join(p for p in parts if p)


# SDXL hard-caps prompts at 77 CLIP tokens on BOTH encoders (CLIP-L and bigG
# tokenize differently) and silently drops the tail. The working target is
# ~66 tokens (user: diminishing returns after 66; hard ceiling 70). So the
# least-weighted tags must sit at the end. Location lists are capped to the
# first N phrases (concrete scene beats; LLM puts generic style words last).
# 5->3 (2026-10-12 user call): 2-char panels measured 78-79 tokens over the 77
# hard cap; trimming location to the first 3 phrases lands totals ~70-71.
LOCATION_TAG_CAP = 3


def panel_prompt(panel: dict, assets: dict) -> str:
    """One panel prompt, in the locked 3.13 order, tuned to ~66 tokens
    (77-token hard cap on both encoders, 70 max by user choice):

    quality -> count anchor -> rating (always rating_mature, Pony quality dial)
    -> per-character (TRIGGER + ONE rotated
    anchor entry, action, expression) -> location (capped)
    -> shot/camera/mood -> fx.

    Identity budget (user): name + at most ONE anchor entry, under 7 tokens —
    panels never repeat the full identification set. The entry rotates with
    panel number (offset per character) so different panels reinforce
    different features; the LoRA (trained to the trigger alone) carries the
    rest. Rating up front (Pony score-class dial); identity early so it
    survives any truncation.
    """
    char_by_id = {c["character_id"]: c for c in assets["characters"]}
    loc_by_id = {l["location_id"]: l for l in assets["locations"]}

    entries = [char_by_id[c["character_id"]] for c in panel["characters"]]
    parts = [QUALITY]
    if entries:
        parts.append(count_anchor([e["gender"] for e in entries]))
    parts.append(panel["rating"])
    for pos, (e, spec) in enumerate(zip(entries, panel["characters"])):
        ident = e["character_id"]
        anchors = e.get("anchors") or [
            a.strip() for a in e.get("anchor", "").split(",") if a.strip()]
        if anchors:
            idx = (panel.get("panel_num", 0) + pos) % len(anchors)
            ident += f", {anchors[idx]}"
        parts.append(
            f"{ident}, {spec['action'].lower()}, {spec['expression'].lower()}"
        )
    loc = [t.strip() for t in loc_by_id[panel["location_id"]]["description_tags"].split(",")]
    parts.append(", ".join(t.lower() for t in loc[:LOCATION_TAG_CAP]))
    if (scm := [x for x in (panel["shot"], panel["camera"], panel["mood"]) if x]):
        parts.append(", ".join(x.lower() for x in scm))
    if panel.get("fx"):
        parts.append(panel["fx"].lower())
    return ", ".join(p for p in parts if p)
