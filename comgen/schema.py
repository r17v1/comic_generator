"""Schema validation for assets.json and panel_spec.json.

Each validator takes the raw parsed JSON (dict or list, whatever the
LLM managed) and returns a normalized, safe structure. Raises ValueError
with a specific, actionable message on failure (the message is fed back
to the LLM on retry).
"""
from __future__ import annotations

import re

RATINGS = ("rating_mature",)  # the only rating dial (Pony quality tag)
# character_id: LoRA trigger token + filename, so one clean lowercase word.
CHAR_ID_RE = re.compile(r"^[a-z][a-z]{2,19}$")
# location_id: purely a label for prompt text, snake_case is fine.
LOC_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,49}$")


class SchemaError(ValueError):
    pass


# score_/rating_/nsfw tokens must never leak into LLM-authored identity or
# clothing text — the pipeline owns those tags (tags.py).
_TAG_RE = re.compile(r"\b(score_\w+|rating_\w+|nsfw|sfw)\b", re.I)

# Garment words. If one sneaks into lora_caption/anchors the LoRA fuses that
# outfit into the face; in a POSE it double-specifies what the clothing field
# already says. Word-bounded so "hands", "shoulder", "dressings" never match.
_CLOTHING_RE = re.compile(
    r"\b(bra|bralette|bralettes|panties|panty|briefs|underwear|lingerie|"
    r"stockings|stocking|thong|thongs|garters|garter|bodysuit|corset|"
    r"jacket|coat|hoodie|sweater|sweatshirt|shirt|tee|tank|blouse|cardigan|"
    r"gown|dress|skirt|pants|jeans|shorts|suit|poncho|leggings|bodice|apron|"
    r"kimono|robes|robe|clothes|clothing|garmen|bikini|swimsuit|swimwear|"
    r"uniform|outfit|vest|cape|top)\b",
    re.I,
)


def _obj(data, key: str):
    try:
        v = data[key]
    except (KeyError, TypeError, IndexError) as e:
        raise SchemaError(f"missing required field '{key}'") from e
    if not isinstance(v, dict):
        raise SchemaError(f"field '{key}' must be an object")
    return v


def _str(o: dict, key: str, default: str = "") -> str:
    v = o.get(key, default)
    if not isinstance(v, str) or not v.strip():
        raise SchemaError(f"'{key}' must be a non-empty string")
    return v.strip()


def _clean_character(o: dict, idx: int) -> dict:
    char_id = _str(o, "character_id")
    if not CHAR_ID_RE.match(char_id):
        raise SchemaError(
            f"character_id '{char_id}' must be one lowercase a-z word "
            "(it becomes a LoRA trigger token)")
    gender = _str(o, "gender").lower()
    if gender not in ("male", "female"):
        raise SchemaError(f"genger must be 'male' or 'female' (got '{gender}')")
    cap = _str(o, "lora_caption")
    if _TAG_RE.search(cap):
        raise SchemaError("lora_caption must not contain score_/rating_/nsfw tags")
    # Identity is face/hair/eyes/skin/build/marks ONLY. Clothing lives in the
    # ref groups: if a garment word got into the caption, the LoRA would bake
    # that outfit into the face and every panel would lose the person.
    hit = _CLOTHING_RE.search(cap)
    if hit:
        raise SchemaError(
            f"lora_caption must be CLOTHES-FREE (identity only: face, hair, eyes, "
            f"skin, build, marks) — '{hit.group(0)}' is clothing and belongs in "
            f"the ref groups, not in the string the LoRA learns")
    if len(cap.split()) < 10:
        raise SchemaError(
            "lora_caption should be ~15-22 words covering: hair style+color, eye "
            "color+shape, face/head shape, skin, build, 1-2 distinctive marks")
    # Short IDENTIFIER entries (hair, eye color, build, one more mark). The
    # trigger name is the main identity (the LoRA learns it); entries are
    # hints used in VARYING subsets — one per panel (rotated by panel number)
    # and a rotated pair in each ref caption — never the whole set at once,
    # so the model never expects a fixed full set. Each entry must stay
    # short enough that `name + entry` fits under 7 tokens
    # (comgen.tokens.check_identifiers, enforced in stage 1).
    raw_anchors = o.get("anchors")
    if isinstance(raw_anchors, str):
        raw_anchors = [s for s in (x.strip() for x in raw_anchors.split(",")) if s]
    if not isinstance(raw_anchors, (list, tuple)) or not raw_anchors:
        raise SchemaError(
            "anchors must be a list of 3-5 short identifier phrases "
            '(e.g. ["platinum hair", "amber eyes", "slim"])')
    anchors, seen = [], set()
    for a in raw_anchors:
        a = str(a).lower().strip()
        if not (1 <= len(a.split()) <= 3):
            raise SchemaError(f"anchor entry '{a}' must be 1-3 words only")
        if _TAG_RE.search(a):
            raise SchemaError(f"anchor entry '{a}' must not contain score_/rating_/nsfw tags")
        hit = _CLOTHING_RE.search(a)
        if hit:
            raise SchemaError(
                f"anchor entry '{a}' must be CLOTHES-FREE — '{hit.group(0)}' is clothing")
        if a in seen:
            raise SchemaError(f"anchor entries must be distinct ('{a}' appears twice)")
        seen.add(a)
        anchors.append(a)
    if not (3 <= len(anchors) <= 5):
        raise SchemaError(
            "anchors must be 3-5 distinct short entries "
            "(hair, eye color, build, at most one more mark)")
    return {
        "character_id": char_id,
        "display_name": _str(o, "display_name", char_id),
        "gender": gender,
        "lora_caption": cap,
        "anchors": anchors,
        "differentiator": _str(o, "differentiator", ""),
    }


# --- ref plan: 3 clothing-state groups x 8 framing slots = 24 refs/char ---
#
# The 8 framing slots are FIXED in code (guaranteed coverage of close-ups +
# full shots front/back/side); the LLM only adds body language per slot.
REF_FRAMINGS = (
    "face close-up, frontal (expression detail)",
    "waist-up, three-quarter view",
    "full body front, head to feet",
    "full body back, showing the back",
    "full body side, profile or side three-quarter",
    "waist-up from the side, profile",
    "full body, low angle from below",
    "full body, high angle from above",
)

# The 3 clothing states, in canonical order. Refs are fully clothed;
# all rating_mature (fixed in code, never LLM free text).
REF_STATES = (
    ("signature", "the character's signature outfit for this story role", "rating_mature"),
    ("casual", "a second, clearly different casual outfit (different style and colors)", "rating_mature"),
    ("sporty", "a third, clearly different outfit (sporty or work style)", "rating_mature"),
)


def clean_ref_group(idx: int, o) -> dict:
    """Validate ONE ref group (clothing phrase + 8 poses) from a single-phrase
    LLM call. Refs are fully clothed; the rating is fixed, not LLM text."""
    state_name = REF_STATES[idx][0]
    if not isinstance(o, dict):
        raise SchemaError(f"ref group '{state_name}' must be an object with "
                          "'clothing' and 'poses'")
    # collapse internal whitespace so "No  clothing." == "no clothing"
    clothing = re.sub(r"\s+", " ",
                      str(o.get("clothing", "")).strip().lower().strip(" .;")).strip()
    if not clothing:
        raise SchemaError(f"ref group '{state_name}': 'clothing' must be a non-empty tag phrase")
    if _TAG_RE.search(clothing):
        raise SchemaError(f"ref group '{state_name}': clothing must not contain "
                          "score_/rating_/nsfw (the pipeline assigns the rating)")
    if not _CLOTHING_RE.search(clothing):
        raise SchemaError(f"ref group '{state_name}': clothing must name actual garments "
                          f"(colors + items), got '{clothing}'")
    if clothing == "no clothing":
        raise SchemaError(f"ref group '{state_name}': refs must be fully clothed — "
                          "name actual garments (colors + items)")
    poses_raw = o.get("poses")
    if not isinstance(poses_raw, list):
        raise SchemaError(f"ref group '{state_name}': 'poses' must be a list of 8 phrases")
    poses = [re.sub(r"\s+", " ", str(x).strip().lower()) for x in poses_raw]
    poses = [p.rstrip(" .;") for p in poses if p.strip()]
    if len(poses) != 8:
        raise SchemaError(f"ref group '{state_name}': need EXACTLY 8 poses (got {len(poses)})")
    if len(set(poses)) != 8:
        raise SchemaError(f"ref group '{state_name}': the 8 poses must all be different")
    # Framing coverage (the fixed slot mix the user asked for): across the 8,
    # we need close-ups AND full shots, with front / back / side all covered.
    norm = " ".join(poses).replace("close up", "close-up")
    missing = []
    if "close-up" not in norm:
        missing.append("a 'close-up'")
    if "front" not in norm:
        missing.append("a 'front' pose")
    if "back" not in norm:
        missing.append("a 'back' pose")
    if "side" not in norm and "profile" not in norm:
        missing.append("a 'side'/'profile' pose")
    if norm.count("full body") < 2 and norm.count("full-body") < 2:
        missing.append("at least 2 'full body' poses")
    if missing:
        raise SchemaError(f"ref group '{state_name}': the 8 POSes together must cover "
                          f"{', '.join(missing)} — the slot list in the prompt requires "
                          f"close-ups + full shots (front, back, side)")
    for p in poses:
        if not (2 <= len(p.split()) <= 12):
            raise SchemaError(f"ref group '{state_name}': pose '{p}' must be 2-12 words")
        if _TAG_RE.search(p):
            raise SchemaError(f"ref group '{state_name}': pose '{p}' must not contain score_/rating_ words")
        hit = _CLOTHING_RE.search(p)
        if hit:
            raise SchemaError(f"ref group '{state_name}': pose '{p}' must be CLOTHING-FREE "
                              f"('{hit.group(0)}' is handled by the clothing field)")
        bad = ("he", "him", "his", "she", "her", "hers", "couple", "another person")
        if any(re.search(rf"\b{w}\b", p) for w in bad):
            raise SchemaError(f"ref group '{state_name}': pose '{p}' must be a SOLO "
                              "character (no other people or pronouns)")
    return {"clothing": clothing, "poses": poses}


def add_ref_plan(char: dict, groups: list) -> dict:
    """Attach the 3 validated ref groups + the derived 24 ref_views (write-once
    artifacts; downstream consumes char['ref_views'] as plain strings).
    All states are rating_mature (fully clothed), so no suffix is added."""
    if len(groups) != len(REF_STATES):
        raise SchemaError(f"ref plan needs exactly {len(REF_STATES)} outfit groups ")
    plan, views = [], []
    for g, (_name, _text, rating) in zip(groups, REF_STATES):
        for pose in g["poses"]:
            views.append(f"{pose}, {g['clothing']}")
        plan.append({"clothing": g["clothing"], "rating": rating, "poses": g["poses"]})
    out = dict(char)
    out["ref_plan"] = plan
    out["ref_views"] = views
    return out


def validate_assets(data) -> dict:
    if isinstance(data, list):
        raise SchemaError(
            "top level must be an object with 'characters' and 'locations' keys "
            "(an array was given)")
    if not isinstance(data, dict):
        raise SchemaError("assets must be a JSON object")
    chars = data.get("characters")
    locs = data.get("locations")
    if not isinstance(chars, list) or not (1 <= len(chars) <= 2):
        raise SchemaError("need 1-2 entries in 'characters' (story must cast to two people)")
    if not isinstance(locs, list) or not (1 <= len(locs) <= 3):
        raise SchemaError("need 1-3 entries in 'locations'")
    ids = {c.get("character_id") for c in chars if isinstance(c, dict)}
    if len(ids) != len(chars):
        raise SchemaError("character_id values must be unique")
    characters = [_clean_character(c, i) for i, c in enumerate(chars)]
    loc_ids = [l.get("location_id") for l in locs if isinstance(l, dict)]
    if len(loc_ids) != len(locs) or len(set(loc_ids)) != len(locs):
        raise SchemaError("location_id values must be unique")
    locations = []
    for l in locs:
        lid = _str(l, "location_id").lower()
        if not LOC_ID_RE.match(lid):
            raise SchemaError(f"location_id '{lid}' must be lowercase letters/digits/_ only")
        locations.append({
            "location_id": lid,
            "name": _str(l, "name", lid),
            "description_tags": _str(l, "description_tags"),
        })
    return {
        "story": data.get("story", "") if isinstance(data.get("story"), str) else "",
        "characters": characters,
        "locations": locations,
    }


def _clean_panel(o: dict, idx: int, char_ids: set, loc_ids: set) -> dict:
    num = o.get("panel_num")
    if not isinstance(num, int) or not (1 <= num <= 5):
        raise SchemaError(f"panel {idx}: panel_num must be an int 1-5")
    orient = _str(o, "orientation").lower()
    if orient not in ("portrait", "landscape"):
        raise SchemaError(f"panel {num}: orientation must be portrait|landscape")
    chars = o.get("characters", [])
    if not isinstance(chars, list) or len(chars) > 2:
        raise SchemaError(f"panel {num}: 'characters' must be a list of 0-2 entries")
    seen = set()
    clean_chars = []
    for c in chars:
        cid = _str(c, "character_id").lower()
        if cid not in char_ids:
            raise SchemaError(f"panel {num}: unknown character_id '{cid}' "
                              f"(approved: {sorted(char_ids)})")
        if cid in seen:
            raise SchemaError(f"panel {num}: character '{cid}' listed twice")
        seen.add(cid)
        clean_chars.append({
            "character_id": cid,
            "action": _str(c, "action"),
            "expression": _str(c, "expression"),
        })
    lid = _str(o, "location_id").lower()
    if lid not in loc_ids:
        raise SchemaError(f"panel {num}: unknown location_id '{lid}' "
                          f"(approved: {sorted(loc_ids)})")
    rating = _str(o, "rating").lower()
    if rating not in RATINGS:
        raise SchemaError(f"panel {num}: rating must be one of {list(RATINGS)}")
    return {
        "panel_num": num,
        "orientation": orient,
        "shot": _str(o, "shot"),
        "camera": _str(o, "camera"),
        "mood": _str(o, "mood"),
        "location_id": lid,
        "characters": clean_chars,
        "fx": o.get("fx", "") if isinstance(o.get("fx"), str) else "",
        "rating": rating,
    }


def validate_panels(data, assets: dict) -> list:
    if not isinstance(data, list):
        raise SchemaError("panel_spec must be a JSON array of 5 objects")
    if len(data) != 5:
        raise SchemaError(f"need EXACTLY 5 panels (got {len(data)})")
    char_ids = {c["character_id"] for c in assets["characters"]}
    loc_ids = {l["location_id"] for l in assets["locations"]}
    panels = [_clean_panel(p, i, char_ids, loc_ids) for i, p in enumerate(data)]
    nums = sorted(p["panel_num"] for p in panels)
    if nums != [1, 2, 3, 4, 5]:
        raise SchemaError(f"panel_num must be 1..5 exactly once (got {nums})")
    return sorted(panels, key=lambda p: p["panel_num"])
