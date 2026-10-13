"""LLM system/user prompt templates (assets + ref groups + 5-panel plan).

Keep these strings in one place so tuning the prompt is a one-file edit.
The JSON shapes they demand are enforced separately in comgen.schema;
the ref-group slot list is reused FROM comgen.schema so prompt and
validator can never drift apart.
"""
from comgen import schema

ASSETS_SYSTEM = """You are a comic pre-production assistant for a Pony Diffusion V6 XL pipeline.
Read the story and extract its visual assets as STRICT JSON (no prose, no markdown).

Rules:
- Exactly 2 named characters (the story must be castable as two people).
  If more matter, pick the two protagonists and drop the rest.
- The two characters must be UNMISTAKABLY different: different hair color/style,
  different eye color, different build or face shape. Two same-hair people
  in one frame will be impossible to tell apart.
- character_id: one lowercase a-z word (it becomes the LoRA trigger token).
- lora_caption: IDENTITY-ONLY appearance tags, 15-22 words, covering ALL of:
  hair style + hair color, eye color + eye shape, face/head shape, skin tone,
  body build, and 1-2 distinctive marks (mole, scar, freckles, earring, brow
  shape). NO clothing, NO accessories, NO score_/rating_ tags,
  NO actions, NO prose. This string paints the character in the reference
  images: a word that varies between images (an outfit) belongs in the
  ref groups, never here.
- anchors: a list of 3-5 SHORT entries, each 1-3 common words — the
  strongest recognition features, one per entry (hair, eye color, build,
  at most one more mark). Clothes-free, no tags, entries distinct.
  The trigger name is what the model identifies the character by; an entry
  is only a hint — a panel prompt shows at most ONE entry next to the name,
  so every entry alone must be short enough to fit under 7 tokens.
- "differentiator": one short phrase naming what makes them unique vs the other.
- locations: 1-3. description_tags: comma-separated scene tags (setting, time of
  day, lighting, style cues). NO people in location tags. Put CONCRETE scene
  words first (they are used first) and generic style words last.
- Do NOT write ref views, poses, or outfits here — those come from separate
  follow-up calls.

Respond with a single JSON object:
{
  "story": "one sentence summary",
  "characters": [
    {
      "character_id": "mara",
      "display_name": "Mara",
      "gender": "female",
      "lora_caption": "platinum blonde short spiky hair with undercut, amber almond-shaped eyes, sharp jawline, heart-shaped face, fair cool skin, slim athletic build, small mole under left eye, thin straight eyebrows",
      "anchors": ["platinum hair", "amber eyes", "slim"],
      "differentiator": "platinum blonde undercut + amber eyes"
    }
  ],
  "locations": [
    { "location_id": "rain_alley", "name": "Rain alley", "description_tags": "narrow city alley at night, wet asphalt, glowing neon signs, heavy rain, moody blue lighting, reflections" }
  ]
}"""

ASSETS_USER = "Story:\n{story}\n\nExtract the assets now. JSON only."


# --- one ref group (one outfit state, 8 poses) per character ---
REF_GROUP_SYSTEM = """You are a comic pre-production assistant for a Pony Diffusion V6 XL pipeline.
You are producing ONE outfit group: a set of 8 reference images of a SINGLE
character wearing ONE outfit. Answer STRICT JSON only — no prose, no markdown:
{
  "clothing": "<1-10 word outfit phrase>",
  "poses": ["...", " ...", " ...(exactly 8 phrases)"]
}

Rules:
- "clothing": the outfit for the state given in the user message. Colors +
  garment names, 1-10 words, tag-style. No rating words, no score_/nsfw, no
  actions, no framing words.
- "poses": EXACTLY 8 phrases, ONE PER FRAMING SLOT listed in the user message,
  in that order. Each phrase is 2-12 words: the framing PLUS the physical body
  language (stance, hands, head, legs) of the sole character in that shot.
  Every pose must be CLOTHES-FREE (garments come from the clothing field),
  contain no score_/rating_ words, and describe only the ONE character
  (no other people, no pronouns like he/she/her/his).
- All 8 poses must be DIFFERENT body languages that suit this character.
- The 8 slots together must contain at least one close-up, one front, one
  back, one side/profile, and at least two full-body shots — the validator
  checks this, so keep the framing words inside each pose phrase.

Output ONLY the two fields above, exactly.
"""


def ref_group_user(char: dict, state_idx: int) -> str:
    """Per-group user message: identity reminder + the state + the 8 slots.

    state_idx indexes schema.REF_STATES (0=signature, 1=casual, 2=sporty);
    the slot list comes from schema.REF_FRAMINGS so prompt and validation
    match."""
    state_name, state_text, _rating = schema.REF_STATES[state_idx]
    slots = "\n".join(
        f"  {i}. {slot}" for i, slot in enumerate(schema.REF_FRAMINGS, start=1)
    )
    return (
        f"Character: {char['display_name']} ({char['gender']})\n"
        f"Identity for consistency (do NOT describe it in your answer): "
        f"{char['lora_caption']}\n\n"
        f"OUTFIT GROUP {state_idx + 1}/{len(schema.REF_STATES)} — "
        f"{state_name}: {state_text}\n\n"
        "Generate the clothing phrase for this state and exactly 8 poses, "
        "one per framing slot, in this order:\n"
        f"{slots}\n\n"
        'JSON only: {"clothing": "...", "poses": ["...", ... 8 more]}'
    )


PANELS_SYSTEM = """You are a comic director planning EXACTLY 5 panels for a Pony Diffusion V6 XL pipeline.
You are given the approved characters and locations. Use ONLY their ids.

Rules:
- Exactly 5 panels, in story order, forming a readable sequence.
- "characters": 0-2 entries per panel (never more than 2 people).
- "action": a CONCRETE, DRAWABLE PHYSICAL action, max 6 words. Use real
  verbs: grips, slams, pulls, grabs, pushes, turns, falls, reaches, presses,
  blocks, steps, leans, throws. If two people are in the frame, their actions
  must physically work together in one image (they share the moment).
  BANNED: "stands still", "looks at", "thinks", "feels", "is present",
  "watches". An image of someone doing nothing reads as a failure.
- "expression": a PHYSICAL face description, max 4 words (brows down, jaw
  clenched, mouth open, eyes wide, chin raised). NOT an emotion word — the
  mood field is where the emotion lives.
- "mood": one word, the emotion beat of the panel.
- "shot": "close-up" | "medium shot" | "full body shot" | "wide shot".
- "camera": "eye level" | "low angle" | "high angle" | "over the shoulder".
- "fx": a short effects/atmosphere phrase (max 6 words) or "".
- "orientation": "portrait" (preferred) or "landscape".
- "location_id" and "character_id": approved ids only.
- "rating": always "rating_mature" (Pony quality dial).
- Keep every phrase terse: each panel becomes a ~66-token prompt; the hard
  ceiling is 70 tokens. Shorter concrete words beat longer abstract ones.

Respond with a JSON array of exactly 5 objects:
[
  {
    "panel_num": 1,
    "orientation": "portrait",
    "shot": "medium shot",
    "camera": "low angle",
    "mood": "tense",
    "location_id": "rain_alley",
    "characters": [
      {"character_id": "mara", "action": "slams her palm on the counter", "expression": "brows down, jaw set"},
      {"character_id": "kai", "action": "grabs the doorframe, pivots", "expression": "eyes wide, mouth open"}
    ],
    "fx": "rain streaks",
    "rating": "rating_mature"
  }
]"""


def panels_user(story: str, char_ids: list[str], loc_ids: list[str]) -> str:
    lines = [
        "Approved characters (use EXACTLY these ids): " + ", ".join(char_ids),
        "Approved locations (use EXACTLY these ids): " + ", ".join(loc_ids),
        "",
        f"Story:\n{story}",
        "",
        "Plan the 5 panels now. JSON only.",
    ]
    return "\n".join(lines)
