"""File helpers with the never-overwrite rule (checklist 3.18).

Generated artifacts are expensive (LLM calls, generations, trains):
if a file already exists in the run folder, we refuse to clobber it.
"""
from __future__ import annotations

import json
from pathlib import Path


class OverwriteRefused(RuntimeError):
    """Raised when a write would clobber an existing artifact."""


def read_json(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_json_once(path: Path, data: object) -> None:
    path = Path(path)
    if path.exists():
        raise OverwriteRefused(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text_once(path: Path, text: str) -> None:
    path = Path(path)
    if path.exists():
        raise OverwriteRefused(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
