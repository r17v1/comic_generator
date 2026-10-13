"""Append-only run logger: prints and appends to <run_dir>/run.log."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


class RunLog:
    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "run.log"

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def __call__(self, msg: str) -> None:
        self.log(msg)
