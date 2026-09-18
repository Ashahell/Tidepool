# src/tmt/data.py
from __future__ import annotations
from pathlib import Path
from typing import Iterator

def iter_wikipedia_bytes(root: str = "wikipedia_clean") -> Iterator[bytes]:
    for p in sorted(Path(root).rglob("wiki_*")):
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                b = line.encode("utf-8", errors="ignore")
                if b:
                    yield b

def load_val_bytes(path: str, limit: int = 20000) -> bytes:
    data = Path(path).read_bytes()[:limit]
    return data
