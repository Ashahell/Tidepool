# src/tmt/data.py
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import random

def iter_wikipedia_bytes(root: str = "wikipedia_clean") -> Iterator[bytes]:
    for p in sorted(Path(root).rglob("wiki_*")):
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                b = line.encode("utf-8", errors="ignore")
                if b:
                    yield b

def skip_bytes(root: str = "wikipedia_clean", n: int = 0) -> Iterator[bytes]:
    """Yield the byte stream of iter_wikipedia_bytes one byte at a time,
    dropping the first n bytes. Offset-counted across chunks, O(1) memory."""
    skip = max(0, int(n))
    for chunk in iter_wikipedia_bytes(root):
        if skip >= len(chunk):
            skip -= len(chunk)
            continue
        start = skip
        skip = 0
        for i in range(start, len(chunk)):
            yield chunk[i:i + 1]

def load_val_bytes(path: str, limit: int = 20000) -> bytes:
    data = Path(path).read_bytes()[:limit]
    return data

def epoch_lines(root: str, epoch: int, seed: int) -> list:
    """All non-empty lines under root, shuffled deterministically per epoch."""
    lines = []
    for p in sorted(Path(root).rglob("wiki_*")):
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            lines.extend(l for l in f if l.encode("utf-8", errors="ignore"))
    rng = random.Random(seed + epoch)
    rng.shuffle(lines)
    return lines

def split_corpus(src_root: str, dst_train: str, dst_val: str,
                 val_frac: float = 0.05) -> None:
    """Split each wiki_* file into train/val by lines; val takes the tail."""
    for p in sorted(Path(src_root).rglob("wiki_*")):
        if not p.is_file():
            continue
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        n_val = max(1, int(len(lines) * val_frac)) if lines else 0
        tr, va = lines[: len(lines) - n_val], lines[len(lines) - n_val:]
        for text, dst in ((tr, dst_train), (va, dst_val)):
            out = Path(dst) / p.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(text), encoding="utf-8")
