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

QUERY_MARKER = b"\x00\x00\x00"
PAYLOAD_LENS = (4, 8, 16)
FILLER_LENS = (8, 32, 128)
TINY_PAYLOAD_LENS = (2, 4)
TINY_FILLER_LENS = (0, 2, 4)

def copy_episode(rng: random.Random,
                 payload_lens=PAYLOAD_LENS,
                 filler_lens=FILLER_LENS) -> bytes:
    """STORE payload + filler + MARKER + payload(target).

    Reset the model before feeding. Standard next-byte CE on the trailing
    payload segment is the recall signal — no extra loss term.
    """
    plen = rng.choice(payload_lens)
    flen = rng.choice(filler_lens)
    payload = bytes(rng.randint(32, 126) for _ in range(plen))
    filler = bytes(rng.randint(0, 255) for _ in range(flen))
    return payload + filler + QUERY_MARKER + payload

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
