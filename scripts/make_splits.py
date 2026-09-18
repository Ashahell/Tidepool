# scripts/make_splits.py
"""Split a wiki_* corpus into disjoint train/val roots (val takes line tails)."""
from __future__ import annotations
import argparse
import sys
sys.path.insert(0, "src")
from tmt.data import split_corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/vk4a")
    ap.add_argument("--train", default="data/vk4a_train")
    ap.add_argument("--val", default="data/vk4a_val")
    ap.add_argument("--frac", type=float, default=0.05)
    args = ap.parse_args()
    split_corpus(args.src, args.train, args.val, args.frac)
    print(f"split {args.src} -> {args.train} + {args.val} (val_frac={args.frac})")


if __name__ == "__main__":
    main()
