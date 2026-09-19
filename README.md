# Tidepool

Attempted AI Anti Alzheimer.

A PyTorch research harness for a tiny byte-level recurrent language model
that learns continually from a stream — plus the meta-loop that searches
for better ways to train it.

The model reads one byte at a time into recurrent state (no context
window, no tokenizer), predicts the next byte, and keeps learning online.
Around it sits a Dream-RSI-style outer loop: discovery trees of training
runs, exact replay simulation over history, and LLM-driven improvement of
the exploration policy itself.

Status is honest, not flashy: the port is proven against the reference
equations, training is stable and monitored through 6M steps, and a stack
of negative results (replay ×3, EMA) is recorded alongside the wins.
Memory probes still read zero — long-range retention is unproven. See
`wiki/` for the full record, numbers included.

## Quickstart

```bash
uv venv .venv
uv pip install --python .venv/bin/python torch numpy safetensors pyyaml pytest matplotlib
# put wiki_* text files under data/mycorpus/, then split:
.venv/bin/python scripts/make_splits.py --src data/mycorpus --train data/train --val data/val
# train (tiny config), with periodic held-out eval:
.venv/bin/python scripts/train.py --config configs/tiny.yaml --steps 10000 \
    --data data/train --eval-every 2000 --eval-val data/val/wiki_prose
# watch it:
.venv/bin/python scripts/plot_curves.py  # -> runs/curves.png
# tests:
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

## Layout

- `src/tmt/` — model (recurrent byte LM + RTRL traces), training engine, evaluation suite, data helpers
- `src/rsi/` — outer loop (discovery tree, replay simulator, sandbox, rewriters, search driver)
- `scripts/` — train / chat / eval / baselines / plotting / search CLIs
- `configs/` — model + search configs (`tiny.yaml`, `small*.yaml`, `rsi_smoke.yaml`)
- `tests/` — 50+ tests, including finite-difference gradient proofs and memory-oracle gates
- `wiki/` + `raw/` — Karpathy-style knowledge base: every claim links to its source record

## Sources of inspiration

- [Test-Model-Thing](https://github.com/jrz97619761/test-model-thing) by jrz97619761 — the original MLX proof-of-concept this ports: byte I/O, latent prediction, recurrent trace units, test-time training. A solo high-school build; the architecture under study here.
- [Dream-RSI](https://github.com/zhengkid/Dream-RSI) (Google / DeepMind / UMD / UVA) — history-as-replay-simulator for recursive self-improvement at the exploration layer; the design behind `src/rsi/`.
- [Andrej Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — the LLM-maintained wiki pattern (`wiki/` + `raw/` + evidence lint) and the "make the algorithm convince you" engineering ethos.

## License

MIT — see LICENSE.
