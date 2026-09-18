# Vulkan4Aros Training Data

> Sources: tmt-torch session record, 2026-09-18
> Raw: [2026-09-18-vk4a-training](../../raw/tmt-torch/2026-09-18-vk4a-training.md); [2026-09-18-monitoring](../../raw/tmt-torch/2026-09-18-monitoring.md); [2026-09-18-serious-run-1](../../raw/tmt-torch/2026-09-18-serious-run-1.md); [2026-09-18-knob-round](../../raw/tmt-torch/2026-09-18-knob-round.md)
> Updated: 2026-09-18

## Overview

Training uses Vulkan4Aros bytes, not Wikipedia: 3.8 MB of llm-wiki prose
plus ~1.4 MB of ICD C sources under gitignored data/vk4a/ (files named
wiki_* for the existing loader). First 20k-step run on dim-64/2-layer
moved training loss 6.638 → 2.436 but scored bpb=10.247 (above the 8.0
uniform baseline — miscalibrated, overconfident-wrong), mem=0.000,
cont=0.000, stab=0.992. Machinery proven; model quality is day-zero.

## Monitoring (Live)

Train/val split is now disjoint (5% line tails). loss.csv carries
per-component columns; the variance term sits at ~0 while CE does the
work. Heartbeat state.json, NaN watchdog with emergency checkpoint,
periodic held-out eval to eval.csv, curves.png plot. First monitored run:
held-out bpb 7.862 → 7.490 over 300 steps (below uniform, improving).
Full suite 37 passed.

## Serious Run 1 (Overfitting Baseline)

dim-128/4-layer, 30k steps in 49 s (610 B/s): training loss fell while
held-out bpb rose 6.660 → 14.301 monotonically after ~2k steps —
memorization plus miscalibration, the exact pathology the RSI loop's
continual-learning search space (replay, LR, update_every,
regularization) exists to fix. Throughput is fine; generalization is the
bottleneck. Curves in runs/curves.png (gitignored).

> **Status: Outdated** (2026-09-18)
> The knob round cured it by hand: lr 1e-4 holds bpb flat-to-improving
> (6.93 → 6.33, all below uniform), update_every 32 also stabilizes
> (~7.4–7.8). The disease was step-size noise, not structural amnesia —
> no architecture search needed. Baseline is now lr 1e-4; the RSI loop
> stays parked until a structural residual appears. Tiny-20k showed one
> late spike to 9.423 at step 20000 (unrepeated, no claim).

## See Also

- [TMT PyTorch Port](pytorch-port.md)
- [Dream-RSI Loop](rsi-loop.md)
