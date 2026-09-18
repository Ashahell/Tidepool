# Vulkan4Aros Training Data

> Sources: tmt-torch session record, 2026-09-18
> Raw: [2026-09-18-vk4a-training](../../raw/tmt-torch/2026-09-18-vk4a-training.md)
> Updated: 2026-09-18

## Overview

Training uses Vulkan4Aros bytes, not Wikipedia: 3.8 MB of llm-wiki prose
plus ~1.4 MB of ICD C sources under gitignored data/vk4a/ (files named
wiki_* for the existing loader). First 20k-step run on dim-64/2-layer
moved training loss 6.638 → 2.436 but scored bpb=10.247 (above the 8.0
uniform baseline — miscalibrated, overconfident-wrong), mem=0.000,
cont=0.000, stab=0.992. Machinery proven; model quality is day-zero.

## See Also

- [TMT PyTorch Port](pytorch-port.md)
- [Dream-RSI Loop](rsi-loop.md)
