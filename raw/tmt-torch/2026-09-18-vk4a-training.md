# First real training on Vulkan4Aros bytes — run record

> Source: tmt-torch training runs runs/train_probe + runs/train_vk4a (checkpoints gitignored; loss.csv excerpt below)
> Collected: 2026-09-18
> Published: 2026-09-18

## Corpus built 2026-09-18

Simple Wikipedia dump idea dropped (356 MB download deleted before
extraction finished). Corpus is Vulkan4Aros only, under gitignored
data/vk4a/: wiki_prose (3.8 MB: llm-wiki log.md + index.md + all
raw/articles/*.md concatenated) and wiki_code (~1.4 MB: all
src/vulkan/library/*.c and *.h concatenated, capped at 2000000 bytes).
train.py reads files matching wiki_* via iter_wikipedia_bytes --data.

## Runs measured 2026-09-18

Config configs/tiny.yaml: dim 64, layers 2, lr 0.0005, update_every 1,
decay_groups 4, seed 42. Probe: 300 steps in 3.736 s wall (~80
steps/s), losses 9.57589340209961, 8.582010269165039,
8.713501930236816, 9.878942489624023 for steps 1-4. Main run: 20000
steps, exit 0; loss.csv first500 mean 6.638, last500 mean 2.436, min
0.03 (min is a single-step outlier, likely a short-line stop term).

Eval of the trained checkpoint (scripts/eval_memory.py, smoke preset,
4096-byte val slice of the same prose): bpb=10.247 mem=0.000
cont=0.000 stab=0.992 composite=-119.803 fail=None. Bits-per-byte above
the 8.0 uniform baseline: the model is miscalibrated after 20k online
steps (overconfident-wrong), not a capable predictor. Memory and
continual probes score 0. This is a machinery baseline, not a result.
