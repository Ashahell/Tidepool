# Slots v2 copy retention experiment (2026-09-20)

## Setup
- Base: `configs/copy_selective.yaml` (dim 128, layers 4, lr 1e-4, selective) — verified against `configs/small_lr1e4.yaml` (dim/layers/lr identical).
- Slots arm: `configs/copy_slots.yaml` (= base + `slots: 16`, `aux_mem_w: 1.0`), reused (file predates branch).
- Control arm: base unchanged (slots off).
- Both arms: 30000 steps pure-copy (`--copy-frac 1.0`, `--epochs 1`, data `data/vk4a_train`), trained under v2 HEAD `9c421e7` (shared keys + temperature + usage protection).
  - slots_on ckpt: `runs/slots_on_v2/model.safetensors` (final loss 7.119, CE 5.516)
  - slots_off ckpt: `runs/slots_off_v2/model.safetensors` (final loss 7.757, CE 6.090; byte-identical to the pre-v2 control — expected, v2 touches only the slots path)
- NOTE: an earlier `2026-09-20-slots-copy.md` version of this record scored checkpoints trained under pre-v2 code (`b14485b`); those runs are superseded by the `_v2` runs above. Scores were 0.0/0.0 in both cases.

## Scores (ingesting copy probe, seed 7, 10 trials/cell)
`EvalConfig(copy_lengths=[4,8], intervening_lengths=[8,32], num_copy_trials=10)`

| arm | overall | copy4_after_8 | copy4_after_32 | copy8_after_8 | copy8_after_32 |
|-----|---------|---------------|----------------|---------------|----------------|
| slots_on (v2) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| slots_off | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Interpretation
Pure-copy training alone teaches neither arm to retain-and-reproduce across a gap at this scale/schedule; slots v2 addressing is unproven either way, not disproven. No outcome gate per brief: 0.0 is a finding. The v2 change moves no needle on this probe — consistent with the probe demanding exact byte reproduction while training optimises next-byte CE on copy episodes with filler gaps (credit assignment across the gap remains the bottleneck, not addressing sharpness).

## Open questions
- Does any arm learn the copy task at all (train-loss / near-gap sanity), or is 30k steps pure-copy insufficient?
- Does the aux retrieval loss move during training (is the read path getting gradient)?
- Would cued/gated variants (selective copy) or longer schedules separate the arms?
