# Slots copy retention experiment (2026-09-20)

## Setup
- Base: `configs/copy_selective.yaml` (dim 128, layers 4, lr 1e-4, selective) — verified against `configs/small_lr1e4.yaml` (dim/layers/lr identical).
- Slots arm: `configs/copy_slots.yaml` (= base + `slots: 16`, `aux_mem_w: 1.0`), committed.
- Control arm: base unchanged (slots off).
- Both arms: 30000 steps pure-copy (`--copy-frac 1.0`, `--epochs 1`, data `data/vk4a_train`).
  - slots_on ckpt: `runs/slots_on/model.safetensors`
  - slots_off ckpt: `runs/slots_off/model.safetensors`

## Scores (ingesting copy probe, seed 7, 10 trials/cell)
`EvalConfig(copy_lengths=[4,8], intervening_lengths=[8,32], num_copy_trials=10)`

| arm | overall | copy4_after_8 | copy4_after_32 | copy8_after_8 | copy8_after_32 |
|-----|---------|---------------|----------------|---------------|----------------|
| slots_on | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| slots_off | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Interpretation
Pure-copy training alone teaches neither arm to retain-and-reproduce across a gap at this scale/schedule; slots addressing is unproven either way, not disproven.

## Open questions
- Does any arm learn the copy task at all (train-loss / near-gap sanity), or is 30k steps pure-copy insufficient?
- Does the aux retrieval loss move during training (is the read path getting gradient)?
- Would cued/gated variants (selective copy) or longer schedules separate the arms?
