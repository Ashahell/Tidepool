# MARCH anchors: pre-registered kill criteria and box

> Source: review prescription, written BEFORE the first anchor run
> Collected: 2026-09-20
> Published: 2026-09-20

## Attempt

Minimal MARCH-style retrieval on the frozen RTU trunk: periodic
top-layer state checkpoints (every k steps) into a capped anchor
bank with learned content-conditioned keys, softmax routing
(incl. learned null), readout fused into the decoder input. Trunk
frozen (encoder + RTU layers); anchor keys/router/decoder train.
Online single-sample byte regime kept.

## Kill criteria (binding, checked in order)

1. Addressing: payload-in-top4 on fresh copy4/after8 episodes
   significantly above chance (~0.31) after ≤3k episodes.
2. Exact recall: non-zero exact copy recall at ≥512 intervening
   bytes on trusted probes.
3. Anchor causality: zeroing/shuffling anchors must hurt (model
   uses them, not the trunk alone).
Fail (1) → kill immediately, no extension. Pass (1), fail (2)/(3)
→ one extension max (dose to 11k episodes), then kill. Compute box:
discriminator ~5 min; extension ~20 min. No further spend.

## Why this attempt is allowed

It is a different architecture class (external addressed store,
trunk as feature extractor), not another trick on the single
evolving state. The retired hypothesis stays retired regardless
of outcome.
