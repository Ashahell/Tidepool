# Decision: open the external-store line

> Source: review directive + author decision, 2026-09-20
> Collected: 2026-09-20
> Published: 2026-09-20

## Decision (one paragraph)

We open the external-store line under the same proof standards.
Both recurrent-state hypotheses (trunk state, state anchors) are
dead and stay dead; the only direction with prior-art support
(HAM, MARCH, Gated DeltaNet-2) and no falsifying probe is a true
external key-value memory beside a frozen RTU trunk used as feature
extractor. This is a new, narrow scientific question — whether a
real external store supplies the pointing this family lacked — not
a continuation of the old hope.

## Hypothesis

A frozen RTU trunk + a true external key-value memory (learned
write/erase/protect + content addressing + addressing supervision)
produces non-zero exact recall and above-chance addressing under the
existing online single-sample byte-level regime.

## What makes it external (not a renamed slot)

Separate store matrix owned by nothing but the memory module;
entries written by a learned write gate, erasable only through a
learned erase gate, with a protect bit the write path must respect;
reads strictly by content-addressed query/key match (no learned
position logits, no uniform fallbacks); addressing loss from the
first run. Slots v1/v2 and anchors failed exactly these points.

## Kill criteria (binding, in order)

1. Addressing (payload-in-top4, chance ~0.31) significantly above
   chance after ≤3k episodes, REPLICATED on a second seed.
2. Non-zero exact copy recall at ≥512 intervening bytes.
3. Store causality: zeroing/shuffling the external store hurts.
One non-replicating seed is not enough; require replication or kill.
Fail (1) → kill immediately. Pass (1), fail (2)/(3) → one
dose-matched extension, then kill. Box: discriminator ~5 min,
extension ~20 min.

## Will not

Re-open trunk state, anchors, slots-as-cache, or any retired line;
scale first; soften probes; aim the RSI loop at the new store
before it passes the gate. If the external store fails these
criteria, the conclusion is that online byte-level recurrent models
aiming at long-term addressable memory are not viable under the
tested conditions, and the project archives with the evidence.
