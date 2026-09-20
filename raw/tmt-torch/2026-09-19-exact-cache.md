# Exact-match cache — wrong shape for recall

> Source: tmt-torch ExactCache module + evals on rtrl200k ckpt (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

ExactCache (bounded K-gram table, LRU, interpolate λ): unit tests 4/4
green. Copy probe with K=4: 0/20. Prose bpb with cache 5.759 vs plain
5.750 (neutral). First attempt with K=16 recorded nothing (feed shorter
than K — configuration error, caught immediately).

Reading: exact matching fails by construction at the query boundary —
the end-of-feed context was never observed continued, so generation
opens with a guaranteed miss and never recovers. kNN-LM works through
SIMILARITY retrieval, not exact match; exactness is the wrong shape for
recall-from-scratch (it only continues verbatim repetitions). bpb
neutral confirms no prediction value either at this scale. Next honest
design: exact STORAGE with soft similarity RETRIEVAL (verbatim keys,
softmax read) — preserves content like exact, generalizes like soft.
