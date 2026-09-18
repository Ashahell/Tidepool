# Per-domain diagnosis — forgetting confounded by coverage

> Source: tmt-torch eval_memory runs on saved 200k checkpoints + corpus measurement
> Collected: 2026-09-18
> Published: 2026-09-18

## Measured 2026-09-18

Per-domain eval of the 200k checkpoints (eval_memory smoke preset):
off-200k prose 16.926 vs code 11.869; ema-200k prose 20.629 vs code
14.759. Both domains degraded, prose worse in both.

Corpus: data/vk4a_train/wiki_code 1413794 bytes, wiki_prose 3787024
bytes. 200000 steps cover ~200 KB = 4.0 % of the corpus, all inside
wiki_code (sorted first). No 200k-run model ever saw prose in training.

Reading: the long-run curves confound forgetting with single-pass
local overfitting — the stream drifts (code interior) while eval sits
fixed (prose tail). Early good prose bpb came from overlap, not
knowledge; its decay is drift-plus-miscalibration, not proven loss of
learned prose. All forgetting conclusions to date need re-measurement
under multi-epoch shuffled coverage before any cure is judged. The
correct next step is epoch training, not another stabilizer.
