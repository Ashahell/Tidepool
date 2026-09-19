# Rank analysis — retention exists but blurry

> Source: tmt-torch cued rank eval on saved copy checkpoints (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

First-payload-byte rank under the cued protocol (30 trials, chance mean
rank 128): d128-30k mean 53.5 (top1 0/30, top5 0/30); d128-150k mean
44.8 (top1 0/30, top5 2/30); d256-30k mean 54.8 (top1 0/30, top5 1/30).

Reading: retention is NOT absent — payload ranks far above chance, and
150k beats 30k (44.8 vs 53.5), so it improves with training. But it is
blurry: rank ~50, never top-1. Argmax-exact probes demand precision the
trace does not have. The quest reframes from "create memory" to
"sharpen the trace": capacity, training length, readout calibration, or
contrastive sharpening. The zero-probe verdicts stand (exact recall is
absent) but no longer imply nothing is stored.
