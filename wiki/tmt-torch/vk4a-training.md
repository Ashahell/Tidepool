# Vulkan4Aros Training Data

> Sources: tmt-torch session record, 2026-09-18
> Raw: [2026-09-18-vk4a-training](../../raw/tmt-torch/2026-09-18-vk4a-training.md); [2026-09-18-monitoring](../../raw/tmt-torch/2026-09-18-monitoring.md); [2026-09-18-serious-run-1](../../raw/tmt-torch/2026-09-18-serious-run-1.md); [2026-09-18-knob-round](../../raw/tmt-torch/2026-09-18-knob-round.md); [2026-09-18-long-run](../../raw/tmt-torch/2026-09-18-long-run.md); [2026-09-18-replay-delivery](../../raw/tmt-torch/2026-09-18-replay-delivery.md); [2026-09-18-replay-comparison](../../raw/tmt-torch/2026-09-18-replay-comparison.md); [2026-09-18-priority-delivery](../../raw/tmt-torch/2026-09-18-priority-delivery.md); [2026-09-18-priority-comparison](../../raw/tmt-torch/2026-09-18-priority-comparison.md); [2026-09-18-noise-control](../../raw/tmt-torch/2026-09-18-noise-control.md); [2026-09-18-ema-delivery](../../raw/tmt-torch/2026-09-18-ema-delivery.md); [2026-09-18-ema-comparison](../../raw/tmt-torch/2026-09-18-ema-comparison.md); [2026-09-18-coverage-confound](../../raw/tmt-torch/2026-09-18-coverage-confound.md); [2026-09-18-reset-interim](../../raw/tmt-torch/2026-09-18-reset-interim.md); [2026-09-18-reset-extension](../../raw/tmt-torch/2026-09-18-reset-extension.md); [2026-09-18-reset-completion](../../raw/tmt-torch/2026-09-18-reset-completion.md); [2026-09-18-rtrl-200k](../../raw/tmt-torch/2026-09-18-rtrl-200k.md); [2026-09-18-memory-zero](../../raw/tmt-torch/2026-09-18-memory-zero.md); [2026-09-18-copy-training](../../raw/tmt-torch/2026-09-18-copy-training.md); [2026-09-18-pure-copy](../../raw/tmt-torch/2026-09-18-pure-copy.md); [2026-09-18-copy-accum](../../raw/tmt-torch/2026-09-18-copy-accum.md); [2026-09-19-gru-contrast](../../raw/tmt-torch/2026-09-19-gru-contrast.md); [2026-09-19-scale-copy](../../raw/tmt-torch/2026-09-19-scale-copy.md); [2026-09-19-scheduled-sampling](../../raw/tmt-torch/2026-09-19-scheduled-sampling.md); [2026-09-19-single-episode](../../raw/tmt-torch/2026-09-19-single-episode.md); [2026-09-19-copy-1m](../../raw/tmt-torch/2026-09-19-copy-1m.md); [2026-09-19-selective-long](../../raw/tmt-torch/2026-09-19-selective-long.md); [2026-09-19-selective-copy](../../raw/tmt-torch/2026-09-19-selective-copy.md); [2026-09-19-decay-audit](../../raw/tmt-torch/2026-09-19-decay-audit.md); [2026-09-19-rank-flat](../../raw/tmt-torch/2026-09-19-rank-flat.md); [2026-09-19-per-position](../../raw/tmt-torch/2026-09-19-per-position.md); [2026-09-19-mlp-readout](../../raw/tmt-torch/2026-09-19-mlp-readout.md); [2026-09-19-recall-head](../../raw/tmt-torch/2026-09-19-recall-head.md); [2026-09-19-linear-probe](../../raw/tmt-torch/2026-09-19-linear-probe.md); [2026-09-19-rank-analysis](../../raw/tmt-torch/2026-09-19-rank-analysis.md); [2026-09-19-tiny-cued](../../raw/tmt-torch/2026-09-19-tiny-cued.md); [2026-09-19-long-copy](../../raw/tmt-torch/2026-09-19-long-copy.md)
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

## Long Run (200k): Cure Expires, Loop Un-Parked

lr 1e-4 held bpb flat (6.3535–7.4096) only to ~40k steps, then degraded slowly
to 18.115 by 200k, while train loss kept falling (8.47217845916748 → 5.57870626449585) and
stability stayed ~1.0. Retention died first (continual 0.3218 → 0.0 by
75k). Low LR delayed the disease; it did not cure it. This slow
forgetting-plus-miscalibration is structural — the RSI loop's replay /
regularization search is un-parked.

## Prioritized Replay (Landed, Tested Negative)

Loss-weighted sampling (alpha, default 1.0) with tag refresh and uniform
fallback; live-only monitoring kept; 46/46 green.

## Priority Comparison (Negative Result)

60k-step fair pair: priority ON tracks baseline to ~20k, then diverges
upward from ~25k and ends 8.4 bpb worse (17.012 vs 8.616) — worse than
uniform replay's 3.5-point loss. Replaying high-loss bytes digs into
mistakes (overconfidence on hard/noisy bytes compounds) instead of
smoothing them. Both replay shapes hurt; remaining candidates are
reservoir/diverse sampling, EWC, or a different stabilizer (slower
schedules, weight averaging).

## Noise Control (Content Matters)

Random-byte extra updates (same budget, no buffer content) end at 9.442:
off 8.616 < random 9.442 < uniform 12.119 < priority 17.012. Update
count alone costs ~0.8 bpb; content adds the rest. The effective-LR-only
story is dead — recency bias and dig-into-mistakes survive the control.

## EMA Delivery (Mechanics)

Shadow params tracked per optimizer step (init-on-first, decay blend
under no_grad); using_ema() swaps params only (state/buffers untouched)
and restores; checkpoints persist ema_* keys; train eval auto-uses
averaged weights when present, eval_memory takes --ema. 50/50 green.

## Coverage Confound (Forgetting Unproven)

Per-domain evals: off-200k prose 16.926 vs code 11.869, ema-200k prose
20.629 vs code 14.759 — both degraded, prose worse. But 200k steps cover
4.0 % of the corpus, all inside wiki_code: no long-run model ever saw
prose in training. The decay curves confound forgetting with
single-pass local overfitting under stream drift. All forgetting
conclusions need re-measurement under multi-epoch shuffled coverage;
epoch training is next, not another stabilizer.

## EMA Comparison (Negative Result)

Averaged-weight eval ties baseline at 60k (8.534 vs 8.616) and ends
worse at 200k (20.398 vs 18.115). Averaging noisy weights follows the
same decay. Nothing tried holds past ~40k: uniform replay, priority
replay, EMA all fail; low LR only delays. Untried: reservoir diversity,
EWC, slower schedules, bigger models.

## Replay Comparison (Negative Result)

60k-step fair pair: replay ON (size 512, k 1) never beats OFF at any of
12 checkpoints, ends 3.5 bpb worse (12.119 vs 8.616). Uniform recency
replay re-trains already-learned bytes — recency bias doubled, not
countered. Next candidates: reservoir/diverse buffer, high-loss
prioritization, or EWC.

## Replay Buffer (Landed, Untested Against Forgetting)

In-model recency deque (`replay_size`, `replay_k`), shared accumulation,
live-only monitoring, off-path bit-identical. Merged; 41/41 green.

> **Status: Outdated** (2026-09-18)
> The comparison has run since: uniform replay ends 3.5 bpb worse (see
> Replay Comparison above); prioritized ends 8.4 worse.

## Per-Line Reset (Intervention Confirms Diagnosis)

Epoch mode now resets recurrent state per line (2-line change). Reset
run evals at 20/40/60k: 6.261, 5.367, 6.262 — flat, best sustained
numbers ever — vs no-reset 7.028, 12.694, 17.419 at the same steps.
Stale cross-line state was the poison. Reset is now standard in epoch mode. Full 2-epoch reset curve pending; it decides whether forgetting is gone or merely slower.

## Reset Extension (160k, Still Flat)

Extended curve through 160k steps: 80k 7.601, 100k 6.315, 120k 8.175,
140k 6.786, 160k 6.367 — oscillating 6.3–8.2 with no monotonic decay,
against no-reset 11–18 across the same span and climbing. Forgetting has
not reappeared. (The stopped no-reset run's recovered tail — 25.637 at
200k, 34.073 at 240k, silent harness death at 256500 steps with finite
loss — leaves its post-240k evals a permanent gap.)

## Memory: True Zero (Ingesting Probe)

New probe feeds through model.ingest (oracles order 1.0/0.0/0.0, suite
70 green). The trained 200k RTRL checkpoint scores 0.0 at every cell
including copy-4 after 8 bytes. Good prediction with zero retention:
different phenomena, properly instrumented baseline.

## Copy Training (No Signal Yet)

Episode protocol (triple-null marker, per-episode reset, frac 0.2) with
no new loss term; 20k steps: probe 0.0 everywhere, payload CE 7.382 vs
uniform 5.545 — not even marker recognition. Inconclusive on capability
(a few hundred diluted episodes); needs longer/higher-frac training
before any claim.

## Pure Copy (Cannot Learn Recall at This Scale)

Entire stream copy episodes (frac 1.0), corrected RTRL gradients, 30k
steps: payload CE 7.923, probe 0.0. Dilution, wrong-gradients, and probe
excuses all removed — the architecture as built does not acquire exact
recall at dim128/30k-episodes. Next hypotheses need their own
experiments: capacity, update noise, protocol.

## Copy With Accumulated Updates (Noise Hypothesis Weak)

Same pure-copy conditions at update_every 32: payload CE 5.846 (vs 7.923
at update_every 1), probe still 0.0. 32x accumulation buys 2 CE points
but zero recall — update statistics are not the blocker. Remaining:
capacity (scale up) or protocol (different recall signaling).

## GRU Contrast (Failure Not TMT-Specific)

Same-scale GRU baseline (161152 vs 132993 params, same copy protocol,
30k steps): copy 0.0 and assoc 0.0 for both architectures. Classic
gated recurrence fails identically — exact recall does not emerge from
online single-sample training here regardless of cell type. Next
discriminator is scale, not architecture swaps at 130–160k.

## Scale Copy (Capacity Not the Blocker)

dim256/layers8 (662017 params, 5x) on pure copy, 30k steps: payload CE
7.187 (vs 7.923 dim128), probe still 0.0. Equal-step comparison;
bigger model may want more steps (confound noted). Process fix in the
same pass: run artifacts now land under the ckpt dir (shared loss.csv
collisions bitten three times).

## Selective Copy (Marginal Gain, Still Zero)

Input-dependent gated decay (FD-proven) on pure copy, 30k steps:
payload CE 7.614 vs 7.923 plain, probe 0.0. Gate engaged, early evals
coincide by determinism (direct 6-decimal bpb differs). Inconclusive,
not negative — needs longer/wider conditions. Loader now backfills
missing gate keys with zeros and falls back loudly on incompatible
optimizer state.

## Selective Long (Prediction Improves, Recall Absent)

150k selective steps: payload CE 6.276 (7.614 → 6.276 over 30→150k),
probe still 0.0. Selectivity helps prediction, not recall. The wall
stands across every mechanism, scale, and budget tried.

## Tiny Curriculum + Cued Recall (Protocol Dead)

Payload 2–4, filler 0–4, frac 1.0, 30k steps: all six cells 0.0
including copy-2 after zero gap. Marker-cued eval (matching training):
0/20 at both lengths. Protocol mismatch excuse dead — short, minimal,
matched, still nothing. Left: much longer training, scale, structural
memory.

## Long Copy (Time Doesn't Unlock Recall)

150k pure-copy steps: payload CE 6.494 (7.923 → 7.229 → 6.494 across
30/60/150k — grinding, still above uniform 5.545), probe 0.0,
held-out evals flat ~7.0. Time alone ruled out at this scale. Left:
protocol redesign, much bigger scale, structural memory.

## Single Episode (Main Path Can't Memorize)

One fixed 16-byte episode x 300 reps: output degenerates to a
marker/null cycle, match False. The separate recall head fit 8/8 fixed
— so the split is precise: frozen-state readout can fit, the jointly
trained main path cannot, with zero generalization required. Failure is
in joint recall training through the main head.

## Scheduled Sampling (Perturbs, Doesn't Cure)

Root cause isolated first: fillerless fails identically (filler-noise
dead); teacher-forced CE 0.028 vs free collapse (exposure-bias class).
Fix: --ss-prob own-prediction inputs. 300 reps give payload fragments
(shape improvement, no match); 1000 reps collapse back to the null
cycle. Attractor dynamics dominate free generation; SS alone
insufficient. Next: generation constraints, attractor analysis,
structural decode changes.

## 1M Copy (Prediction Crosses Chance, Recall Zero)

1000000 pure-copy steps: payload CE 4.778 (below uniform 5.545 for the
first time), probe still 0.0, held-out ~6.8–7.0. Soft prediction and
exact reproduction split cleanly — the model beats chance on payload
bytes yet reproduces none exactly. Time exhausted alongside everything
else as an explanation.

## RTRL 200k (Forgetting Gone)

First long run under corrected gradients: bpb band 5.6–7.6 across all
200k steps, final 5.929, no monotonic component — vs single-step 18.115
at the same horizon. The forgetting curve was an artifact of missing
temporal gradients. All pre-correction training conclusions are legacy
data. Open: memory probes still 0.0 everywhere, including these evals.

## Reset Completion (6M Steps, Flat Throughout)

The reset run hit its 6000000-step cap and exited 0: final loss
0.8481239676475525 (CE component 0.07734909653663635, state_norm
13.127248764038086), closing evals 6.267 / 6.419 / 6.835. Held-out bpb
never left 6–8 across the entire run. No forgetting, no decay, no
explosion — with per-line reset, under single-step autograd dynamics
(predates the unmerged RTRL correction).

## Rank Analysis (Trace Exists, Blurry)

Cued first-payload-byte rank vs chance mean 128: d128-30k 53.5, d128-150k
44.8 (top5 2/30), d256-30k 54.8. Retention exists well above chance and
improves with training — but never top-1. Quest reframes from "create
memory" to "sharpen the trace" (capacity, length, calibration,
contrast). Exact-recall zeros stand; absence does not.

## Per-Position Accuracy (Diffuse, Not Recency)

Recall-head per-position hit rate 0.04/0.02/0.02/0.04 vs chance 0.0105 —
flat across positions, no recency gradient. The trace is diffuse, not
decaying: 2–4x chance everywhere, usable nowhere.

## Rank Flat (No Sharpening With Training)

Mean rank 53.5 → 44.8 → 54.8 → 50.8 across 30k/150k/scale/207k: the
apparent improvement was noise. Rank is stationary; training does not
sharpen the trace.

## Decay Audit (Timescales Survive, Still Zero)

Per-dim half-lives span ~3 to ~1000 steps in every layer of both
trained checkpoints — no collapse. Timescale diversity is necessary
but not sufficient: ~1000-step dims don't yield copy-8-after-8.
Information is not written retrievably at any horizon. Left: selective
gates, reconstructive losses, explicit slots.

## Linear Probe (Readout Fails, Storage Works)

Frozen-state linear classifier on first payload byte: copy-trained
0.1000, general RTRL-200k 0.0700 vs chance 0.0105, majority 0.0500.
Information is linearly decodable at 7–10x chance — even untrained for
copy. Failure localizes to the readout path, not state capacity. Next:
auxiliary decode loss, contrastive sharpening, or distilling the probe
into the head.

## Recall Head (Fits, Doesn't Generalize)

Separate linear head trained on episode queries, trunk frozen: held-out
copy2/copy4 0/20, but fixed-set 8/8. State separates payloads and the
head has capacity; the mapping does not transfer to fresh payloads
(possibly filler-signature memorization). Next: nonlinear readout,
heavier varied training, or accept the wall.

## Nonlinear Readout (Capacity Not the Gap)

MLP head (128-256-256) same protocol: held-out 0/20, fixed-set 8/8 —
identical split at higher capacity. Readout power is not the gap.
Left: far heavier varied training, scale, structural memory — or
single-pass state not factoring into content-addressable form.

## See Also

- [TMT PyTorch Port](pytorch-port.md)
- [Dream-RSI Loop](rsi-loop.md)
