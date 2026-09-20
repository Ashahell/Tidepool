# Tidepool Final Assessment: a recurrent feature extractor, not a memory system

> Sources: tmt-torch campaign record, 2026-09-18/19
> Raw: [2026-09-18-pytorch-port-record](../../raw/tmt-torch/2026-09-18-pytorch-port-record.md); [2026-09-18-rsi-loop-record](../../raw/tmt-torch/2026-09-18-rsi-loop-record.md); [2026-09-18-vk4a-training](../../raw/tmt-torch/2026-09-18-vk4a-training.md); [2026-09-18-replay-comparison](../../raw/tmt-torch/2026-09-18-replay-comparison.md); [2026-09-18-priority-comparison](../../raw/tmt-torch/2026-09-18-priority-comparison.md); [2026-09-18-noise-control](../../raw/tmt-torch/2026-09-18-noise-control.md); [2026-09-18-ema-comparison](../../raw/tmt-torch/2026-09-18-ema-comparison.md); [2026-09-18-rtrl-200k](../../raw/tmt-torch/2026-09-18-rtrl-200k.md); [2026-09-18-memory-zero](../../raw/tmt-torch/2026-09-18-memory-zero.md); [2026-09-18-pure-copy](../../raw/tmt-torch/2026-09-18-pure-copy.md); [2026-09-18-remediation-record](../../raw/tmt-torch/2026-09-18-remediation-record.md); [2026-09-19-consultant-review](../../raw/tmt-torch/2026-09-19-consultant-review.md); [2026-09-19-review-3](../../raw/tmt-torch/2026-09-19-review-3.md); [2026-09-19-gru-contrast](../../raw/tmt-torch/2026-09-19-gru-contrast.md); [2026-09-19-scale-copy](../../raw/tmt-torch/2026-09-19-scale-copy.md); [2026-09-19-rank-flat](../../raw/tmt-torch/2026-09-19-rank-flat.md); [2026-09-19-decay-audit](../../raw/tmt-torch/2026-09-19-decay-audit.md); [2026-09-19-transformer-anchor](../../raw/tmt-torch/2026-09-19-transformer-anchor.md); [2026-09-19-single-episode](../../raw/tmt-torch/2026-09-19-single-episode.md); [2026-09-19-scheduled-sampling](../../raw/tmt-torch/2026-09-19-scheduled-sampling.md); [2026-09-19-decode-ceiling](../../raw/tmt-torch/2026-09-19-decode-ceiling.md); [2026-09-19-hybrid-fails](../../raw/tmt-torch/2026-09-19-hybrid-fails.md); [2026-09-19-lr-sweep](../../raw/tmt-torch/2026-09-19-lr-sweep.md); [2026-09-19-low-lr-long](../../raw/tmt-torch/2026-09-19-low-lr-long.md); [2026-09-19-rank-analysis](../../raw/tmt-torch/2026-09-19-rank-analysis.md); [2026-09-19-linear-probe](../../raw/tmt-torch/2026-09-19-linear-probe.md); [2026-09-19-exact-cache](../../raw/tmt-torch/2026-09-19-exact-cache.md); [2026-09-19-ingate-copy](../../raw/tmt-torch/2026-09-19-ingate-copy.md); [2026-09-19-slots-copy](../../raw/tmt-torch/2026-09-20-slots-copy.md); [2026-09-19-slots-v2-copy](../../raw/tmt-torch/2026-09-20-slots-v2-copy.md); [2026-09-19-causality](../../raw/tmt-torch/2026-09-19-causality.md); [2026-09-19-continuation-value](../../raw/tmt-torch/2026-09-19-continuation-value.md); [2026-09-19-memorization-gate](../../raw/tmt-torch/2026-09-19-memorization-gate.md); [2026-09-19-retrieval-gap](../../raw/tmt-torch/2026-09-19-retrieval-gap.md); [2026-09-19-fastweight-retrieval](../../raw/tmt-torch/2026-09-19-fastweight-retrieval.md); [2026-09-19-credit-addressing](../../raw/tmt-torch/2026-09-19-credit-addressing.md); [2026-09-19-retirement](../../raw/tmt-torch/2026-09-19-retirement.md); [2026-09-20-anchor-kill-box](../../raw/tmt-torch/2026-09-20-anchor-kill-box.md); [2026-09-20-anchor-kill](../../raw/tmt-torch/2026-09-20-anchor-kill.md)

## Verdict

After ~35 controlled experiments across two days, Tidepool (PyTorch RTU,
130–160k params, online single-sample training) is a recurrent feature
extractor with persistent state, not a demonstrated long-term memory
system. Exact recall is 0.0 on every generalization probe tried — and,
critically, the campaign located the failure precisely rather than
merely observing it: **retention is present, retrieval is absent,
credit accelerates fit but not algorithm, and addressing never rises
above chance under any regime.**

## The evidence chain (each link measured, none inferred)

1. **Prediction without retention.** bpb 5.6–7.6 sustained over
   million-step runs; payload CE below uniform; exact recall 0.0
   everywhere, including GRU and transformer baselines at the same
   scale (failure not architecture-specific).
2. **The gate.** The main path could not memorize ONE fixed 16-byte
   episode in 300 reps (null-cycle attractor) — while a readout on
   frozen states fit 8/8. Split: storage fittable, joint generation
   broken. Later closed: 2000 reps memorizes exactly
   (tests/test_memorize_one.py GREEN) — the gate was training amount.
3. **Retention is not scarce.** Training builds slow channels by
   itself: copy150k decays end min 0.77 / mean 0.96 with 64 units
   >0.99 per layer. Linear probes decode the payload at 10x chance
   from slow-channel state (0.00% from fast-channel state).
4. **Causality absent.** Query-point intervention (normal vs zeroed vs
   shuffled vs replaced state): A=B=C=D=1/50. The state carries signal
   the output does not use.
5. **Retrieval, for real.** Delta-rule fast weights (Ba et al. 2016;
   DeltaNet) move memorization 2000 → 300 reps — the only mechanism in
   project history to help — but generalization recall stays 0/20 at
   300k dense steps.
6. **Credit, isolated.** Episode-BPTT machinery gives a clean 2x2
   memorization table: credit 20x, retrieval 7x, combined 40x
   (2000 → 50 episodes). On generalization: faster per-update query
   learning, no snap.
7. **Addressing null.** Query-key matching at chance on every
   checkpoint under every regime (single-step, BPTT, aux, query-gated
   aux); everywhere-aux even poisons it below chance. The slow weights
   do not learn keys. Work stopped here by the continuation-value rule.

## Rejected hypotheses (all with controls, 0 regressions each)

Lower LR; replay ×3 (uniform/priority/noise); EMA; accumulation;
selective gates; input gates; write masks; readout sparsity;
scheduled sampling; slots ×2; exact/soft stores; 5x scale; 11-value
LR sweep; time to 1M steps; fast weights (for algorithm);
episode-BPTT (for algorithm); aux retrieval pressure (both forms).

## Genuine positives (stand regardless of outcome)

- Faithful MLX→Torch port with a frozen eval contract and 122-test
  suite (112 fast + memorization gate + 9 RTRL oracles).
- RTRL remediation: missing-trace correction found by FD (4.3e-9),
  true accumulation, 2-file checkpoints, baselines.
- Full Dream-RSI loop implemented and parked behind an explicit,
  unmet gate (33/33 tests) — search held back by evidence, not hope.
- One-episode exact free generation (memorization gate GREEN).
- 40x memorization table isolating credit from retrieval.
- Proof discipline: canonical path quarantined and guarded
  (tests/test_canonical.py); every number ingested to raw with
  sources; wiki lint 0 errors throughout.

## Open invoice (ranked honestly, per review 2026-09-19)

1. **RTRL trace extension to S** — real remaining mechanism work.
   The only item that is a plan rather than a hope.
2. **Span-supervised / contrastive key objectives** — plausible.
   Directly targets the isolated failure (addressing at chance).
   Currently executing under a hard box (see Kill criteria).
3. **Scale far outside the tested regime** — a hope, not a plan.
   5x scale and 1M steps did not create the algorithm; orders of
   magnitude more might, but there is no mechanism story for why.
4. **A different task family** — partially an admission the current
   task exposes the failure correctly. Changing the task to one that
   does not require genuine addressing is not progress toward the
   original goal.

Anti-cope note: "retention is present" (slow channels, linear
decodability) is a diagnostic clue, not progress. Storage without
retrieval is not memory in any useful sense. The fast-weight result
cuts the other way too: the ONLY mechanism that ever moved a memory
number still failed to generalize — a strong negative signal about
the whole approach at this scale and regime.

## Kill criteria (binding)

If addressing (payload-in-top4 on fresh copy4/after8 episodes, chance
~0.31) remains at chance after the span-supervised attempt below plus
at most one contrastive variant, we RETIRE the pure recurrent +
side-store hypothesis for this goal. No further tricks in this
family. The remaining assets (eval discipline, RSI infrastructure,
proof standards) transfer to a fundamentally different mechanism or
a re-scoped goal (online feature extractor + external memory).

**Kill executed 2026-09-19.** Span-supervised pointer loss, verified
live and dose-matched (11.6k episodes): payload-top4 0.375 ≈ chance,
recall 0/20; fixed-episode 54 vs 49 without ptr. The +1 variant is
deliberately unspent (direct supervision of the kill metric failed;
weaker variants cannot plausibly succeed). The hypothesis is
RETIRED. Standing fork: (a) external differentiable memory with
stronger inductive bias, or (b) re-scope and drop the memory claim
on the recurrent trunk.

**Update 2026-09-20: the MARCH-anchor variant is also killed.** Minimal
top-layer anchors on the frozen trunk (BPTT, keys stepping): one seed
rose to 0.36 vs 0.27 chance at 3k episodes, but the dose-matched
extension sat at chance with recall 0/20 throughout — the rise did
not replicate. An en-route repeat of the slots mistake (bank created
after the optimizer; first discriminator measured a frozen router)
was caught, fixed, and pinned by test. The box is exhausted; no
512-probe, no +1 variant. Both the trunk-state hypothesis and the
anchor variant are retired. Only the fork remains.

## Reproduction

Gate: `PYTHONPATH=src .venv/bin/python -m pytest
tests/test_memorize_one.py -q` (~40s, GPU). Dense harness:
`scripts/copy_dense.py --steps 100000 --dim 64 --layers 3 --fw-dk 16
--bptt`. Addressing probe: `scripts/probe_state.py --ckpt <run>
--config <cfg>`. Causality: `scripts/memory_causality.py --ckpt
<run> --config <cfg>`. Full suite minus slow gates: `pytest tests/
--deselect tests/test_rtrl.py --deselect tests/test_memorize_one.py`.

## See Also

- [Architecture and Evidence](architecture-and-evidence.md) (living synthesis)
- [PyTorch Port](pytorch-port.md) · [RSI Loop](rsi-loop.md) · [Vulkan4Aros Training](vk4a-training.md)
