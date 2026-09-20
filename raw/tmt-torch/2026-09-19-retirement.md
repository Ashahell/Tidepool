# Kill criteria executed: pure recurrent + side-store retired

> Source: tmt-torch runs/copydense_fw_ptr{,700k} + tests/test_ptr.py
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Span-supervised pointer loss (NLL of query→payload-key attention,
verified live: l_ptr ~2.5–3.1 engaged, ring populated, falling
ep0→ep1): fixed episode solves at 54 eps vs 49 without ptr (ptr
adds nothing where everything works); fresh episodes 3k eps →
payload-top4 0.250 (below chance ~0.31); dose-matched 11.6k eps →
payload-top4 0.375 (~1 SE above chance, null), recall 0/20 at all
probes. Matched-dose no-ptr baseline 0.362 — supervision added
nothing.

## The binding decision

Kill criteria (final-assessment.md): span attempt + at most one
contrastive variant. The +1 is deliberately UNSPENT: the span loss
directly optimizes the kill metric itself, with the mechanism
verified live. A weaker/indirect variant cannot plausibly succeed
where the direct objective failed dose-matched — spending it would
be drift, not diligence. The pure recurrent + side-store hypothesis
for exact long-range recall is RETIRED.

## Standing assets and fork

Retained: eval discipline (122-test suite, frozen contract),
Dream-RSI infrastructure (parked, gated), proof standards
(canonical quarantine + guard), all harnesses
(copy_dense/copy_bptt/memory_causality/probe_state). Fork: (a)
fundamentally different memory (external differentiable store with
stronger inductive bias, trunk not the primary store), or (b)
re-scope to online feature extractor + external memory, dropping
the long-term-memory claim on the recurrent trunk.
