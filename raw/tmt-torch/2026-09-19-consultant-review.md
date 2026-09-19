# External consultant review — Tidepool autopsy + phased path

> Source: trusted 3rd-party consultant messages, pasted 2026-09-18/19
> Collected: 2026-09-19
> Published: 2026-09-19

## Verdicts recorded

Solid: RTRL proven to 4.3e-9; stable 6M-step training with bpb 5.6–7.6;
honest negatives (replay x3, EMA, pure copy, accumulation); RSI loop
actually closed with tests; 72 green tests. Ahead of most continual
learning talk.

Central failure: long-range retention is zero everywhere — the entire
point of the TMT claim. The model predicts locally and forgets
immediately; the memory is theater. Evidence: per-line reset fixed
poisoning (state norm 15k to 70); lower LR only delays; replay hurts;
pure copy fails at dim128; RTRL approximately equals single-step at
this scale.

Prescription: treat the RTU as a local feature extractor, not memory.
Freeze the composite score; retention (copy 512/2k/8k/32k, associative
retrieval, bracket matching, state reconstructibility) is the only gate.
Change the inductive bias hard (multi-timescale/complex RTUs, selective
gates, reconstructive auxiliary losses, tiny explicit memory, real copy
curriculum). Point the RSI loop at retention. Scale probes (GRU/LSTM
baselines, over-parameterized version). No new complexity until
retention moves. Dream-RSI itself not invalidated: it needs graded
signals, and Tidepool proves the outer loop is implementable; it cannot
invent missing inductive bias.

Dream-RSI implications: orthogonal and transferable as machinery;
low-yield pointed at zero-retention architectures; needs
differentiated outcomes to climb.

Phased path: Phase 1 retention-only gate with milestones (20% copy at
2k, 50% at 8k, surviving sequential tasks, loop-discovered gains);
Phase 2 inductive-bias fixes one at a time; Phase 3 learning rules
after retention exists; Phase 4 RSI loop retargeted with retention in
tree nodes; Phase 5 scale only after capability exists. Stop: complex
replay at zero retention, celebrating CE/BPB, outer-loop sophistication
first, assuming hyperparameters create memory.
