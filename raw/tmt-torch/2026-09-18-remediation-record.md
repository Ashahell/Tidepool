# RTRL remediation measurements — verification record

> Source: tmt-torch SDD execution records + pytest output on branch feat/rtrl-fix (later merged)
> Collected: 2026-09-18
> Published: 2026-09-18

## Measured 2026-09-18

Source-level equivalence (NumPy transcription vs torch): forward and
trace margins at most 1.2e-6 (states/logits/traces, tolerance 1e-5);
loss margin 6.7e-4 (tolerance 1e-3 hedge). RTRL finite-difference proof:
1-step all classes at most 6.3e-6, 2-step at most 1.6e-5, 5-step decay
4.3e-9, embed 1.2e-9 (tolerance 1e-4; pred-target rows excluded per
upstream stop_gradient). Memory oracles on the ingesting probe:
PerfectMemory 1.0, OneByteMemory 0.0, Amnesiac 0.0. Ablation matrix
(8 cells): RTRL bpb 7.454 vs single-step 7.446, copy 0.0 everywhere,
reset cells bit-identical. Full suite counts observed: 46, 68, 70, 72
passed at successive landings (72 passed final).
