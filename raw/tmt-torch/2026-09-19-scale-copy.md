# Scale-up on pure copy — capacity not the blocker

> Source: tmt-torch run runs/copy256 (30k steps, frac 1.0, dim256/layers8) + probe eval (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

dim256/layers8 (662017 params, 5x dim128), pure copy episodes, 30000
steps, corrected RTRL, exit 0 (~12 steps/s). Payload CE 7.187 (vs 7.923
dim128; uniform 5.545), ingesting probe 0.0 at copy4 x after8/32.
Held-out prose bpb flat ~7.7 throughout (copy stream vs prose eval).

Reading: 5x capacity moves payload CE marginally and recall not at
all. Scale alone does not unlock exact recall at this training budget
(equal-step comparison; bigger model may want more steps — confound
noted, not resolved). Remaining: protocol redesign, much longer copy
training, or structural memory.
