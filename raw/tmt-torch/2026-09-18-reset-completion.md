# Reset run completion (6M steps) — no forgetting observed

> Source: tmt-torch run runs/epoch_reset (train.log, loss.csv, state.json; gitignored)
> Collected: 2026-09-18
> Published: 2026-09-18

## Measured 2026-09-18

Per-line-reset run reached its 6000000-step cap and exited 0 (process
gone, heartbeat final): step 6000000 loss 0.8481239676475525 (l_var
0.4433974623680115, l_pred 0.3273750841617584, l_ce 0.07734909653663635,
l_stop 2.3277359559870092e-06, state_norm 13.127248764038086),
5960000 bpb=6.267, 5980000 bpb=6.419, 6000000 bpb=6.835. Held-out bpb
never left the 6–8 band across the entire run; train CE fell to 0.077.
No forgetting, no decay, no explosion through 6M steps with per-line
reset (single-step autograd dynamics — predates the RTRL correction,
which is unmerged).
