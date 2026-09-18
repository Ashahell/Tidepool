# Serious run 1 (dim128/layers4, 30k steps) — overfitting record

> Source: tmt-torch run runs/serious1 (train.log, loss.csv, eval.csv, state.json, node.json; checkpoints gitignored)
> Collected: 2026-09-18
> Published: 2026-09-18

## Run measured 2026-09-18

configs/small.yaml (dim 128, layers 4, lr 0.0005, update_every 1),
30000 steps on data/vk4a_train, exit 0 in 49.1 s wall (610.6
bytes_per_sec on 8 CPU cores) via train.py --steps 30000 --eval-every
2000.

Held-out bpb by step: 2000: 6.660, 4000: 6.790, 6000: 8.745, 8000:
8.172, 10000: 9.012, 12000: 8.784, 14000: 9.690, 16000: 11.610, 18000:
12.807, 20000: 11.843, 22000: 11.924, 24000: 11.929, 26000: 13.386,
28000: 13.646, 30000: 14.301. Composite -74.230 down to -168.716.
Training loss fell while held-out bpb rose monotonically after ~2000
steps: pure online single-byte updates memorize, then miscalibrate
(overconfident-wrong). Final components: l_var 0.14055204391479492,
l_pred 1.9760305881500244, l_ce 1.9381849765777588, l_stop
0.0011033909395337105, state_norm 4130.05859375. node min_loss
0.014731924049556255 (single-step outlier).

Reading: this is the continual-learning pathology the RSI loop exists to
search against (replay buffers, lower LR, update_every > 1,
regularization). Throughput is not the bottleneck; generalization is.
