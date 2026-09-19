# Linear probe — information present, readout fails

> Source: tmt-torch scripts/probe_state.py runs on saved checkpoints (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Linear classifier trained on frozen final-layer states to predict the
first payload byte (400 train / 100 test episodes, chance 0.0105,
majority 0.0500): copy-trained ckpt probe_acc=0.1000; general RTRL-200k
ckpt probe_acc=0.0700.

Reading: payload information is linearly decodable at 7–10x chance —
even in a model never trained to copy. The readout (the model's own
decoder under argmax) fails, not the storage. Failure localizes to
decoding/training of the readout path, not state capacity. Next: train
the readout (auxiliary decode loss), sharpen via contrast, or distill
the linear probe back into the head.
