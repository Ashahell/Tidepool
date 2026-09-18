# TMT PyTorch Port

> Sources: tmt-torch session record, 2026-09-18
> Raw: [2026-09-18-pytorch-port-record](../../raw/tmt-torch/2026-09-18-pytorch-port-record.md)
> Updated: 2026-09-18

## Overview

The 230-line MLX byte-level recurrent LM (152 stars, 4.5M parameters with
dim=512 and layers=16, about 12 hours on Simple Wikipedia) was ported to
CUDA/PyTorch with surgical fixes in one go (approach A), executed as five
reviewed tasks plus a fix wave on branch feat/tmt-port (12 commits,
fast-forward merged), ending with a controller-verified gate of 13 passed.

## Why This Shape

The port keeps the single-byte online loop and RTRL trace structure so
behavior stays comparable to the MLX reference, while fixing the known
weaknesses during the port rather than after. The Dream-RSI side
(688 stars, paper + site out, full code still pending release) contributes
only the outer-loop design for now: the eval suite and Node-JSON logging
are built to plug into its explore → replay → dreaming loop later.

## What Landed

Torch 2.14.0+cu130 with CUDA 13.0 on NVIDIA GeForce RTX 5070 Ti (sm (12, 0),
cuda_available True). Model core with RTRL trace buffers, multi-timescale
decay init from half-lives 8, 64, 512, 4000 (per-step decays 0.917, 0.989,
0.999, 1.0, spread 0.083, tested exact-values plus ordering plus spread
>0.05), eval suite with composite weights -12.0, 6.0, 9.0, 4.0 and -0.8,
and CLIs with bounded generation (gen_bytes, max_bytes=256,
stop_threshold=0.35), loss.csv, and node.json output.

## Execution Record

Two transcription bugs caught by tests (missing file handle in YAML load,
2-D instead of 1-D byte tensors), one plan defect corrected by measurement
(the impossible spread >0.2 threshold), and one final fix wave for a
generation hang plus CLI logging gaps. Parked deliberately: machine-pinned
CUDA test, tolerant checkpoint loading, and the MLX-faithful trace update.
Next: merge decision made (merged), then a first real training run and the
Dream-RSI outer loop on top of the inner-loop scorer.

## See Also

- [Implementation plan](../../docs/superpowers/plans/2026-09-18-tmt-pytorch-port.md) (project doc, not a wiki article)
