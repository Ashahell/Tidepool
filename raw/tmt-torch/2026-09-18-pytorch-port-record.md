# tmt-torch PyTorch port — session record

> Source: tmt-torch project session record (upstream repo snapshots + build/test measurements taken during the session)
> Collected: 2026-09-18
> Published: 2026-09-18

## Upstream sources observed 2026-09-18

Test-Model-Thing (jrz97619761/test-model-thing): 152 stars, MIT license.
The reference model has 4.5M parameters with dim=512 and layers=16, trained
about 12 hours on a small Simple Wikipedia dump using MLX. The entire model
is a 230-line main.py plus benchmark.py. It does byte-level input/output,
JEPA-style latent-space prediction, per-dim sigmoid-decay recurrent state
(RTUs), continuous streaming, and test-time training.

Dream-RSI (zhengkid/Dream-RSI): 688 stars. Research framework from a
Google / Google DeepMind / University of Maryland / University of Virginia
team. Paper + site are out; full code is still pending release ("being
prepared"). Core idea: online explore, construct replay simulator from
discovery trees, dreaming-based policy improvement. No model weight updates.

## Environment measured 2026-09-18

Host GPU is NVIDIA GeForce RTX 5070 Ti with sm (12, 0). Installed
torch 2.14.0+cu130 with CUDA 13.0 via uv venv on Python 3.14.
Verification printed cuda_available True.

## Port execution measured 2026-09-18

Work happened on branch feat/tmt-port: 12 commits from 2edba4b
("feat: add TMTConfig with YAML defaults") through 206a857
("docs: add project llm-wiki (index, log, port record)"), then merged
to master with a fast-forward merge. Approach A was chosen: faithful port
plus surgical fixes in one go.

Decay init uses half-lives 8, 64, 512, 4000, giving per-step decays
0.917, 0.989, 0.999, 1.0 with spread 0.083. The test asserts exact
per-group values plus ordering plus spread >0.05. An early draft asserted
spread >0.2, which the formula cannot satisfy; the implementer measured
spread 0.083 and the threshold was corrected.

Five subagent-driven tasks (config, model core, decay fixes, engine/CLIs,
gate) each passed task review; one fix round restored the half-life
formula; one final fix wave added bounded generation gen_bytes with
max_bytes=256 and stop_threshold=0.35, a metrics.py re-export, train
--data/--seed flags with loss.csv and node.json output, and eval node.json
writes. Final gate: 13 passed across 6 test files (test_cli.py,
test_config.py, test_fixes.py, test_gate.py, test_model.py,
test_wiring.py), controller-verified on the merged tree.

Eval suite composite weights are -12.0 for bits-per-byte, 6.0 for
long-range memory, 9.0 for continual retention, 4.0 for stability, and
-0.8 efficiency penalty per GPU-hour.
