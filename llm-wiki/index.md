# tmt-torch llm-wiki

Project knowledge for the TMT PyTorch port (Karpathy LLM-Wiki pattern:
catalog here, chronological log in `log.md`, immutable design docs and
postmortems in `raw/articles/`).

## Project

Port of [jrz97619761/test-model-thing](https://github.com/jrz97619761/test-model-thing)
(MLX byte-level recurrent LM with RTUs + latent prediction) to CUDA/PyTorch,
as the foundation for Dream-RSI-style meta-search over TMT configs
([zhengkid/Dream-RSI](https://github.com/zhengkid/Dream-RSI)).
Approach A: faithful port + surgical fixes (multi-timescale decays,
anti-collapse regularization, leak guards, eval fixes).

- Design spec: `docs/superpowers/specs/2026-09-18-tmt-pytorch-port-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-18-tmt-pytorch-port.md`
- Branch: `feat/tmt-port` (11 commits, 13/13 tests green, unmerged)

## Content catalog

- [TMT PyTorch port: plan, execution, and record](raw/articles/2026-09-18-tmt-pytorch-port.md) — 📋 **2026-09-18.** The full story: sources, approach A, module mapping, fix list, eval suite, SDD execution with rulings, gate results, what's next.
