# Anchor attempt killed: addressing at chance, box exhausted

> Source: tmt-torch runs/anchor_disc{,2,_ext} + opt-ownership fix
> Collected: 2026-09-20
> Published: 2026-09-20

## Measured 2026-09-20

Minimal MARCH (top-layer state checkpoints every step, bank ≤256/600,
learned keys + softmax routing + null, trunk frozen, BPTT so keys
learn, decoder trains): 3k-episode discriminator with keys actually
stepping: addr 0.30 → 0.29 → 0.3625 vs chance 0.267 (rising,
borderline). Dose-matched extension (11.6k episodes, seed 21):
addr flat 0.28–0.31 ≈ chance, recall 0/20 at all probes. The rise
did not replicate; preponderance is null.

En-route bug (own goal, caught by the frozen metric): the bank was
created AFTER the optimizer, so key/router params were never
stepped — first discriminator measured a frozen router. Fixed
(bank before opt, as slots were), pinned by
test_anchor_params_owned_by_optimizer. The valid discriminator is
disc2/ext, not disc1.

## Kill decision

Box criterion (1) fails at extension: addressing not significantly
above chance. Per the pre-registered box: KILL, no 512-probe, no
+1 variant. The retired hypothesis stays retired; the anchor
variant joins it. Assets retained unchanged (harnesses now include
anchor flags + addressing probe for future external-memory work).
