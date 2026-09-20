# Hybrid store fails: states too history-dominated

> Source: tmt-torch SoftExact evals on rtrl200k ckpt (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

SoftExact (verbatim keys, cosine retrieval, LRU): copy probe 0/20 at
lam 1.0 and 0.5; associative probe 0/10 at both lambdas — on the
trained 200k model that predicts well.

Reading: query states never resemble stored states enough, because raw
recurrent states are dominated by full history, not content. Exact
match fails at boundaries (query context unseen); soft match fails on
similarity (history swamps content). Keying on bare key-byte embeddings
would work trivially (a Python dict) and prove nothing. Conclusion:
similarity retrieval over raw small-model states is a dead end without
learned factorization of content from context — which is exactly what
slots v1/v2 failed to learn.
