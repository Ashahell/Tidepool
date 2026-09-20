# External-store attempt killed: addressing below chance

> Source: tmt-torch runs/ext_disc (frozen trunk, BPTT, 3k episodes)
> Collected: 2026-09-20
> Published: 2026-09-20

## Measured 2026-09-20

Minimal true external store (separate K/V matrix, learned
write/erase/protect, content-only reads, learned temperature,
trunk frozen, BPTT so keys learn, decoder trains): 3k episodes →
addr 0.0875–0.10 vs chance 0.267 (payload-top4, 20 trials;
~−3σ, systematically BELOW chance), recall 0/20 at all probes.
Queries point away from payload slots (recency bias signature:
marker-state queries match recent filler keys).

En-route bug (caught, fixed, pinned): store K/V attached in
record mode still link the freed prior-episode graph after reset
(zero_ keeps grad_fn) — second finish_episode faults. Fixed by
detach in reset() + finish_episode(); regression test
test_bptt_two_episodes_no_stale_graph. Same bug class as fw.S.

## Kill decision

Box criterion (1) fails: addressing not above chance — below it.
Per the pre-registered box: attempt KILLED, no extension, no +1.
Reading per the decision note: the line's central prediction
(external store learns addressing where recurrent variants could
not) failed its first fair test, and failed below chance rather
than at it. The terminal clause is now live: absent a strategic
decision otherwise, the project archives with the evidence.
Recommended: accept archival. A second seed cannot supply the
replication a pass would require, and nothing in 40+ experiments
suggests the next variant differs.
