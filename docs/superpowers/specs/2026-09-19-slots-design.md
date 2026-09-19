# Explicit Slot Memory — Design

**Date:** 2026-09-19
**Status:** Sections 1–2 approved, awaiting spec review
**Motivation:** every parametric lever reads 0.0 on recall; only a mechanism change is left untried.

## 1. Module design

Sidecar `SlotMemory(dim, n_slots=16)`: slots `(n_slots, dim)` buffer
(zero-init), content-addressed write (softmax over slot keys, blended
update) and read (query = decoder state, weighted sum out). Single head
(v1 scope). Config `slots: int = 0` (0 = off, exact current path).
Clean boundary: trunk RTRL math untouched, attribution clean.

## 2. Integration + tests

`TMTModel` holds the module (or None); a separate `mem_head`
(`Linear 2*dim → 256`) reads `[h; read_vec]` only when slots are on —
main decoder shape untouched, old checkpoints load. Training is plain
end-to-end CE on copy episodes (differentiable write/read, no new loss
term yet; auxiliary read loss explicitly deferred). Tests:
write-then-read identity on the module alone; slots=0 path identical
(existing suite); integration smoke (finite training + checkpoint
roundtrip with slot state).

## Out of scope

Multi-head slots, auxiliary read loss, slot addressing search, scale-up
of slot models.
