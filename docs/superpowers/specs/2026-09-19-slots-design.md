# Explicit Slot Memory — Design

**Date:** 2026-09-19 (revised per external review)
**Status:** Awaiting re-review
**Motivation:** every parametric lever reads 0.0 on recall; only a mechanism change is left untried.

## 1. Module design (revised: learned retrieval, honest cache writes)

Sidecar `SlotMemory(dim, n_slots=16)` with query/key projections
(`Linear dim→dim`, learned) and fixed-blend content write. Deliberate,
documented split: the WRITE side is a non-learned Hebbian cache
(in-place buffer updates carry no gradient — stated, not hidden);
the READ side learns (query/key projections get gradients through the
read path every step). Temperature fixed at 1.0 (v1 scope).
Config `slots: int = 0` (0 = off, exact current path).

## 2. Integration + tests (revised)

`TMTModel` holds the module (or None); separate `mem_head` on
`[h; read]`; main decoder untouched. Two training signals on copy
episodes: end-to-end CE (existing) PLUS auxiliary retrieval loss
(`1 - cos(read_vec, embed(target))`, weight `aux_mem_w`, default 0.0)
that directly pressures reads to reconstruct targets. Generation
contract: NO writes during generation (read-only recall), unit-tested
by generation determinism (two consecutive generations identical).
Tests: write/read retrieval with projections; off-path identical;
aux-loss helper unit test; integration smoke (finite + roundtrip);
retention gate = copy accuracy at distances post-training (experiment
record, Task 3 — a unit suite passing here proves nothing about
recall).

## Out of scope

Multi-head slots, learned write strength/erase gates, addressing
search, scale-up of slot models.
