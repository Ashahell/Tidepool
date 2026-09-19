# Slot Addressing v2 — Design

**Date:** 2026-09-19
**Status:** Approved, replacing v1 addressing wholesale
**Motivation:** v1 measured 0.0/0.0; review convicted the Hebbian write rule.

## 1. Shared keys, temperature, usage protection

Single `key` projection serves write addressing and read addressing:
it learns retrieval through the read path and improves writes as a
side effect. Write scores use key-space similarity with temperature
`slot_temp` (default 1.0). Per-slot usage counters accumulate write
mass; protect scaling suppresses writes to high-usage slots
(`w * sigmoid(protect_gain * (protect_thresh - usage))`, constants
5.0/1.0 in code). Erase via the existing blend. Writes stay
gradient-free buffer ops — documented, unchanged from v1.

## 2. Integration (unchanged surfaces)

Config gains `slot_temp: float = 1.0`. Read path, mem_head, aux loss,
read-only generation, reset (now also zeroes usage), slots=0 path all
as v1. Retention experiment re-run identically (slots v2 vs off).

## Out of scope

Learned write strength, multi-head, addressing search, scale-up.
