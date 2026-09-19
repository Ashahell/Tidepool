# Generation constraint reveals fragments

> Source: tmt-torch inline SS-300 run + constrained generation eval (output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Same SS-300 single-episode setup, generation with byte 0x00 banned:
unconstrained emits oo@~-then-null-cycles (no match); constrained emits
oo@~oo@~Moo@~oo@ — payload fragments repeating, still no exact match.

Reading: the null attractor masked a partial signal — banning it
reveals payload-structure emission. Sequencing still fails (repetition
instead of progression), but the failure moved from degenerate to
fragmentary. Constraints are diagnostic, not a cure; next: repeat
penalties, sampling instead of argmax, or accepting the sequencing gap.
