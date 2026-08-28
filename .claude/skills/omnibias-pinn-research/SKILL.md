---
name: omnibias-pinn-research
description: Ambitious, falsifiable PINN research inside omnibias-pinn — turn every caveat into an acceptance test, iterate after a losing baseline, and claim the capability the moment the absolute gate passes. Use when closing PINN capability gaps (causality, geometry, operators, spectral bias) or designing a new benchmark.
---

# PINN research — constructive by default

Closed-form activation derivatives and exact hard cages are leverage. Use
them to dissolve named limitations on named problem families. Claim a
capability the moment its absolute gate passes.

## Why nested AD fails

Nested-AD PINNs stall on high-order residuals, leak soft BCs, and report
relative comparisons between two diverged arms. A missing baseline or a
failed first smoke is treated as impossibility. Spectral bias and causality
stay folklore because the residual operator itself is approximate.

## What only this tower unlocks

Hard cages, closed-form towers, one-shot collocation, causal marching,
conditioned operators, multilevel FBPINN. Absolute gates: multi-seed skill
> 0, by-construction identity, or a sound certificate. The four-gap matrix
is the named unlock nested AD stacks do not clear.

## Use

1. Default to achievable. Hunt the constructive route first.
2. Earn the claim when the gate passes. Relative comparisons between two
   failing arms do not count.
3. Prefer hard BCs / SDF cages / causal marching / FBPINN / conditioned
   operators / one-shot least-squares.
4. Turn every caveat into a falsification test (`test_*_junction_raises`).
5. Iterate after a losing baseline. Scope reduction is the last move.

Phase A: bold prototypes in alpha submodules (`train`, `domain`, `operator`),
GPU sweeps under `$OMNIBIAS_SCRATCH`. Phase B: acceptance-gated APIs, parity
tests, smoke + multi-seed JSON with a `gates` block.

Validity floor: ask whether the experiment is well-posed before reading MSE.
Worked case: parametric heat with explicit RK4 blew up; ETDRK4 plus a
maximum-principle guard made the bakeoff real.

New top-level packages still earn independence (`omnibias-new-package`).

## Extend

Compose with `omnibias-pinn`, `omnibias-fields`, `omnibias-empirical-validation`,
`omnibias-frontier`, `omnibias-verify`. Gates: `benchmarks/_gates.py`.

## Next invention

The next four-gap row that is still a leftover: turn it into a multi-seed
absolute win with a hard cage or closed-form residual, then ship the `gates`
JSON.

## Bakeoffs

`docs/benchmarks/pinn_four_gap_matrix.md`; laplacian / polylaplacian /
derivative_order JSON under `docs/benchmarks/`.

## Further references

- `docs/benchmarks.md`, `docs/honesty.md`
