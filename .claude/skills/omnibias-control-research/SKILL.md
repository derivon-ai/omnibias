---
name: omnibias-control-research
description: Ambitious, falsifiable control-systems research inside omnibias-control — jet-adjoint policy gradients, certified truncation horizons, sound gradient-bias enclosures, and certified contact smoothing. Use when closing control-systems capability gaps or designing an adjoint / certified-control benchmark.
---

# Control-systems research (theory Group 10)

The bet: differentiable RL does not currently ship a *certified* horizon or a
*sound* policy-gradient-bias enclosure. omnibias can, because
`omnibias.dynamics.spectral_radius_bound` and `omnibias.verify.lipschitz_bound`
already exist. Lead with that moat.

## Why nested AD fails

BPTT and PPO pick a horizon by hand. Nested AD through a rollout cannot
enclose `||grad J - grad J_h||`. Reimplementing spectral-radius or Lipschitz
primitives inside the policy loop forks the rigorous register. Lookalike
training-loop PIDs (`08-10` / `08-11` / `09-29`) are not a plant MIMO adjoint.

## What only this tower unlocks

Exact `dpi/dy` from the jet. Certified truncation from enclosed closed-loop
monodromy. Sound (conditional) policy-gradient-bias enclosure. Certified
contact smoothing on the `sigma^(n)` tower. Two limits, both available:
CBF-QP temperature collapse (`beta -> inf`); adjoint bias collapse
(`delta -> 0`). `__lineage__ = "both"`.

## Use

1. No new package. `test_theory_homes.py` pins the workspace closed. Land in
   `omnibias.core.adjoint` or `omnibias.control.{ocp,horizon,bundle,robotics}`
   / `.{jax,torch}.{adjoint,policy,envs,contact}` / `.certified.gradient_bias`.
   Read `theory/10-control/01-ledger.md` first.
2. Reuse certified primitives verbatim. Discrete monodromy via
   `omnibias.core.verified.linalg.matmul` (the adjoint `M_j` sequence is
   already discrete).
3. Keep both founding limits labelled. New files that touch both get the
   boilerplate paragraph (`omnibias.control.horizon` docstring) and go in
   `PENALTY_FILES` in `packages/omnibias-core/tests/test_concept_terminology.py`.
4. Absolute gates from a measured Phase-A curve on this repo's hardware
   (`omnibias-empirical-validation`).

Smoke JSON is committed; named-baseline RL sample-efficiency (G4) and
robotics cost parity (G8) are recorded as later slices.

## Extend

Compose with `omnibias-control`, `omnibias-dynamics`, `omnibias-verify`,
`omnibias-empirical-validation`. Tests:
`python -m pytest packages/omnibias-control/tests -q`.

## Next invention

A MIMO `DifferentiableEnvironment` whose `certified_horizon` plus
`truncation_bias_bound` are consumed automatically by `actor_adjoint_jet_step`,
with a `gates` JSON against `bptt_step` on DoubleGyre.

## Bakeoffs

`docs/benchmarks/actor_adjoint_control_smoke.json`,
`certified_horizon_smoke.json`, `gradient_bias_enclosure_smoke.json`.

## Further references

- `docs/api/adjoint_control.md`
- `docs/cookbook/certified-policy-gradient.md`
