---
name: omnibias-ferminet
description: Bridge FermiNet with closed-form Laplacians, restricted-depth ansätze, analytic nuclear Hessians, and a reproducible QMC sampling substrate. Use when inventing a Born-Oppenheimer derivative, a Hermite ladder, or a local-energy estimator that rides the sigma tower.
---

# omnibias-ferminet

FermiNet bridge for omnibias: folx-compatible Laplacian adapters, restricted-depth
Tier-2 FermiNet ansatz with closed-form Laplacian, multiblock primitives for analytic
nuclear Hessian / Born-Oppenheimer derivative work, and (in `sampling.py`) a
reproducible, ansatz-agnostic quantum Monte-Carlo sampling substrate (Metropolis /
Langevin walkers, a local-energy estimator, autocorrelation diagnostics, and a
walker-level parameter log-derivative accumulator) -- the first "quantum-lattice-engine"
work item.

## Why nested AD fails

FermiNet Laplacians via nested AD are the campaign bottleneck: every extra electron
derivative rebuilds the graph. There is no closed-form Hermite ladder or nuclear Hessian
on a generic JAX VMC stack.

## What only this tower unlocks

FermiNet bridge for omnibias: folx-compatible Laplacian adapters, restricted-depth
Tier-2 FermiNet ansatz with closed-form Laplacian, multiblock primitives for analytic
nuclear Hessian / Born-Oppenheimer derivative work, and (in `sampling.py`) a
reproducible, ansatz-agnostic quantum Monte-Carlo sampling substrate (Metropolis /
Langevin walkers, a local-energy estimator, autocorrelation diagnostics, and a
walker-level parameter log-derivative accumulator) -- the first "quantum-lattice-engine"
work item.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

| You want | Import |
| --- | --- |
| Folx-compatible Laplacian adapters | `omnibias.ferminet` |
| Hermite ladder | `omnibias.ferminet.hermite` |
| QMC sampling | `omnibias.ferminet` (`sampling.py`) |
| Stochastic reconfiguration | `omnibias.ferminet.stochastic_reconfiguration` |
| Gaussian Slater `log|det M|` | `omnibias.ferminet.antisymmetric` (closed-form Laplacian) |
| Bloch / twist mixed partials | `omnibias.ferminet.bloch` (real affine twist, not a complex phase) |

Pair with `omnibias-jax` for the bit-identical field Laplacian.

## Extend

- Source: [`packages/omnibias-ferminet`](../../../packages/omnibias-ferminet).
- Namespace: `omnibias.ferminet`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-ferminet/tests -q`.
- Compose with `omnibias-jax`, `omnibias-pinn`, `omnibias-deepmind-campaign` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A restricted-depth FermiNet whose closed-form Laplacian matches nested AD to 1e-12 in
float64 and whose nuclear Hessian is an exact jet, not a finite difference.


## Bakeoffs

Nested-AD cost and closed-form accuracy live in
[`docs/benchmarks/`](../../../docs/benchmarks/):
`laplacian_scaling.json`, `polylaplacian_order.json`,
`derivative_order.json`, `jet_vs_nested_ad_smoke.json`.
Heavy regeneration follows the workspace compute rule, not a skill taboo.

## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
