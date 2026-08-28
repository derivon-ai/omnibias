---
name: omnibias-convex
description: Solve differentiable certified LP/QP with a closed-form-Hessian log-barrier interior-point method, KKT implicit-function gradients, and verified optimality enclosures. Use when inventing a new convex front-end or making argmin a differentiable op with a sealed dual gap.
---

# omnibias-convex

Differentiable + certified convex optimization (LP/QP) for omnibias: closed-form-Hessian
log-barrier interior-point solver, KKT implicit-function gradients (argmin as a
differentiable op), and verified optimality enclosures (jax + torch).

## Why nested AD fails

Unrolling an interior-point loop through nested AD is a long, ill-conditioned graph.
Implicit differentiation without an exact Hessian of the barrier loses the KKT residual.
Generic QP solvers do not return Interval dual bounds.

## What only this tower unlocks

Differentiable + certified convex optimization (LP/QP) for omnibias: closed-form-Hessian
log-barrier interior-point solver, KKT implicit-function gradients (argmin as a
differentiable op), and verified optimality enclosures (jax + torch).

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

| You want | Import |
| --- | --- |
| Log-barrier IP / KKT grads | `omnibias.convex` |
| Arrangement LP front-end | `omnibias.convex.arrangement` |
| Torch / jax twins | `omnibias.convex.{torch,jax}` |

Temperature collapse (`beta -> inf`) hardens learned-facet gates; the
barrier Hessian is the closed-form tower, not nested AD.

## Extend

- Source: [`packages/omnibias-convex`](../../../packages/omnibias-convex).
- Namespace: `omnibias.convex`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-convex/tests -q`.
- Compose with `omnibias-discrete`, `omnibias-routing`, `omnibias-sos`, `omnibias-verify` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A box-QP whose KKT IFT gradient and Neumaier-Shcherbina dual enclosure agree on a named
portfolio / control QP to machine precision.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
