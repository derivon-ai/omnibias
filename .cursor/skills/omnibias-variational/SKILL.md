---
name: omnibias-variational
description: Run least-action calculus: action integrals, Euler-Lagrange / Euler-Poisson residuals, constrained variations, Hamiltonian / Noether, and symplectic integrators with trajectory derivatives from the closed-form tower. Use when inventing a new action or a certified variational residual.
---

# omnibias-variational

Least-action / variational calculus for omnibias: action integrals, Euler-Lagrange
residuals, the arbitrary-order Euler-Poisson functional derivative, the first (Gateaux)
variation, constrained (holonomic / isoperimetric) variational calculus, Hamiltonian /
Noether machinery, classical field theory, geodesics-as-least-action, and symplectic
integrators, with the trajectory derivatives supplied closed-form by the omnibias-fields
sigma-tower and rigorous action enclosures from omnibias.core.verified.

## Why nested AD fails

Nested AD through a discretized action is a long graph with no Euler-Poisson identity.
Generic variational PINNs finite-difference the functional derivative and cannot enclose
the action.

## What only this tower unlocks

Least-action / variational calculus for omnibias: action integrals, Euler-Lagrange
residuals, the arbitrary-order Euler-Poisson functional derivative, the first (Gateaux)
variation, constrained (holonomic / isoperimetric) variational calculus, Hamiltonian /
Noether machinery, classical field theory, geodesics-as-least-action, and symplectic
integrators, with the trajectory derivatives supplied closed-form by the omnibias-fields
sigma-tower and rigorous action enclosures from omnibias.core.verified.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Trajectory derivatives are closed-form; domain quadrature is numerical
(integral sense 2).

| You want | Import |
| --- | --- |
| Action / EL / Euler-Poisson | `omnibias.variational` |
| Hamiltonian / Noether | `omnibias.variational` |

## Extend

- Source: [`packages/omnibias-variational`](../../../packages/omnibias-variational).
- Namespace: `omnibias.variational`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-variational/tests -q`.
- Compose with `omnibias-fields`, `omnibias-geometry`, `omnibias-pinn`, `omnibias-verify` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A named Lagrangian whose Euler-Poisson residual is an exact jet and whose action
enclosure contains the symplectic-integrator discrete action.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
