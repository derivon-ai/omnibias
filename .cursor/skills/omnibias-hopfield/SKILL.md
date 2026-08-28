---
name: omnibias-hopfield
description: Build modern Hopfield / attention-as-operator layers with a closed-form log-sum-exp Jacobian and Hessian. Use when inventing a germ memory, a certified attention Lipschitz, or a jet-Hopfield architecture.
---

# omnibias-hopfield

Modern Hopfield networks and attention-as-operator with closed-form log-sum-exp
Jacobian/Hessian (torch + jax).

## Why nested AD fails

Nested AD through softmax attention is a dense Jacobian at every head; the Hessian is
rarely available. There is no exact log-sum-exp tower on a generic transformer stack.

## What only this tower unlocks

Modern Hopfield networks and attention-as-operator with closed-form log-sum-exp
Jacobian/Hessian (torch + jax).

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

| You want | Import |
| --- | --- |
| Closed-form LSE Jacobian / Hessian | `omnibias.hopfield` |
| Jet Hopfield germ memories | `omnibias.{torch,jax}.architectures.jet_hopfield` |

## Extend

- Source: [`packages/omnibias-hopfield`](../../../packages/omnibias-hopfield).
- Namespace: `omnibias.hopfield`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-hopfield/tests -q`.
- Compose with `omnibias-struct`, `omnibias-curvature`, `omnibias-verify` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A germ-memory Hopfield whose contact match is an exact jet and whose attention Lipschitz
is a sealed Interval.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
