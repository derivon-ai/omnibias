---
name: omnibias-qcalculus
description: Compute q-numbers, Gaussian q-binomials, Jackson q-derivatives / q-integrals, and q-exponentials whose q -> 1 limit recovers ordinary calculus. Use when inventing a q-deformed tower or a hybrid Jackson cell.
---

# omnibias-qcalculus

Quantum / q-calculus: exact q-numbers, q-factorials, Gaussian (q-)binomials and
q-Pochhammer symbols, the Jackson q-derivative and q-integral, q-exponentials and
q-deformed Bernoulli / Euler numbers, and basic hypergeometric series with certified
geometric tails. The q -> 1 limit recovers ordinary calculus (a distinct limit, never
conflated with the delta -> 0 founding collapse). Built on omnibias-core and
omnibias-difference, with bit-identical torch/jax Jackson-derivative twins.

## Why nested AD fails

Nested AD has no Jackson derivative. Generic q-calculus notebooks are symbolic only;
they do not share ActivationSpec or bit-identical torch/jax twins, and they mix `q -> 1`
with `delta -> 0`.

## What only this tower unlocks

Quantum / q-calculus: exact q-numbers, q-factorials, Gaussian (q-)binomials and
q-Pochhammer symbols, the Jackson q-derivative and q-integral, q-exponentials and
q-deformed Bernoulli / Euler numbers, and basic hypergeometric series with certified
geometric tails. The q -> 1 limit recovers ordinary calculus (a distinct limit, never
conflated with the delta -> 0 founding collapse). Built on omnibias-core and
omnibias-difference, with bit-identical torch/jax Jackson-derivative twins.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

The `q -> 1` limit is a named specialization, distinct from founding
bias collapse (`delta -> 0`) and from temperature collapse (`beta -> inf`).

| You want | Import |
| --- | --- |
| q-numbers / Jackson derivative | `omnibias.qcalculus` |
| Hybrid Jackson cell | `omnibias.qcalculus._core.hybrid` |

## Extend

- Source: [`packages/omnibias-qcalculus`](../../../packages/omnibias-qcalculus).
- Namespace: `omnibias.qcalculus`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-qcalculus/tests -q`.
- Compose with `omnibias-timescale`, `omnibias-difference`, `omnibias-core` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A Jackson–OMBU hybrid whose `q -> 1` jet matches the ordinary tower bit-for-bit and
whose q-binomial identities seal on the Lean kernel.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
