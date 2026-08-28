---
name: omnibias-timescale
description: Unify continuous and discrete calculus on a TimeScale via delta / nabla derivatives whose graininess mu -> 0 recovers the derivative tower. Use when inventing a Hilger hybrid or a new graininess schedule.
---

# omnibias-timescale

Time-scale (Hilger) calculus unifying the continuous and discrete registers: a TimeScale
(R, hZ, quantum q^Z, or a finite set) with forward/backward jump operators and
graininess, the delta and nabla derivatives dispatching to the closed-form omnibias
tower on R, the forward difference on hZ, and the Jackson q-derivative on q^Z, plus the
delta integral, the Hilger exponential and its circle-plus group, and linear dynamic
equations. The graininess mu -> 0 recovers the derivative tower -- the founding delta ->
0 bias collapse, generalized. Built on omnibias-core, omnibias-difference and
omnibias-qcalculus, with bit-identical torch/jax delta-derivative twins.

## Why nested AD fails

Nested AD is defined on R, not on hZ or q^Z. Generic time-scale code does not dispatch
to the closed-form tower on R and the Jackson derivative on q^Z from one TimeScale
object.

## What only this tower unlocks

Time-scale (Hilger) calculus unifying the continuous and discrete registers: a TimeScale
(R, hZ, quantum q^Z, or a finite set) with forward/backward jump operators and
graininess, the delta and nabla derivatives dispatching to the closed-form omnibias
tower on R, the forward difference on hZ, and the Jackson q-derivative on q^Z, plus the
delta integral, the Hilger exponential and its circle-plus group, and linear dynamic
equations. The graininess mu -> 0 recovers the derivative tower -- the founding delta ->
0 bias collapse, generalized. Built on omnibias-core, omnibias-difference and
omnibias-qcalculus, with bit-identical torch/jax delta-derivative twins.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

`mu -> 0` recovers the ordinary tower. Distinct from `q -> 1` and from
temperature collapse.

| You want | Import |
| --- | --- |
| TimeScale / delta / nabla | `omnibias.timescale` |
| Hilger hybrid | `omnibias.timescale._core.hybrid` |

## Extend

- Source: [`packages/omnibias-timescale`](../../../packages/omnibias-timescale).
- Namespace: `omnibias.timescale`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-timescale/tests -q`.
- Compose with `omnibias-qcalculus`, `omnibias-difference`, `omnibias-core` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A hybrid cell whose Hilger exponential on hZ and closed-form tower on R agree in the `mu
-> 0` jet to machine precision.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
