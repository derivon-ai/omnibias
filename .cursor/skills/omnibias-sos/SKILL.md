---
name: omnibias-sos
description: Certify positivity with Sum-of-Squares / Positivstellensatz decompositions and a rigorous interval LDL^T PSD certificate that can earn theorem_prover_verified. Use when inventing a new SOS front-end or an arrangement-adapted basis.
---

# omnibias-sos

Certified universal positivity by optimization: sound Sum-of-Squares /
Positivstellensatz decompositions and the auxiliary-functional (background) method. A
floating-point SDP proposes a Gram matrix; the proof is a rigorous interval LDL^T
positive-definiteness certificate that reuses omnibias.core.verified and the
Mathlib-free Lean kernel obligation, so certificates can earn theorem_prover_verified.

## Why nested AD fails

A floating-point SDP is not a proof. Nested AD cannot certify that a Gram matrix is PSD.
Generic SOS parsers do not outward-round LDL^T or feed the Mathlib-free kernel.

## What only this tower unlocks

Certified universal positivity by optimization: sound Sum-of-Squares /
Positivstellensatz decompositions and the auxiliary-functional (background) method. A
floating-point SDP proposes a Gram matrix; the proof is a rigorous interval LDL^T
positive-definiteness certificate that reuses omnibias.core.verified and the
Mathlib-free Lean kernel obligation, so certificates can earn theorem_prover_verified.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Propose in float, certify with Interval LDL^T.

| You want | Import |
| --- | --- |
| SOS / Positivstellensatz | `omnibias.sos` |
| NPA moment hierarchy | `omnibias.sos.npa` (lower bounds only; not full diagonalization) |
| Combinatorial SOS (clique / 3-XOR) | `omnibias.sos.combinatorial` (not `SosDegreeFamily`; not P vs NP) |
| Arrangement-adapted bases | `omnibias.core.verified.trial_spaces` + sos |

## Extend

- Source: [`packages/omnibias-sos`](../../../packages/omnibias-sos).
- Namespace: `omnibias.sos`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-sos/tests -q`.
- Compose with `omnibias-verify`, `omnibias-discrete`, `omnibias-certificate-lean` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A polynomial whose interval LDL^T certificate seals on the Lean kernel and whose SDP
proposal is recovered from an arrangement-adapted basis.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
