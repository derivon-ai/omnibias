---
name: omnibias-holonomic
description: Run the D-finite engine: Ore algebra, Gosper, creative telescoping, Lean-certified binomial identities, and Jacobian / Keller finite searches. Use when inventing a new annihilator, a rank syzygy, or a finite-family discovery on exact Q.
---

# omnibias-holonomic

A D-finite / holonomic engine: exact Ore (skew-polynomial) algebra for the shift and
differential operators, D-finite / P-recursive objects closed under sum / Hadamard /
Cauchy product, Gosper's algorithm for closed-form indefinite and definite
hypergeometric summation, creative telescoping (guessed-then-verified annihilating
recurrences), and Lean-certified binomial identities whose per-coefficient rational
obligations the omnibias Lean kernel discharges (theorem_prover_verified earned only on
a genuine lake pass). Pure Python; builds on omnibias-core, omnibias-difference and
omnibias-symbolic.

## Why nested AD fails

Nested AD cannot certify a hypergeometric identity. Float SVD rank is not a syzygy.
Generic CAS sessions do not seal `theorem_prover_verified` on per-coefficient rational
obligations.

## What only this tower unlocks

A D-finite / holonomic engine: exact Ore (skew-polynomial) algebra for the shift and
differential operators, D-finite / P-recursive objects closed under sum / Hadamard /
Cauchy product, Gosper's algorithm for closed-form indefinite and definite
hypergeometric summation, creative telescoping (guessed-then-verified annihilating
recurrences), and Lean-certified binomial identities whose per-coefficient rational
obligations the omnibias Lean kernel discharges (theorem_prover_verified earned only on
a genuine lake pass). Pure Python; builds on omnibias-core, omnibias-difference and
omnibias-symbolic.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

| You want | Import |
| --- | --- |
| Ore / Gosper / telescoping | `omnibias.holonomic` |
| Rank syzygy | `omnibias.holonomic.rank_syzygy` |
| Keller / Jacobian n=2 | `omnibias.holonomic.{keller,jacobian_n2}` |
| Ore layer / export | `omnibias.holonomic._core.{layer,export}` |

## Extend

- Source: [`packages/omnibias-holonomic`](../../../packages/omnibias-holonomic).
- Namespace: `omnibias.holonomic`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-holonomic/tests -q`.
- Compose with `omnibias-symbolic`, `omnibias-difference`, `omnibias-discovery-engine`, `omnibias-formal` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A guessed annihilator whose creative-telescoping certificate and Lean binomial identity
are sealed in one ProofMachine pass.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
