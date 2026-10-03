---
name: omnibias-holonomic
description: Run the D-finite engine: Ore algebra, Gosper, creative telescoping, Lean-certified identities, Jacobian / Keller searches, and the finite twin-prime sieve ledger. Use for annihilators, exact-Q syzygies, or finite-family discovery.
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
| Twin-prime finite obligations | `omnibias.holonomic.twin_prime` |
| Ore layer / export | `omnibias.holonomic._core.{layer,export}` |

## Extend

- Source: [`packages/omnibias-holonomic`](../../../packages/omnibias-holonomic).
- Namespace: `omnibias.holonomic`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-holonomic/tests -q`.
- Compose with `omnibias-symbolic`, `omnibias-difference`, `omnibias-discovery-engine`, `omnibias-formal` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.
- Twin-prime work separates exact local factors and 3-D log-polyhedral cell
  partitions from the external remainder, terminal-shell Möbius cancellation,
  and uniform-asymptotic hypotheses. The FI (3.1) Vaughan identity is replayed
  coefficientwise on finite divisor lattices; exact dyadic/rho/routing
  subchecks reject averaged-shift and explicitly circular inputs. A0, A1a
  finite combinatorics, A1b0 structural subchecks, and atlas geometry may be
  certified independently; neither a finite prefix nor a float diagnostic
  licenses the parent.
- The terminal-shell candidate uses the exact signed transform
  `r*d*t - q*k = 2`. Replay it coefficientwise in the free `log(p)` basis;
  never take absolute values before recombining the oriented
  `Lambda = mu * log` terms. Exact Wright-range and Kloosterman-kernel cell
  arithmetic are bookkeeping only: the loss-budgeted fixed-2 completion lemma
  remains an external analytic obligation.
- The bounded-gap scalar target after the named gap-186 baseline is a
  source-valid `k=39` rational generalized-Rayleigh crossing for the explicit
  diameter-182 tuple. A crossing over caller-declared matrices is finite
  arithmetic only until cap/source losses, support, distribution, and the DHL
  implication are independently discharged.
- Parameterize the pinned PrimeGaps186 evaluator through
  `twin_prime_bounded_gap.build_prime_gap_input_manifest`: first reproduce the
  exact `k=40` 97-component / 149-form manifest, then require `k=39` to use
  outer/face dimensions `39/38`, convolution length `98265`, and normalization
  power `39`. The numerical engine also has a hidden midpoint offset `k/2`;
  use the fail-closed pinned-source generator rather than replacing visible
  `40` literals by hand. A manifest or source rewrite is not an Arb/FLINT
  integral replay.
- The public `python-flint==0.9.0` wheel fails the upstream signed-FFT guard
  and the corrected FLINT patch is not published. The generated evaluator
  therefore splits its sole signed polynomial product into nonnegative
  positive/negative parts and tests that path explicitly. Require full `k=40`
  source-law replay before trusting a `k=39` result. That 97-component replay
  now clears `1/50000`; do not upgrade it to bitwise equivalence with the
  unavailable corrected-FLINT build.
- Consume a completed run with `certify_prime_gap_numerical_receipt` (or
  `scripts/verify_prime_gap_receipt.py`): it reconstructs the exact task
  inventory and final arithmetic and seals a hash binding to the full receipt.
- Long evaluator runs use generated `--checkpoint` / `--resume-log` support.
  Resume accepts only exact task matches and never upgrades partial rows to a
  certificate; final assembly still requires all 97 components.
- Optimize \(k=39\) weights only after
  `prime_gap_quadratic_kernel_spec(dimension=39)` fixes the 77-variable
  signature/degree order and 3,003-entry interval-matrix convention.
  `certify_prime_gap_rounding_reserve` discharges both source-ceiling stages;
  `certify_prime_gap_quadratic_receipt` rejects missing/non-dyadic matrices,
  source/task drift, and descriptor reorderings; matrix screening still
  requires extracted cap/source kernels and every final candidate requires a
  complete direct numerical receipt.

## Next invention

A guessed annihilator whose creative-telescoping certificate and Lean binomial identity
are sealed in one ProofMachine pass.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
