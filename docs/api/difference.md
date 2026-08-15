# omnibias-difference

The founding `delta -> 0` register: discrete calculus and analytic combinatorics
read off the closed-form derivative towers. This is *the* bias collapse the
library is named for — `K` biases on a difference stencil coalesce as
`delta -> 0` and the finite difference *becomes* the derivative `sigma^(K-1)`,
which the closed-form tower evaluates exactly (no `1/delta^(K-1)` cancellation).

- **Certified finite-difference -> derivative** — the rigorous interval-tower
  enclosure of `sigma^(n)(z)`, the numerical finite-difference estimate, and a
  *certified* Taylor-remainder bound proving the estimate collapses into the
  enclosure as `delta -> 0`.
- **Umbral / Sheffer calculus** (`umbral`) — the forward-difference operator, Newton
  interpolation, the monomial <-> falling-factorial (Stirling) change of basis, and the
  full Sheffer surface: `sheffer_sequence` / `associated_sequence` *generation* from a
  `(g, f)` pair, `appell_sequence`, `umbral_composition`, and the operator layer
  (`shift_polynomial` `E^a`, `delta_operator_apply` `Q = f(D)`, `pincherle_derivative`
  `f'(D)`, `umbral_functional`). All exact `Fraction` arithmetic, gathered in the
  `omnibias.difference.umbral` namespace and flat-re-exported for convenience.
- **Asymptotic-coefficient reading** — Stirling (off the Bell tower), Bernoulli
  (off the `tanh` tower) and Euler (off the `sech` tower) numbers, exact as
  `int` / `Fraction`, plus their leading asymptotics.
- **Proof-carrying analytic combinatorics** (`generating`) — EGF/OGF algebra,
  singularity/saddle-point asymptotics, and **certified** asymptotic *enclosures*
  (`bell_dobinski_enclosure`, `catalan_asymptotic`, certified Bernoulli/Euler
  enclosures) that close the float-only "no error bars" gap, plus measured
  exact-vs-asymptotic fallback thresholds.
- **Singularity analysis / Padé / Sheffer–Riordan** — the Flajolet–Sedgewick
  `transfer_theorem` mapping an OGF singularity to `[zⁿ]` coefficient asymptotics with a
  **certified** error term; exact-rational Padé approximants (`pade_approximant`) and
  Thiele continued fractions (`thiele_interpolation` / `thiele_evaluate`) with certified
  remainders; and `sheffer_classify` plus the Riordan-array group product / inverse
  (`riordan_array`) and `connection_constants` (the Fundamental Theorem of Riordan
  Arrays). Padé / Riordan are `closed-form` (exact rational); the singularity asymptotics
  are `numerical` (certified). Baselines: raw truncated series (Padé wins) and float
  coefficient asymptotics (the certified enclosures win).
- **Lean-checkable special-number identities** (`identities`) — Bernoulli /
  Euler recurrences and `ζ(1−2m)` regressions emitted as finite *rational*
  obligations that earn `theorem_prover_verified` **only** on a genuine `lake`
  pass (see [certificates & the Lean loop](core.md)).
- **Bit-identical torch / jax twins** for the finite-difference stencil operator.

This package is also the home of an all-tiers **data-driven refinement program**:
each capability is an instrumented probe (grid + random soundness, an `mpmath`
oracle, `K ≥ 8` seeds, a named baseline) that surfaces gaps/flaws/bugs and locks
the fix in with a regression test. The umbrella smoke is
`docs/examples/difference_validate.py`; the shared probe utilities live in
`omnibias.difference.validation`.

!!! note "Honest registers, not conflated senses"
    Extraction is **closed-form** (the towers + exact coefficients); the
    finite-difference estimate and the mpmath comparison are **numerical**. This
    is the `delta -> 0` *derivative* collapse — **not** the `beta -> inf`
    feasibility penalty of `omnibias-convex` / `-control` / `-routing`. Same
    word, opposite limit; see [`docs/theory.md`](../theory.md).

## Public API

::: omnibias.difference
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source

## Irregular / Birkhoff stencils

Exact rational weights for arbitrary node and per-node order sets (theory
01-04). Nodes in `StencilRequest` are dimensionless `c_i` (units of the
scale `h`). Scale-free weights satisfy `A_{i,p} = a_{i,p} h^{q-p}` (the
spec's `A = h^q a` only closes if rewritten this way). Order is asymptotic
in `h`. `is_poised_exact` is the exact-`Q` oracle; `omnibias.core.multipack.is_poised`
stays the numerical rank test.

::: omnibias.difference._core.irregular
    options:
      show_root_heading: false
      heading_level: 3

## Refinement-program validation harness

Shared, pure-Python probe utilities (`enclosure_soundness`, an `mpmath` oracle
adapter, `baseline_compare`, and a `Finding` / `FindingsLedger` JSON writer) used
by every workstream's probe and by the umbrella smoke.

::: omnibias.difference.validation
    options:
      show_root_heading: false
      heading_level: 3

Status: Alpha (`0.1.0a1`).
