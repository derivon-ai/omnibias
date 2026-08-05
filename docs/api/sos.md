# omnibias-sos

Certified *universal* positivity by optimization. A polynomial `p(x) >= 0` for
**all** `x` iff it has a Sum-of-Squares decomposition `p(x) = z(x)^T Q z(x)` with
a positive-semidefinite Gram matrix `Q`. omnibias-sos finds `Q` with a
floating-point semidefinite program (a *proposer*) and then **proves** the result
with a rigorous interval LDL^T positive-definiteness certificate built on
`omnibias.core.verified` -- the same finite obligation the Mathlib-free Lean
kernel re-checks, so a sealed certificate can earn `theorem_prover_verified`.

- **Global SOS positivity** -- a sound proof that a polynomial is nonnegative
  everywhere.
- **Positivstellensatz** -- constrained positivity `p >= 0` on `{g_i >= 0}` via
  certified SOS multipliers (fixed-degree Putinar form).
- **Auxiliary-functional (background) method** -- a certified `for all data`
  bound on the infinite-time average of an observable for a polynomial ODE /
  Galerkin-truncated system.

!!! note "Soundness, not the solver"
    The SDP solve is a floating-point *proposer* and never touches the proof.
    The proof is the outward-rounded interval LDL^T certificate; a failed
    rational rounding or positive-definite margin returns **inconclusive**, never
    a false positivity claim. The auxiliary-functional bound is a statement about
    a **finite-dimensional / Galerkin-truncated** system, not a continuum PDE
    regularity or global-regularity claim (`unproven_claim = False` on every certificate).

## Public API

::: omnibias.sos
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source

Status: Alpha (`0.1.0a1`).
