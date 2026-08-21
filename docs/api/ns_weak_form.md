# Weak-form NS-adjacent enclosures (07-02)

A strong residual needs `lap u`. The weak form moves one
derivative onto a test function with a closed-form
antiderivative, so the quadrature term drops. Status is
**gated**, not shipped. G1–G6 are CI-gated.

This is a **finite box, finite horizon, finite test space**.
It is not a continuum Navier-Stokes regularity claim. The
parent Clay problem stays an external obligation.

Jets come from the **founding bias collapse** (`delta -> 0`).
Temperature collapse (`beta -> inf`, feasibility) does not
appear. Do not conflate the two.

Every certificate emits a `WidthReport`. A width claim
without a decomposition is refused. `W_repr` often dominates
on a hard flow; G1 exists to make that visible.
`theorem_prover_verified` is not asserted.

Home: `omnibias.pinn.certified.weak_form`. No new package.

## API

::: omnibias.pinn.certified.weak_form
    options:
      show_root_heading: false
      heading_level: 3
