# Validated dynamics and orbits (07-06)

Validated integration is limited by wrapping and by the
Jacobian enclosure. For fields in the activation dictionary
the founding bias collapse (`delta -> 0`) supplies `Df`
and the solution Taylor coefficients from one `sigma`
evaluation per order. Temperature collapse (`beta -> inf`,
feasibility) does not appear. Do not conflate the two.

Every run emits a `WidthBudget`. The existing
`lohner_flow` path is bit-unchanged. Status is **gated**,
not shipped. G1–G6 are CI-gated.

Scope is **one field, one initial box, one finite
horizon**. This is not a continuum existence theorem, not
an attractor statement, and not a Navier-Stokes regularity
result. The parent computer-assisted-proof programme stays
an external obligation.

Home: `omnibias.core.verified.jet_flow`. No new package.
`theorem_prover_verified` is not asserted.

## API

::: omnibias.core.verified.jet_flow
    options:
      show_root_heading: false
      heading_level: 3
