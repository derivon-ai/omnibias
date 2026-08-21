# Scale flow and coarse-graining (03-07)

The tempering scale `alpha` is a third axis. `alpha_c -> inf` is
neither collapse. The founding bias collapse is `delta -> 0`.
Temperature collapse is `beta -> inf` (feasibility). Do not
conflate the two.

The exact law `sigma_alpha^(n)(u) = alpha^n sigma^(n)(alpha u)` is
the existing `tempered` combinator. Linear coarse-graining is
exact (Galerkin restriction). Nonlinear flow is a recorded
truncation: `FlowSystem` refuses to report an exponent without
`truncation_order`. Critical exponents from a truncated flow are
approximations; G6 records a three-order study rather than a
three-digit claim.

Home: `omnibias.core.scale` plus `omnibias.fields.scale`. No new
package. Status is **gated**, not shipped. G1–G6 are CI-gated.

## API

::: omnibias.core.scale
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.fields.scale
    options:
      show_root_heading: false
      heading_level: 3
