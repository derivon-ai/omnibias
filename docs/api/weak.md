# Weak-form VPINN (02-04)

Petrov-Galerkin test functions built from OMBU bumps with closed-form
antiderivatives. Exact integrals hold **only for polynomial coefficient
data on box windows**; otherwise quadrature runs on the coefficient
factor and the path is recorded. Moments of `x^j v` for `j >= 1` scale
the IBP primitive by `1/alpha`. The certified boundary bound from
[mollifier.md](mollifier.md) is **on by default** — analytic bumps are
not compactly supported, so boundary terms are never dropped. SDF
domains stay quadrature-near-boundary and are not claimed.

G1–G5 are CI-gated. G4 conditioning is **earned** on the smoke
artifact: `cond(strong collocation) / cond(weak stiffness)` is far
above `10x` on the named `TestFunctionSpace`. Previous
`g4_is_unit_test` deferral is withdrawn. Status is **gated**, not
shipped. See theory spec 02-04.

## Algebra

::: omnibias.fields.weak._core
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch assembly

::: omnibias.fields.weak.torch
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.fields.weak.jax
    options:
      show_root_heading: false
      heading_level: 3
