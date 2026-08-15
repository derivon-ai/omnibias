# Weak-form VPINN (02-04)

Petrov-Galerkin test functions built from OMBU bumps with closed-form
antiderivatives. Exact integrals hold **only for polynomial coefficient
data on box windows**; otherwise quadrature runs on the coefficient
factor and the path is recorded. The certified boundary bound from
[mollifier.md](mollifier.md) is **on by default** — analytic bumps are
not compactly supported, so boundary terms are never dropped. SDF
domains stay quadrature-near-boundary and are not claimed.

G1/G2/G3/G5 are CI-gated. G4 (condition number) is a unit test on the
discrete matrix. Status is **gated**, not shipped. See theory spec
02-04.

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
