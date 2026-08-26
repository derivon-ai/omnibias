# Pack Fisher metric (04-01)

A logistic pack family is a parametric density, so it has a Fisher
information metric. The score integrand is closed form from the
derivative tower; the expectation is a 1-D quadrature (recorded) or a
Monte Carlo fallback.

This is a **metric on pack parameters**, not the scalar
exponential-family Fisher `A''(theta)` in
`omnibias.curvature.glm_fisher`. Status is **shipped**.
G1–G5 are earned. `K >= 3` central finite-difference packs change
sign, so Fisher is refused (inapplicable, not unmeasured).

The `delta -> 0` limit here **is** the founding bias collapse.
Temperature collapse (`beta -> inf`) does not appear.

Home: `omnibias.curvature.information`. No new package. The optional
`as_manifold_spec` wrap needs `omnibias-geometry`.

::: omnibias.curvature.information
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]
