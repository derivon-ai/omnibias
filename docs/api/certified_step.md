# Certified step (08-09)

A parameter update is **legal** only when a sound
`lipschitz_bound` or `taylor_output_bounds` enclosure of a named
input-output property stays inside a declared cap. Empty or exploding
enclosures reject the step; that is not a robustness claim.

Status is **shipped**. G1–G3 are CI-gated. Distinct from
08-04 (unique zero of a residual map). Not a global min and not CCF
stretch. Lipschitz and Taylor-model paths use the founding bias
collapse (`delta -> 0`) `sigma'` tower. See theory spec 08-09.

Optimizers propose `theta'`. This policy lives in
`omnibias.verify.train_step` and is **not** imported by the T1
`omnibias.{torch,jax}` packages.

## Core policy

::: omnibias.verify.train_step
    options:
      show_root_heading: false
      heading_level: 3
