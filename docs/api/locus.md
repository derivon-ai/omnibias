# Equality locus (01-09) and EqualityLocusLayer (02-12)

The locus is a **constraint manifold**. A level-3 "general closed-form
PDE solver" is **not** claimed. Every solver return carries
`branch` / `condition` / `converged`. Founding `delta -> 0` only.

G1–G5 of 01-09 are CI-gated; G6 is torch/jax parity. 02-12 G1–G3/G5/G6
are CI-gated. G4 Burgers Rankine–Hugoniot is **leftover-recorded**
unearned (leftover #29): `affine_locus` recovers the published-unit
speed, but the noisy-data skill versus contour extraction stays
`--full` (units are not fit from samples). The previous smoke-geometry
`passed=True` stub is withdrawn. G4 is not in CI `all_passed`. Spec
01-09 status is **shipped**. Spec 02-12 status is **shipped**. See
theory specs 01-09 and 02-12.

## Core algebra

::: omnibias.core.locus
    options:
      show_root_heading: false
      heading_level: 3

## Field layer

::: omnibias.fields.locus
    options:
      show_root_heading: false
      heading_level: 3
