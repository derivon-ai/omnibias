# Transfer-matrix layered media (02-11)

1-D layered stacks with ABCD matrices. Distinct from
[`geometry.gauge.transfer`](geometry-gauge.md). `unitarity_residual`
is refused outside lossless reciprocal linear media.
`continuum_claim=False` on every certified gap.

G1–G3/G6 are CI-gated. G4 inverse-design is `--full`. G5 MLP
conservation honesty is on smoke. Status is **gated**, not shipped.
See theory spec 02-11.

## Core algebra

::: omnibias.core.transfer
    options:
      show_root_heading: false
      heading_level: 3

## PINN twins

::: omnibias.pinn.layered
    options:
      show_root_heading: false
      heading_level: 3
