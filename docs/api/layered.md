# Transfer-matrix layered media (02-11)

1-D layered stacks with ABCD matrices. Distinct from
[`geometry.gauge.transfer`](geometry-gauge.md). `unitarity_residual`
is refused outside lossless reciprocal linear media.
`continuum_claim=False` on every certified gap.

G1–G3/G6 are CI-gated and **earned**. G4 inverse-design (10× fewer evals vs
gradient-free, five seeds) is **leftover-recorded** unearned (leftover
#23): no optimizer loop is wired, so the named 5-seed eval-count win
stays `--full`. `stack_matrix` / `certified_band_gap` wall vs
`n_periods` is **leftover-recorded** (leftover #46); the previous
untimed `passed=True` stub is withdrawn. Cost is not in CI
`all_passed`. G5 conservation honesty is
**leftover-recorded** unearned (leftover #27): unstructured 2×2
`|r|^2+|t|^2-1` versus a lossless stack, and `unitarity_residual`
refuses `lossless=False`. No MLP surrogate is wired. The previous
untimed `passed=True` stub is withdrawn. G5 is not in CI `all_passed`.
Status is **shipped**. See theory spec 02-11.

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
