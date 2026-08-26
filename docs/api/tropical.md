# Tropical homotopy (01-08)

A differentiable path between the log and max-plus semirings. Reuses
`MaxPlus` / `logsumexp_beta` / `logsumexp_gap_bound`; it does not fork
them. `beta -> inf` is temperature collapse, not founding `delta -> 0`.
Large `(n, D)` inputs are refused. Sound gap, not P vs NP.

G1 gap soundness is CI-gated. G2 subdivision vs the 01-03 sampler and
G3 jet derivatives are CI-gated. G4 path-following is **earned**
(leftover #32 closed): `path_follow` matches `tropical_anneal_descent`
on the surrounding-exponent family at `2x` fewer evals, five seeds,
with the certified gap on both arms. The driver duck-types
`AnnealSchedule` (struct cannot import discrete). Sampled
`dual_subdivision` wall vs `n`/`D` is **leftover-recorded** (leftover
#19; API refuses `n>10` or `D>3`). Cost is not in CI `all_passed`.
Status is **shipped**. See theory spec 01-08.

## Algebra

::: omnibias.struct._core.tropical
    options:
      show_root_heading: false
      heading_level: 3
