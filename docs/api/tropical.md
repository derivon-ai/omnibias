# Tropical homotopy (01-08)

A differentiable path between the log and max-plus semirings. Reuses
`MaxPlus` / `logsumexp_beta` / `logsumexp_gap_bound`; it does not fork
them. `beta -> inf` is temperature collapse, not founding `delta -> 0`.
Large `(n, D)` inputs are refused. Sound gap, not P vs NP.

G1 gap soundness is CI-gated. G2 subdivision vs the 01-03 sampler and
G3 jet derivatives are CI-gated. G4 path-following is `--full` only.
Status is **gated**, not shipped. See theory spec 01-08.

## Algebra

::: omnibias.struct._core.tropical
    options:
      show_root_heading: false
      heading_level: 3
