# Mollifier calculus (01-05)

A collapsing pack is a mollifier: analytic bases have **certified
exponential tails**, not compact support, and higher-order
(moment-annihilating) kernels take **negative** values. Do not call a
`tanh` bump compactly supported.

`tail_bound` returns an `Interval`. G1–G4 are CI-gated. G4 is the 02-04
weak residual of a known 1-D Poisson versus matched-cost Gauss (two
window edges), not a compact-support claim. Status is **gated**, not
shipped. See theory spec 01-05.

## Core algebra

::: omnibias.core.mollifier
    options:
      show_root_heading: false
      heading_level: 3
