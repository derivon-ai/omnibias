# q-OMBU / timescale hybrid (09-15)

A stack of Jackson `q_derivative` or Hilger `delta_derivative` layers
with ordinary OMBU `sigma` cells. The named limits `q -> 1` and
`mu -> 0` recover ordinary derivatives. Those limits are **not**
founding bias collapse (`delta -> 0`) and **not** temperature
collapse (`beta -> inf`, feasibility). Ordinary `sigma` cells still
use founding bias collapse for `sigma^(n)`. Do not conflate the two.

`q == 1` is a removable singularity: use `q_ombu_limit`, never divide
by zero.

Status is **gated**, not shipped. Not a continuum PDE. Not CCF
stretch. Not NS.

Homes: `omnibias.qcalculus._core.hybrid`,
`omnibias.qcalculus.{torch,jax}.hybrid`,
`omnibias.timescale._core.hybrid`,
`omnibias.timescale.{torch,jax}.hybrid`.

::: omnibias.qcalculus._core.hybrid
    options:
      show_root_heading: false
      heading_level: 3
