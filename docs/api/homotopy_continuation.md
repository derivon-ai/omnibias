# Homotopy continuation (09-20)

A path of finite residual maps `H(theta, tau)`. Each knot takes a
Newton trial and keeps it only if the 08-04 unique-zero ball is
nonempty. An empty ball is a halt, not a forged root.

Jets / `sigma''` on a nest use founding bias collapse (`delta -> 0`).
Temperature collapse (`beta -> inf`, feasibility) does not appear.
Do not conflate the two.

Status is **gated**, not shipped. Not a rewrite of one-step 08-04.
Not CCF stretch. Not a continuum PDE.

Homes: `omnibias.core.homotopy`,
`omnibias.{torch,jax}.optim_homotopy`.

::: omnibias.core.homotopy
    options:
      show_root_heading: false
      heading_level: 3
