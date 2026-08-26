# Collapse-Net (09-11)

Train on a lattice with a founding multi-bias stencil; infer by
founding bias collapse (`delta -> 0`) to the named map
`sigma^(K-1)` (or `sin^(order)`). The continuum limit is that named
map, not a hope and not a continuum PDE. Temperature collapse
(`beta -> inf`, feasibility) does not appear. Do not conflate the
two.

Status is **shipped**. Not CCF stretch.

Homes: `omnibias.core.collapse_net`,
`omnibias.{torch,jax}.architectures.collapse_net`,
`omnibias.difference._core.collapse_net`.

::: omnibias.core.collapse_net
    options:
      show_root_heading: false
      heading_level: 3
