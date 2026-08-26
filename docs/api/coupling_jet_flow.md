# Coupling jet-flow (09-06)

A finite coupling stack whose log-det is `sum log|s| + sum log sigma'`
from the founding bias collapse (`delta -> 0`). Temperature collapse
(`beta -> inf`, feasibility) does not appear. Do not conflate the
two.

The inverse is Newton with exact `sigma'`. This is not
`integrate_cnf`. Status is **shipped**. Not ImageNet
generative SOTA. Not CCF stretch.

Homes: `omnibias.core.coupling_flow`,
`omnibias.score.flow.{torch,jax}.jet_flow`.

::: omnibias.core.coupling_flow
    options:
      show_root_heading: false
      heading_level: 3
