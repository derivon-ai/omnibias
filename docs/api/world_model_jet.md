# World-model-as-jet (09-25)

Predict the next N-jet of a named ODE and plan on the Taylor
polynomial plus a Lohner remainder. founding bias collapse
(`delta -> 0`) supplies spatial jets of a learned `f`. Temperature
collapse (`beta -> inf`, feasibility) does not appear.

Finite-time enclosure on a named ODE. Not Navier–Stokes global
regularity. Not CCF stretch. Not an RL SOTA world model. Status is
**shipped**. The Lohner path imports neither torch nor jax.

Homes: `omnibias.core.jet_world`,
`omnibias.dynamics._core.jet_world`.

::: omnibias.core.jet_world
    options:
      show_root_heading: false
      heading_level: 3
