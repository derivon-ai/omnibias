# Inverse design (09-22)

Newton-on-`x` for `f_theta(x) = y` with exact `sigma'` from founding
bias collapse (`delta -> 0`). Temperature collapse (`beta -> inf`,
feasibility) does not appear.

This inverts the net as a map on the input. It is not 08-03 (that
inverts a *layer* for a hidden target). Not a global inverse. Not
CCF stretch. Status is **shipped**.

Homes: `omnibias.core.inverse_design`,
`omnibias.{torch,jax}.optim_inverse`.

::: omnibias.core.inverse_design
    options:
      show_root_heading: false
      heading_level: 3
