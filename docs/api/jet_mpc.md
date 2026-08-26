# Jet-MPC optimizer (08-12)

Receding-horizon control on `phi(s) = L(theta + s d)`. The founding
bias collapse (`delta -> 0`) supplies the jet. Temperature collapse
(`beta -> inf`, feasibility) does not appear.

The planned law is the first control of 08-11 LQR. The applied step
is that control boxed by `u_max` and an optional Lagrange trust
radius. Only `u_0` is applied. Not plant MPC, not a general
horizon-QP, not a global min. Status is **shipped**.

Homes: `omnibias.core.control_mpc`,
`omnibias.{torch,jax}.optim_mpc`.

::: omnibias.core.control_mpc
    options:
      show_root_heading: false
      heading_level: 3
