# Jet-LQR optimizer (08-11)

Finite-horizon discrete LQR on the directional restriction
`phi(s) = L(theta + s d)`. The founding bias collapse (`delta -> 0`)
supplies the jet. Temperature collapse (`beta -> inf`, feasibility)
does not appear.

The plant is the scalar error `e+ = e + phi''(0) u`. The solve is a
**scalar discrete Riccati recursion**, not the algebraic Riccati /
DARE and not the activation Riccati. `R=0`, `N=1`, `Qf=1` recovers
Newton. Local 1-D restriction. Not a plant LQR, not MPC, not a global
min. Status is **shipped**.

Homes: `omnibias.core.control_lqr`,
`omnibias.{torch,jax}.optim_lqr`.

::: omnibias.core.control_lqr
    options:
      show_root_heading: false
      heading_level: 3
