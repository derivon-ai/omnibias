# Jet-PID optimizer (08-10)

PID on the directional restriction `phi(s) = L(theta + s d)`. The
founding bias collapse (`delta -> 0`) supplies the jet. Temperature
collapse (`beta -> inf`, feasibility) does not appear.

`P = phi'(0)`. `I` is the exact fundamental-theorem increment of the
Taylor *model* (`p(h)-p(0)` or a running FTC of applied steps), not a
discrete sum of past gradients. `D = phi''(0)`. Local 1-D restriction.
Not a plant PID, not LQR, not MPC, not a global min. Status is
**shipped**.

Homes: `omnibias.core.control_pid`,
`omnibias.{torch,jax}.optim_pid`.

::: omnibias.core.control_pid
    options:
      show_root_heading: false
      heading_level: 3
