# Plant PID layer (09-29)

A SISO controller for `y(t) = sigma(alpha t + beta)`. The founding
bias collapse (`delta -> 0`) supplies `sigma'`. The `integral` role
supplies `S` with `S'=sigma`. Temperature collapse (`beta -> inf`,
feasibility) does not appear.

`I` is the exact FTC of the tracking error, not a discrete sum. `D`
is `-alpha sigma'`, not a finite difference. Not the 08-10 jet-PID
trainer. Not cruise-control SOTA. Status is **shipped**.

Homes: `omnibias.core.pid_layer`,
`omnibias.{torch,jax}.plant_pid`.

::: omnibias.core.pid_layer
    options:
      show_root_heading: false
      heading_level: 3
