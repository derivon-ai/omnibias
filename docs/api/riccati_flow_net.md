# Riccati flow net (09-10)

A layer whose forward is the time-`t` flow of the founding Riccati
ODE `ds/dt = s(1-s)` (logistic) or `d tanh/dt = 1-tanh^2`. Depth is
integration time. This is not a DEQ fixed point and not a CNF density
ODE. The flow itself is not founding bias collapse (`delta -> 0`);
jets of `s(t)` in `s0` may still use the tower. Temperature collapse
(`beta -> inf`, feasibility) does not appear. Do not conflate the
two.

Status is **gated**, not shipped. Not ImageNet. Not CCF stretch.

Homes: `omnibias.core.riccati_flow`,
`omnibias.{torch,jax}.architectures.riccati_flow`.

::: omnibias.core.riccati_flow
    options:
      show_root_heading: false
      heading_level: 3
