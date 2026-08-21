# Characteristic-Net (09-08)

A 1-D conservation layer that transports `u` along characteristics of
a learned `v` using a closed-form time integral (window / FTC
panels). When two feet map to one `x` the layer returns
`crossed=True` and does not invent a unique value. The founding bias
collapse (`delta -> 0`) supplies jets of `v`. Temperature collapse
(`beta -> inf`, feasibility) does not appear. Do not conflate the
two.

Status is **gated**, not shipped. Not a rewrite of the 02-13
Cole–Hopf / Miura maps. Not 3-D NS. Not CCF stretch. Not a
shock-capturing theorem.

Homes: `omnibias.pinn.characteristic`,
`omnibias.pinn.{torch,jax}.characteristic`.

::: omnibias.pinn.characteristic
    options:
      show_root_heading: false
      heading_level: 3
