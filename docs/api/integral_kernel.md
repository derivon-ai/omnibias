# Integral-kernel operator (09-14)

A volumetric 1-D DeepONet kernel whose cell **is** the OMBU
`integral` role `S(z + b_hi) - S(z + b_lo)`. Not a surface BEM-Net
(02-06) and not a Fourier multiplier.

The window is the knob. Founding bias collapse (`delta -> 0`) of
that window recovers a `sigma` kernel and is not the default.
Temperature collapse (`beta -> inf`, feasibility) does not appear.
Do not conflate the two.

Status is **shipped**. Not FNO SOTA. Not CCF stretch.
Not NS.

Homes: `omnibias.core.integral_kernel`,
`omnibias.{torch,jax}.architectures.integral_kernel`,
`omnibias.pinn.operator` and
`omnibias.pinn.operator.{torch,jax}.integral_kernel`.

::: omnibias.core.integral_kernel
    options:
      show_root_heading: false
      heading_level: 3
