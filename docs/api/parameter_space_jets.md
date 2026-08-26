# Parameter-space jets (09-27)

Treat a PDE parameter `μ` as a jet coordinate so one pass yields
`∂^{α,β} u / ∂x^α ∂μ^β`. founding bias collapse (`delta -> 0`)
supplies `sigma^(n)` along `z = (x, μ)`. Temperature collapse
(`beta -> inf`, feasibility) does not appear.

`closed_form` is refused unless `μ` enters the jet trunk. A dense
`pde_params` encoder is autodiff. Not a ParamPINN package. Not NS.
Not CCF stretch. Status is **shipped**.

Homes: `omnibias.core.parameter_jets`,
`omnibias.pinn.operator._core.parameter_jets`,
`omnibias.pinn.operator.{torch,jax}.parameter_jets`.

::: omnibias.core.parameter_jets
    options:
      show_root_heading: false
      heading_level: 3
