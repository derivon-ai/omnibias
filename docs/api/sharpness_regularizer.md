# Sharpness regularizer (09-23)

`L_sharp = L + mu * ritz_lambda_max(H)` (or `Tr H`) from exact HVPs.
Founding bias collapse (`delta -> 0`) supplies `sigma''`. Temperature
collapse (`beta -> inf`, feasibility) does not appear.

This is **not** 08-06. That spec only schedules cubic `sigma` / lr.
Ritz underestimates `lambda_max`. Status is **gated**, not shipped.
Not ImageNet SAM. Not CCF stretch.

Homes: `omnibias.core.sharp_loss`,
`omnibias.{torch,jax}.optim_sharp_loss`.

::: omnibias.core.sharp_loss
    options:
      show_root_heading: false
      heading_level: 3
