# Weight-space loss jet (one-layer)

A Riccati hidden layer `f = b + c · σ(W x + β)` is linear in the
step `s` at every preactivation once a weight direction `d` is fixed.
Founding bias collapse (`delta -> 0`) supplies `σ^(n)`. Leibniz
assembly then returns `φ^(k)(0)` for the MSE restriction
`φ(s) = L(θ + s d)` without nested reverse-mode and without a full
`d h / d θ`.

Status is **shipped** as a one-layer primitive. Faà di Bruno is the
chain rule. Not a global min, not a deep-net closed-form loss in every
weight, and not CCF stretch.

Homes: `omnibias.core.weight_loss_jet`,
`omnibias.{torch,jax}.weight_loss_jet`.

::: omnibias.core.weight_loss_jet
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch twin

::: omnibias.torch.weight_loss_jet
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.weight_loss_jet
    options:
      show_root_heading: false
      heading_level: 3
