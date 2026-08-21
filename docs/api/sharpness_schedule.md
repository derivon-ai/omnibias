# Sharpness-scheduled curvature step (08-06)

A few exact Hessian-vector products give a Ritz `lambda_max` of the
loss Hessian. That value sets cubic `sigma` or a gradient learning
rate. Sharpness is a **step-size** signal, not a generalization
certificate and not CCF stretch.

Status is **gated**, not shipped. G1–G3 are CI-gated. Hutchinson is
not the method. Bias collapse (`delta -> 0`) makes HVPs exact through
`sigma''`. No temperature collapse. See theory spec 08-06.

The named map is `sigma = c * max(ell_k, ell_min)` (or
`lr = c / max(ell_k, ell_min)`). Ritz underestimates `lambda_max`; a
too-small `k` can still diverge. An empty or non-positive schedule
refuses the step.

## Core algebra

::: omnibias.core.sharpness
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch twin

::: omnibias.torch.optim_sharpness
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.optim_sharpness
    options:
      show_root_heading: false
      heading_level: 3
