# Depth-causal local jet (08-03)

A layer may take a damped Gauss–Newton step on a **named** local
residual and push a compressed `k`-direction `layer_jet` onward.
Later parameters update first so an earlier layer sees a corrected
downstream map. `compose_jet` is the chain rule, not a skip of it.

Status is **shipped**. G1–G4 are CI-gated. Local GN is
greedy, not a global min, and not CCF stretch. Bias collapse
(`delta -> 0`) supplies the tower. This is not time marching
(`omnibias.pinn.train`). See theory spec 08-03.

## Core algebra

::: omnibias.core.local_jet
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch twin

::: omnibias.torch.train_local
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.train_local
    options:
      show_root_heading: false
      heading_level: 3
