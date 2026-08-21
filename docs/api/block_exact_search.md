# Block / coordinate exact search (08-07)

Along one structured block (last linear layer, one OMBU bias, or one
arrangement normal) the directional loss is a 03-12 jet. Last-layer
least squares is exactly quadratic, so order 2 with remainder bound 0
is exact. `verify=True` is the never-worse backstop.

Status is **gated**, not shipped. G1–G4 are CI-gated. This is a
coordinate / block sweep, not a global solver and not CCF stretch.
Bias collapse (`delta -> 0`) supplies the tower. Arrangement `beta` is
caller-owned. See theory spec 08-07.

## Core algebra

::: omnibias.core.block_search
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch twin

::: omnibias.torch.optim_block_search
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.optim_block_search
    options:
      show_root_heading: false
      heading_level: 3
