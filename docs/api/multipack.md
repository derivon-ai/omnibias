# Multi-pack Birkhoff collapse

Heterogeneous multi-pack units evaluate a scattered Birkhoff sample of
``sigma`` along one transverse coordinate:

```text
F(z) = sum_g c_g * sigma^(n_g)(z + mu_g)
```

Each pack of size ``K = order + 1`` at mean ``mu`` is the founding bias
collapse ``delta -> 0`` of a K-bias stencil onto ``sigma^(order)(z + mu)``.
For a Riccati base this costs **one activation evaluation per distinct
mean**. Representation claims ("can match any prescribed Birkhoff data")
require a poised support; the unit always computes a well-defined
functional. See theory spec 01-01.

## Core algebra

::: omnibias.core.multipack
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch unit

::: omnibias.torch.multipack
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.multipack
    options:
      show_root_heading: false
      heading_level: 3
