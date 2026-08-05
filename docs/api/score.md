# omnibias-score

Score-based / SDE operators composed from the `omnibias-fields` closed-form
gradient / Hessian primitives: the score `grad log p`, the Ito infinitesimal
generator, and the Fokker-Planck adjoint.

## Ops (torch)

::: omnibias.score.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend (`omnibias.score.jax.ops`) is the bit-identical twin. Validated
on the Ornstein-Uhlenbeck process (`L* p_inf = 0`) with torch/jax parity.
