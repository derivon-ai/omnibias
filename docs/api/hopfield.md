# omnibias-hopfield

Modern Hopfield networks and attention-as-an-operator with **closed-form**
log-sum-exp derivatives: the Jacobian is `softmax`, the Hessian is
`beta (diag(p) - p p^T)` -- exact, not autodiff. `modern_hopfield_retrieve` is a
single softmax update of the Ramsauer et al. energy; `attention` is its
`K = V`, multi-query generalization, exposing the well-known duality between
modern Hopfield retrieval and dot-product attention.

## Ops (torch)

::: omnibias.hopfield.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend (`omnibias.hopfield.jax.ops`) is the twin; the closed-form
Jacobian / Hessian are checked against `jax.jacfwd` / `jax.hessian` and against
the torch backend to `rtol=1e-9` in float64.

Status: Alpha (`0.1.0a1`).
