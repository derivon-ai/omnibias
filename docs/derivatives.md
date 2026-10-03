# Choose a derivative API

| Need | PyTorch module | JAX module |
| --- | --- | --- |
| One activation derivative | `omnibias.torch.activations` | `omnibias.jax.activations` |
| Network derivatives along a direction | `omnibias.torch.jet` | `omnibias.jax.jet` |
| All mixed partials through an order | `omnibias.torch.jet_mv` | `omnibias.jax.jet_mv` |
| Direct one-layer or deep Laplacian | `omnibias.torch.laplacian` | `omnibias.jax.laplacian` |
| Field gradient, divergence or Laplacian | `omnibias.fields.torch.ops` | `omnibias.fields.jax.ops` |

## Taylor normalization

`mlp_jet(x0, v, layers, order)` represents the path `x0 + t*v`. Its row `k`
is the directional derivative divided by `k!`. Call `jet_to_tower` for raw
derivatives. Each layer is `(weight, bias, activation)`; weights have shape
`(out_features, in_features)`. `activation=None` is an affine readout.

`mlp_jet_mv(x0, layers, order)` uses one input point of shape `(dim,)` and
stores `D^alpha f / alpha!`. `jet_partials` returns raw partials keyed by
multi-index. Choose directional jets when you need only a few directions;
full mixed jets enumerate all multi-indices through the requested order.

## Mixed derivatives for a heat residual

For coordinates `(t, x)`, the heat equation residual is `u_t - diffusivity*u_xx`:

```python
import torch
from omnibias.torch.jet_mv import jet_partials, mlp_jet_mv

weights = torch.tensor([[0.5, 0.7], [-0.2, 0.3]], requires_grad=True)
bias = torch.tensor([0.1, -0.1], requires_grad=True)
readout = torch.tensor([[0.6, -0.4]], requires_grad=True)
layers = [(weights, bias, "tanh"), (readout, None, None)]
point = torch.tensor([0.2, 0.4])
jet = mlp_jet_mv(point, layers, order=2)
partials = jet_partials(jet, dim=2, order=2)
residual = partials[(1, 0)] - 0.1 * partials[(0, 2)]
residual.square().mean().backward()
assert weights.grad is not None
assert partials[(1, 1)].shape == (1,)
```

## JAX directional jets

The JAX layer tuples and Taylor normalization match PyTorch. Training uses
`jax.grad` with respect to parameters; spatial derivatives come from the jet.

```python
import jax
import jax.numpy as jnp
from omnibias.jax.jet import jet_to_tower, mlp_jet


def residual_loss(weight):
    layers = [(weight, jnp.array([0.1, 0.2]), "tanh"),
              (jnp.array([[0.6, -0.4]]), None, None)]
    tower = jet_to_tower(mlp_jet(jnp.array([0.3]), jnp.ones(1), layers, order=4))
    return jnp.mean((tower[4] - 24.0) ** 2)


weight = jnp.array([[0.5], [-0.3]])
gradient = jax.grad(residual_loss)(weight)
assert gradient.shape == weight.shape
assert bool(jnp.all(jnp.isfinite(gradient)))
```

Enable `JAX_ENABLE_X64=1` before starting Python when testing float64 parity.
For Riccati-class activations such as tanh, directional `mlp_jet` also accepts
`riccati=True`; it changes accumulation order and can change rounding.
Use the [guarantees](guarantees.md) to interpret accuracy and cost.

Bias coalescence (`delta → 0`) yields derivatives. Temperature hardening
(`beta → infinity`) makes a soft partition or relaxation approach a discrete
choice. These are different limits; a hardening bound is not a derivative
error bound.

## A deep Laplacian in 5,000 dimensions

When the residual needs a Laplacian, request that contraction directly.
`deep_field_laplacian` avoids the combinatorial allocation of a full mixed
jet while keeping gradients into the trainable weights:

```python
import torch
from omnibias.torch.laplacian import deep_field_laplacian

dim = 5000
weight = torch.full((4, dim), 0.01, requires_grad=True)
layers = [
    (weight, torch.zeros(4), "tanh"),
    (torch.eye(4), torch.zeros(4), "tanh"),
    (torch.ones(1, 4), None, None),
]
points = torch.full((2, dim), 0.02)
laplacian = deep_field_laplacian(points, layers)
laplacian.square().mean().backward()
assert laplacian.shape == (2, 1)
assert weight.grad is not None
assert bool(torch.all(torch.isfinite(weight.grad)))
```

For a one-hidden-layer field, `neural_field_polylaplacian` evaluates the
repeated Laplacian directly from `sigma^(2k)` and weight norms. For deep
networks, `deep_field_polylaplacian_with_report` distinguishes exact support
enumeration from the sampled fallback. Read the
[dimension and accuracy contract](guarantees.md#laplacians-without-the-mixed-jet-dimension-ceiling)
before choosing between these APIs and full mixed derivatives.
