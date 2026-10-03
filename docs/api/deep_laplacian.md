# Deep-network Laplacian fast lane

The one-layer Laplacian (`omnibias.jax.laplacian` / `omnibias.torch.laplacian`
`neural_field_*`) is cheap for a structural reason: in
`f(x) = b + sum_h c_h sigma(W_h . x + beta_h)` the pre-activation
`z = W x + b` is affine in `x`, so every higher derivative of `z` vanishes and
Faà di Bruno collapses to one term. A deep network has no such collapse, so
the old fallback was the full multivariate jet
(`omnibias.torch.jet_mv.mlp_jet_mv` / `omnibias.jax.jet_mv.mlp_jet_mv`), which
materialises **every** mixed partial to the requested total order --
`comb(D + 2k, D)` of them -- and refuses past
`omnibias.core.multi_index.MAX_MULTI_INDICES` well before `D` reaches the
thousands.

`deep_field_laplacian` / `deep_field_polylaplacian` remove that ceiling by
never materialising the full jet. They dispatch through
`omnibias.core.contraction.select_mode` onto three tiers:

- **Tier A (forward-Laplacian recursion, `k = 1`, any `D`).** Per layer, with
  `u = W a + b` then `a' = sigma(u)`, the recursion carries the Jacobian `J`
  of the running activation with respect to the *original* input and one
  scalar Laplacian `L` per unit -- `O(B . H . D)` memory, the same order as
  the forward pass, with **no combinatorial term at all**. Every `sigma'`,
  `sigma''` comes from the closed-form fastpath, not autodiff through the
  activation.
- **Tier B (support-grouped local jets, `k >= 2`, design fits the budget).**
  The multinomial expansion `Delta^k = (sum_i d_i^2)^k` is evaluated *exactly*
  by grouping terms by support set, restricting the first affine layer per
  support, and reading the diagonal `(2,2,...,2)` row off a local multivariate
  jet of dimension `|S| <= k`. There is no sphere, no directions, and no
  quadrature error.
- **Tier C (unbiased sphere estimator, `k >= 2`, any `D`).** The spherical
  identity is *estimated* with sampled directions once the exact support
  design would exceed `budget`; the result is exact only in expectation.
  Use :func:`deep_field_polylaplacian_with_report` for the resolved tier,
  sample standard error, and a :class:`~omnibias.core.verified.sampled.ConcentrationReport`
  from :func:`~omnibias.core.verified.sampled.hoeffding_enclosure`.

Status is **shipped** (09-33). G1-G5
are gated in `benchmarks/deep_laplacian_scaling.py` and CI-collected via
`packages/omnibias-jax/tests/test_deep_laplacian.py` /
`packages/omnibias-torch/tests/test_deep_laplacian.py` /
`tests/test_deep_laplacian_parity.py`. Bias collapse (`delta -> 0`) supplies
the `sigma'` / `sigma''` fastpath inside every tier; this is not temperature
collapse. Torch-vs-JAX agreement is measured at a tight float64 tolerance
(`~1e-15` absolute on the configs tested), **not** literal bit-identity: the
shared pure-Python fastpath polynomial evaluation is bit-exact between
backends in isolation, but the recursion's intermediate `tensordot` / `sum`
steps pick up a 1-2 ULP gap between backends even at trivial shapes. State
the capability precisely: exact and ceiling-free for the Laplacian,
exact-or-enclosed at any dimension for `Delta^k` -- never "O(1) at arbitrary
order."

The multivariate jet path (`mlp_jet`, `mlp_jet_mv`) is still the right tool
for a general mixed partial that is not a Laplacian / poly-Laplacian
contraction, and for architectures that are not a linear chain of
`(W, b, activation)` layers.

## The ceiling, demonstrated

```python
import torch

from omnibias.core.multi_index import index_position, multi_index_factorial
from omnibias.torch.jet_mv import mlp_jet_mv
from omnibias.torch.laplacian import deep_field_laplacian

torch.set_default_dtype(torch.float64)

dim, hidden, depth = 4, 6, 3
gen = torch.Generator().manual_seed(0)
layers = []
for i in range(depth + 1):
    in_dim = dim if i == 0 else hidden
    out_dim = 1 if i == depth else hidden
    W = torch.randn(out_dim, in_dim, generator=gen) / in_dim**0.5
    b = torch.randn(out_dim, generator=gen) * 0.1
    layers.append((W, b, None if i == depth else "tanh"))

x = torch.randn(dim, generator=gen) * 0.3

# Tier A: exact, no ceiling.
lap_fast = deep_field_laplacian(x, layers)

# The old path: read the order-2 rows off the full multivariate jet.
jet = mlp_jet_mv(x, layers, 2)
pos = index_position(dim, 2)
lap_oracle = sum(
    jet[pos[tuple(2 if j == i else 0 for j in range(dim))]] * multi_index_factorial(
        tuple(2 if j == i else 0 for j in range(dim))
    )
    for i in range(dim)
)
assert torch.allclose(lap_fast[0], lap_oracle[0], atol=1e-9)
```

<!-- docs-test: raises=ValueError -->
```python
import torch

from omnibias.torch.jet_mv import mlp_jet_mv

torch.set_default_dtype(torch.float64)

# The same D=5000 network that deep_field_laplacian handles in O(B.H.D)
# blows the multivariate-jet ceiling immediately.
dim, hidden = 5000, 4
W0 = torch.randn(hidden, dim) / dim**0.5
b0 = torch.randn(hidden) * 0.1
W1 = torch.randn(1, hidden) / hidden**0.5
b1 = torch.randn(1) * 0.1
layers = [(W0, b0, "tanh"), (W1, b1, None)]
x = torch.randn(dim) * 0.1

mlp_jet_mv(x, layers, 2)  # raises ValueError: above the multi-index budget
```

## Core algebra

::: omnibias.core.contraction
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch twin

::: omnibias.torch.laplacian
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - deep_field_value_grad_laplacian
        - deep_field_laplacian
                - deep_field_polylaplacian
                - deep_field_polylaplacian_with_report
                - restrict_first_layer

## JAX twin

::: omnibias.jax.laplacian
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - deep_field_value_grad_laplacian
        - deep_field_laplacian
                - deep_field_polylaplacian
                - deep_field_polylaplacian_with_report
                - restrict_first_layer
