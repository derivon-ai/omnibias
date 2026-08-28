# omnibias-jax

JAX backend for omnibias.

## Top-level API

::: omnibias.jax
    options:
      show_root_heading: false
      heading_level: 3

## Activation registry

::: omnibias.jax.activations
    options:
      show_root_heading: false
      heading_level: 3

## Closed-form integral transforms

Bit-identical twin of
[`omnibias.torch.transforms`](torch.md#closed-form-integral-transforms):
closed-form Laplace / Fourier / Mellin transforms of the activations whose
transform is itself elementary, resolved through the shared identity table in
[`omnibias.core.transforms`](core.md#integral-transform-identities).

Every kernel is a pure array expression and is safe under `jit`, `grad` and
`vmap`. Two JAX-specific points: `fermi_dirac_mellin` *is* differentiable here
(`jax.scipy.special.zeta` defines a gradient rule where `torch.special.zeta`
does not), and its `Re(s) > 1` scope check reads a concrete value, so it is
skipped under tracing -- the same trade-off `omnibias.measure.jax.integraleq`
makes for its solvability check. Validate in eager mode; the arithmetic itself
traces either way.

`TransformBlock` is the functional counterpart of the torch module: `init()`
hands back a parameter pytree and `apply(params, x)` evaluates
`T[sigma](scale * x + shift)`, so it composes with any JAX optimizer without a
module system.

::: omnibias.jax.transforms
    options:
      show_root_heading: false
      heading_level: 3

## Closed-form Laplacian primitives

::: omnibias.jax.laplacian
    options:
      show_root_heading: false
      heading_level: 3

## Faà di Bruno jets

`compose_jet` takes an **arbitrary** derivative tower, so it performs truncated
power-series composition and is **cubic** in the truncation order `N`; it skips
the products that vanish because `v = u - u_0` has valuation 1, which is a
measured ~3x constant factor at bit-for-bit identical values, not a better
exponent. No `O(N^2)` claim is made for it. `compose_jet_riccati` *is*
`O(deg(P) * N^2)`, because a Riccati-class activation satisfies
`sigma' = P(sigma)` (the `riccati_polynomial` on the `ActivationSpec`) and the
chain rule then closes on the activation itself; it needs only `sigma(u_0)`,
never the tower, so it also reaches past the order caps of the `tan` / `cot` /
`coth` fastpath kernels. Opt in with `riccati=True` on `layer_jet` / `mlp_jet`;
the default stays the general kernel so the pinned goldens keep their meaning.
Cost and exactness gates: [`benchmarks/jet_compose_cost.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/jet_compose_cost.py).

::: omnibias.jax.jet
    options:
      show_root_heading: false
      heading_level: 3

## Multivariate (multi-index) jets

::: omnibias.jax.jet_mv
    options:
      show_root_heading: false
      heading_level: 3

## Born-Oppenheimer derivatives

::: omnibias.jax.bo_derivatives
    options:
      show_root_heading: false
      heading_level: 3

## Architectures

::: omnibias.jax.architectures
    options:
      show_root_heading: false
      heading_level: 3

## Optimisers (Gauss-Newton / energy natural gradient, natural / Riemannian gradient)

Alongside the Gauss-Newton / Levenberg-Marquardt core, `natural_gradient_direction` solves
the metric-preconditioned system `(M + damping I) delta = g` on any dense `(P, P)` metric and
`natural_gradient_step` applies the update `theta - lr (M + damping I)^{-1} g` -- the
bit-identical functional twin of the torch `NaturalGradient`. Two closed-form metrics plug in:
the Gauss-Newton **Fisher** `(1/N) J^T J` (`gauss_newton_fisher`; Newton on a residual linear
in `theta`) and the **geometry pullback** `g = J^T h J`
(`omnibias.geometry.jax.ops.pullback_metric`).
The 08-01 recommended stack lives in `omnibias.jax.train_stack`; see
[train_stack.md](train_stack.md).
Closed-form one-layer weight-space loss jets live in
`omnibias.jax.weight_loss_jet`; see [weight_loss_jet.md](weight_loss_jet.md).
Exact jet line search (theory 03-12) lives in `omnibias.jax.line_search`
and is re-exported here; see [line_search.md](line_search.md).
Adaptive pack refinement (theory 03-13) lives in `omnibias.jax.refine`;
see [refine.md](refine.md).
Composed-curvature joint Newton (theory 08-02) lives in
`omnibias.jax.optim_composed` and is re-exported here; see
[composed_curvature.md](composed_curvature.md).
Sharpness-scheduled step (theory 08-06) lives in
`omnibias.jax.optim_sharpness` and is re-exported here; see
[sharpness_schedule.md](sharpness_schedule.md).
Block exact search (theory 08-07) lives in
`omnibias.jax.optim_block_search` and is re-exported here; see
[block_exact_search.md](block_exact_search.md).
Depth-causal local jet (theory 08-03) lives in
`omnibias.jax.train_local`; see [local_jet.md](local_jet.md).
Implicit / DEQ Newton (theory 08-08) lives in
`omnibias.jax.implicit`; see [implicit.md](implicit.md). The
default solver loop is `lax.while_loop`.
`linf_minimax_step` is the L-infinity sibling of Gauss-Newton: a linearized
epigraph step from `linearized_linf_direction` that accepts only when the true
`max|r|` does not increase. It is a generic trainer step for a toy residual,
not a CCF champion retraining path.

```python
import jax
import jax.numpy as jnp
from omnibias.jax.optim import linf_minimax_step

jax.config.update("jax_enable_x64", True)

def residual(theta):
    return jnp.stack([theta[0] - 1.0, 0.5 * theta[0]])

_params, _box, accepted, maxabs = linf_minimax_step(
    residual, jnp.array([0.0]), 1.0
)
assert accepted
assert maxabs < 1.0
```

::: omnibias.jax.optim
    options:
      show_root_heading: false
      heading_level: 3
