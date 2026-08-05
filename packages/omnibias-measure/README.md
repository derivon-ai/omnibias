# omnibias-measure

**Status: Alpha (0.1.0a1).**

Autograd-native **measure-theoretic integration** for omnibias. This is the
third of omnibias's three distinct senses of "integral" (see
[`docs/operator-surface.md`](../../docs/operator-surface.md)):

1. the **closed-form activation antiderivative** window `S(z + b_hi) - S(z + b_lo)`
   (`omnibias.torch.OperatorBlock(op="integral")`),
2. **field domain quadrature** `sum_q w_q u(x_q)` (`omnibias.fields.integrate`),
3. the **measure integral** `int f dmu` against an abstract measure -- *this
   package*.

## What it provides

- `Measure` -- a discrete measure `sum_i w_i delta_{x_i}` on `R^d`
  (nodes + weights) generalizing `omnibias.fields`'s `QuadratureSpec`, with the
  measure algebra you need: `pushforward` (change of variables `T_# mu`),
  `product` (`mu (x) nu`), `reweight` (Radon-Nikodym / importance reweighting),
  and `normalize`. Constructors reuse the field quadrature rules:
  `lebesgue` (Gauss-Legendre box), `gaussian` (Gauss-Hermite), `uniform_mc`
  (Monte-Carlo), plus `empirical`, `counting`, `dirac`.
- `lebesgue_integral(f, measure)` -- the measure integral `int f dmu`.
- `importance_expectation(f, measure, log_weight)` -- (self-normalized)
  importance-sampling expectation from proposal samples.
- `layer_cake_integral(f, measure)` -- the distribution-function formula
  `int f dmu = int_0^inf mu({f>t}) dt` with a soft superlevel indicator
  `sigmoid(beta (f - t))` (exact, stable derivatives from the omnibias sigmoid
  tower), so you can differentiate through a thresholded integrand.
- `simple_function_approx(f, measure, levels)` -- the textbook monotone
  from-below simple-function construction `int f dmu ~= sum_k level_k mu(band_k)`.

## Backends and layers

Everything ships as **bit-identical torch + jax twins on a pure-Python `_core`**
numpy reference (keras is intentionally out of scope for extension ops). The
measure's `nodes` are fixed positions; `weights` and the level-set softness
`beta` can be passed as tensors so gradients flow through them.

Trainable layer wrappers drop an integral into a network:

- torch (`nn.Module`): `LebesgueIntegral`, `ExpectationLayer`, `LayerCakeIntegral`
  (with a learnable `beta = exp(log_beta)` and optionally learnable weights).
- jax (functional / equinox-style pytrees): the same three, with array leaves
  (`nodes`, `weight`, `log_beta`) that `jax.grad` / `optax` train directly.

```python
import torch
from omnibias.measure._core import measure as m
from omnibias.measure.torch import LebesgueIntegral, layer_cake_integral

mu = m.gaussian(32)                       # N(0, 1) probability measure
E_x2 = layer_cake_integral(lambda x: x[:, 0] ** 2, mu, signed=False)  # ~ 1.0

layer = LebesgueIntegral(mu, learnable_weights=True)  # trainable pooling
val = layer(lambda x: torch.tanh(x[:, 0]))            # int tanh(x) dmu
```

## Install

```bash
pip install "omnibias-measure[torch]"   # or [jax], or [all]
```

Depends on `omnibias-core` and `omnibias-fields` (for the reused quadrature
rules); the `torch` / `jax` extras add the differentiable backends.

## Scope (honest labels)

The measure integral here is a **numerical** (quadrature / sampling)
approximation -- `int f dmu` is not a closed form for an arbitrary measurable
`f` (that is true of the Riemann integral too). The autograd-exact, stable
*derivatives* of the soft superlevel indicator come from the omnibias sigmoid
tower. For a **rigorous, outward-rounded** integral or `L^p` / Sobolev norm with
guaranteed bounds, use the certified register in `omnibias.verify` /
`omnibias.core.verified`.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
