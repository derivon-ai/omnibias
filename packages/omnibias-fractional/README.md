# omnibias-fractional

Fractional calculus for omnibias in **two classes**: grid / spectral operators
(Grünwald–Letnikov, Riemann–Liouville, Caputo, FFT) that are *non-local numerical
approximations*, and a *closed-form* analytic fractional derivative on the
analytic-function class (an order-`N` truncation). Cross-backend (PyTorch + JAX).

> **Honesty note.** Fractional derivatives are **non-local**: the value at a
> point depends on the whole history/domain. The Grünwald–Letnikov, Riemann–
> Liouville, Caputo and spectral operators are therefore *grid-based numerical
> approximations*, **not** the exact closed-form sigma-tower derivatives the
> other packages provide; accuracy is controlled by the grid resolution.
> Separately, `fractional_derivative` **is** closed form — but only on the
> *analytic-function class*, as an order-`N` truncation of the gamma-ratio
> Taylor series (exact for degree-`≤ N` polynomials). See
> [`FRACTIONAL_DERIVATIONS.md`](FRACTIONAL_DERIVATIONS.md).

## Install

```bash
pip install "omnibias-fractional[torch]"   # or [jax], or [all]
```

## Use

```python
import numpy as np, torch
from omnibias.fractional.torch import ops as fr

x = np.linspace(0.0, 2.0, 4000)
h = x[1] - x[0]
f = torch.as_tensor(x**3, dtype=torch.float64)
d_half = fr.caputo(f, alpha=0.5, h=h)        # ~ Gamma(4)/Gamma(3.5) x^2.5

# spectral (periodic)
L = 2*np.pi
xp = np.linspace(0, L, 256, endpoint=False)
g = torch.as_tensor(np.sin(xp), dtype=torch.float64)
fr.spectral_fractional(g, alpha=0.5, length=L)   # complex tensor
```

## Learnable order

`alpha` can be a backend tensor, not just a `float` -- then the fractional *order*
is differentiable and can be trained (e.g. discovered by a PINN or the neural-jet
engine). `LearnableOrder` keeps it in a stable band:

```python
import torch
from omnibias.fractional.torch import ops as fr
from omnibias.fractional.torch.order import LearnableOrder

order = LearnableOrder(init=0.5, lo=0.0, hi=2.0)     # bounded, differentiable
L = 2 * torch.pi
f = torch.sin(torch.linspace(0.0, L, 256, dtype=torch.float64))
d = fr.spectral_fractional(f, alpha=order(), length=float(L))
(d.real**2 + d.imag**2).sum().backward()             # gradient flows to the order
```

A Python `float` `alpha` keeps the original numpy kernel unchanged; only a tensor
`alpha` takes the in-backend autograd path.

## Closed-form (analytic) fractional derivative

Given a Taylor jet `a_k = f^(k)(a)/k!` about a terminal `a`, the fractional
derivative is the closed-form gamma-ratio series
`D^alpha f = sum_k a_k Gamma(k+1)/Gamma(k+1-alpha) (x-a)^(k-alpha)` -- no grid, no
history, differentiable in `alpha` and the coefficients:

```python
import torch
from omnibias.fractional.torch import ops as fr

# f(t) = 1 + 2t + 0.5 t^2 as a jet about the terminal a = 0
jet = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
x = torch.linspace(0.1, 2.0, 50, dtype=torch.float64)      # require t = x - a >= 0

d_half = fr.fractional_derivative(jet, x, alpha=0.5)                 # Riemann-Liouville
d_cap  = fr.fractional_derivative(jet, x, alpha=0.5, kind="caputo")  # regular at t=0
# alpha=0 -> f itself; integer alpha -> the ordinary derivative (for t > 0).
```

Pair it with a network's *exact* directional jet (needs `omnibias-torch`):

```python
# a 3 -> 4 -> 1 MLP, differentiated along the ray x(s) = x0 + s v
W1 = torch.randn(4, 3, dtype=torch.float64) * 0.5; b1 = torch.zeros(4, dtype=torch.float64)
W2 = torch.randn(1, 4, dtype=torch.float64) * 0.5; b2 = torch.zeros(1, dtype=torch.float64)
layers = [(W1, b1, "tanh"), (W2, b2, None)]
x0 = torch.zeros(3, dtype=torch.float64)
v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

d = fr.mlp_fractional_derivative(x0, v, layers, x, alpha=0.5, order=5)
```

The order can be an `nn.Parameter` / `LearnableOrder` here too, so a network and
its fractional order train jointly. It is an order-`N` truncation (exact for
degree-`≤ N` polynomials); the intended regime is *non-integer* `alpha`.

## Field fractional partial + fractional-diffusion residual

The closed-form operator lifts onto the `omnibias-fields` `FieldState` substrate:
`field_fractional_partial` is the fractional twin of a per-axis field
`derivative`. It expands the field's Taylor jet **along one axis about the lower
terminal `a`** (re-evaluating the field with that axis pinned to `a`) and sums the
gamma-ratio series at the collocation points -- closed form on the
analytic-function class, differentiable in `alpha` and the field parameters. It
needs `omnibias-fields` (installed by the `[torch]` / `[jax]` extras) and lives in
`omnibias.fractional.<backend>.field` (imported lazily, so the `ops` surface keeps
its `omnibias-fields`-free install).

```python
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.fractional.torch import field as ffield

# any omnibias-fields FieldState with a closed-form per-axis tower
u = OneLayerVectorField(coordinate_spec=CoordinateSpec(("x", "t"), time_axis="t"),
                        components=ComponentSpec(("u",)), hidden=16, base="tanh")
state = u(torch.rand(6, 2, dtype=torch.float64) + 0.1)    # x - a >= 0

d_half = ffield.field_fractional_partial(
    state, "u", axis="x", alpha=0.5, order=8, a=0.0, kind="caputo"
)

# space-fractional diffusion PINN residual  u_t - sum_a D^{alpha_a}_{x_a} u - s
out = ffield.fractional_diffusion_residual(
    state, alphas=(0.5,), order=8, component="u", kind="caputo",
    source=lambda s: torch.zeros_like(s.coords[:, 0]),   # any FieldState -> (B,)
)
loss = out.residual.pow(2).mean()          # drops into any PINN training loop
```

`alphas` gives one order per spatial axis; a plain-number integer order is steered
to the exact derivative tower. The jax twin (`omnibias.fractional.jax.field`) is
bit-identical.

## API

| Op | Class | Domain | Notes |
|---|---|---|---|
| `grunwald_letnikov(f, *, alpha, h)` | numerical (grid) | uniform grid | causal GL discretisation, O(h) |
| `riemann_liouville(f, *, alpha, h)` | numerical (grid) | uniform grid | GL discretisation |
| `caputo(f, *, alpha, h)` | numerical (grid) | uniform grid, `0 < alpha < 1` | GL of `f - f(0)` |
| `spectral_fractional(f, *, alpha, length)` | numerical (spectral) | periodic | FFT multiplier `(ik)^alpha` |
| `fractional_derivative(jet, x, *, alpha, a, kind)` | **closed-form** (analytic class) | analytic (jet) | gamma-ratio series; RL / Caputo; order-`N` truncation |
| `mlp_fractional_derivative(x0, v, layers, t, *, alpha, order, kind)` | **closed-form** (analytic class) | analytic (MLP) | pairs with `mlp_jet`; order-`N` truncation |
| `field.field_fractional_partial(state, name, *, axis, alpha, order, a, kind)` | **closed-form** (analytic class) | field (`FieldState`) | per-axis jet about terminal `a`; needs `omnibias-fields` |
| `field.fractional_diffusion_residual(state, *, alphas, order, component, kind, a, source)` | **closed-form** (analytic class) | field (`FieldState`) | `u_t - sum_a D^{alpha_a}_{x_a} u - s` PINN residual |

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
