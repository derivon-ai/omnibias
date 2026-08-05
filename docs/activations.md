# omnibias activation dictionary

Each backend activation registered in `omnibias.torch.activations` or `omnibias.jax` carries:

- `forward(z)`: `sigma(z)`.
- `derivative(z)`: closed-form `sigma'(z)` if known.
- `fastpath(z, n)`: closed-form `sigma^(n)(z)` for the orders this
  activation supports.
- `integral(z)`: closed-form antiderivative `S(z)` with `S'(z) = sigma(z)`,
  when the backend ships a stable primitive (`None` otherwise). This is what
  powers `OperatorBlock(op="integral")`, whose forward is the definite
  bias-window `S(z + b_hi) - S(z + b_lo)`. Example: `sigmoid`'s antiderivative
  is `softplus`. Which activations carry one is checked by
  `packages/omnibias-torch/tests/test_integral_primitives.py`; the full
  operator surface is in [`operator-surface.md`](operator-surface.md).
- `riccati_polynomial`: the polynomial `P` such that `sigma'(z) =
  P(sigma(z))` for activations in the Riccati class (sigmoid, tanh,
  exp). `None` otherwise.
- `noise_model`: GLM family for which `sigma` is the log-partition
  function; `"none"` for activations not arising as a log-partition.
- `operator_role`: one-line description of the canonical operator role
  of the K=2 bias-collapse unit with this activation.

Use `omnibias.torch.list_activations()` or `omnibias.jax.list_activations()` to see registered names, and the matching backend `get_activation(name)` to retrieve the spec.

## Smooth (Riccati) family

Closed-form derivative tower available at every order. Default choice
for PINN-style architectures and any setting where higher derivatives
of `sigma` are needed.

| name       | `sigma(z)`                | `sigma'(z)`                      | Riccati `P` |
|------------|---------------------------|----------------------------------|-------------|
| `sigmoid`  | `1 / (1 + e^{-z})`        | `s (1 - s)` with `s = sigma(z)`  | `s - s^2`   |
| `tanh`     | `tanh(z)`                 | `1 - tanh^2 z`                   | `1 - t^2`   |
| `softplus` | `log(1 + e^z)`            | `sigmoid(z)`                     | derivative tower from order 1 reuses sigmoid's |
| `gaussian` | `exp(-z^2/2)`             | `-z * exp(-z^2/2)`               | not Riccati; uses Hermite tower instead |

Fast-path implementation:

- `sigmoid`: Eulerian-polynomial recursion (`omnibias.torch.fastpath.eulerian`).
- `tanh`: Legendre-style recursion `T_{n+1} = (1 - t^2) T_n'`
  (`omnibias.torch.fastpath.legendre`).
- `softplus`: `softplus^(n>=1)(z) = sigma^(n-1)(z)`, reuses Eulerian.
- `gaussian`: probabilist's Hermite identity
  `g^(n)(z) = (-1)^n He_n(z) g(z)` (`omnibias.torch.fastpath.hermite`).

## Proximal family

Designed so that the K=2 bias-collapse output **is** a classical
proximal operator. `huber` now carries an all-orders **almost-everywhere**
tower (`n = 2` is the indicator `1[|z| <= tau]`, `n >= 3 -> 0`; the kink
deltas are dropped); `arctan` and `log1pu2` keep their analytic tower up
to `n = 2`.

| name       | K=2 collapse output         | Operator role                                  |
|------------|-----------------------------|------------------------------------------------|
| `huber`    | `clip(z, -tau, tau)`        | proximal of `||.||_1` (LASSO ISTA soft-shrink) |
| `arctan`   | `1 / (1 + z^2)`             | Cauchy IRLS weight                             |
| `log1pu2`  | `2 z / (1 + z^2)`           | redescending M-estimator (Black-Anandan)       |

Build a Huber spec with custom threshold via
`omnibias.torch.activations.proximal.make_huber_spec(tau=...)`. The default
`huber` spec uses `tau = 1.0`; pass `register=True` to add a custom-tau
spec to the global registry.

## Classical family

Drop-in compatibility with existing pretrained backbones. `exp`, `silu`,
and `gelu` now carry **exact all-orders** towers; `relu` carries an
all-orders **almost-everywhere** tower (see the piecewise family below).

| name   | `sigma(z)`            | `sigma'(z)`                          | tower |
|--------|-----------------------|--------------------------------------|-------|
| `exp`  | `exp(z)`              | `exp(z)` (eigenfunction of `d/dz`)   | exact, all `n` |
| `relu` | `max(z, 0)`           | Heaviside step (`H(0) = 0`)          | a.e., all `n` (`n >= 2 -> 0`) |
| `silu` | `z * sigma(z)`        | `sigma + z * sigma * (1 - sigma)`    | exact, all `n` |
| `gelu` | `z * Phi(z)` (exact)  | `Phi(z) + z * phi(z)`                | exact, all `n` |

`silu` and `gelu` reuse the sigmoid / Gaussian towers through the Leibniz
product rule, `silu^(n)(z) = z\,sigma^(n)(z) + n\,sigma^(n-1)(z)` and
`gelu^(n)(z) = z\,Phi^(n)(z) + n\,Phi^(n-1)(z)`, so `op="laplacian"` and
higher-order operator blocks are now valid for the whole classical family.

## Piecewise (almost-everywhere) family

The "hard" non-smooth activations. Each has a **closed-form all-orders**
fast path on the *almost-everywhere* (a.e.) / regular-part convention: on
every open piece the tower is the exact classical derivative, and the
**singular part** (Dirac deltas living on the measure-zero breakpoint set)
is dropped. Boundary values follow the PyTorch convention (`H(0) = 0`,
`sign(0) = 0`). All are registered in `omnibias.torch`, `omnibias.jax`, and
`omnibias.keras`, and are bit-identical across backends.

- **Piecewise-linear** (`n >= 2 -> 0`): `leaky_relu`, `prelu`, `relu6`,
  `hardtanh`, `hardsigmoid`, `softshrink`, `hardshrink`, `threshold`, `abs`,
  `sign`, `step` (alias `heaviside`).
- **Piecewise-smooth** (exact per-arm tower, all orders): `elu`, `selu`,
  `celu`, `hardswish`, `softsign`.

Fixed-parameter variants come from the factories `make_leaky_relu_spec`,
`make_hardtanh_spec`, `make_elu_spec`, `make_celu_spec`,
`make_softshrink_spec`, `make_hardshrink_spec`, `make_threshold_spec`
(torch / keras `omnibias.<backend>.activations.piecewise`; jax
`omnibias.jax.activations`).

!!! warning "Behavioral change"
    `relu` and `huber` previously *raised* `NotImplementedError` for order
    `>= 2`; they now return the almost-everywhere value. See the CHANGELOG.

## Beta-tempered smooth surrogates

The complementary *smooth* family: differentiable surrogates with a
temperature `beta` that sharpen to the hard activation as `beta -> inf`,
and whose higher-order bumps *become* the dropped Dirac deltas in that
limit. Each reuses an existing closed-form tower through the backend-neutral
`omnibias.core.tempered` combinator, so every order is exact and
bit-identical across backends.

| surrogate     | definition                              | hard limit (`beta -> inf`) |
|---------------|-----------------------------------------|----------------------------|
| `soft_relu`   | `softplus(beta z) / beta`               | `relu`                     |
| `soft_step`   | `sigmoid(beta z)`                       | `step` / `heaviside`       |
| `soft_sign`   | `tanh(beta z)` (= `smooth_sign`)        | `sign`                     |
| `soft_abs`    | `sqrt(z^2 + eps^2) - eps` (= `softabs`) | `abs`                      |

The scaling identity is `f_beta^(n)(z) = beta^(n-p) g^(n)(beta z)` with
`p = 1` for `soft_relu` (the `/ beta`) and `p = 0` for the bounded
surrogates. `soft_sign` / `soft_abs` are aliases folded into the existing
`smooth_sign` / `softabs` specs. Extra factories (not auto-registered):
`make_swish_spec(beta)` (`z * sigmoid(beta z)`; `silu` is `beta = 1`) and
`make_soft_leaky_relu_spec(slope, beta)`.

`beta` may be a Python float **or** a tensor / `nn.Parameter` / traced
`Array` for a learnable, differentiable temperature. The learnable-`beta`
wrappers are `omnibias.torch.TemperedActivation` (an `nn.Module`),
`omnibias.keras.TemperedActivation` (a `Layer`), and the functional
`omnibias.jax.activations.tempered_activation`; `LearnablePReLU`
(torch / keras) is the learnable-slope analogue. This is the same
`beta -> inf` annealing trick used by the quantizers in
[`omnibias-binary`](api/binary.md).

## Design rules

When choosing a base activation for an operator-typed layer, ask:

1. **What operator role do I need?** Look up the K=2 collapse output in
   the table; pick the activation whose collapse matches.
2. **What derivative orders do I need?** Smooth (Riccati), classical, and
   beta-tempered families give exact towers at every order; the piecewise
   family gives an all-orders *a.e.* tower (singular part dropped);
   `arctan` / `log1pu2` cap at `n = 2`.
3. **What noise model does the upstream loss assume?** Match `sigma` to
   the GLM family whose log-partition it is.
4. **What inductive bias do I want at init?** Lemma 1 init makes the
   layer behave as the base `sigma` at step zero, no matter what `K` is.

## Adding a custom activation

```python
import torch
from omnibias.core import ActivationSpec
from omnibias.torch.activations import register_activation

spec = ActivationSpec(
    name="my_swish",
    forward=lambda z: z * torch.sigmoid(z * 1.7),
    derivative=None,        # mark unavailable
    fastpath=None,
    riccati_polynomial=None,
    noise_model="none",
    operator_role="custom; not a known proximal or GLM family",
)
register_activation(spec)
```

`register_activation` returns the spec unchanged, so it also reads as a decorator
over a spec-producing expression.

After registration `omnibias.torch.get_activation("my_swish")` and
`OperatorMultiBiasUnit(..., base="my_swish")` work as for any built-in.
The `OperatorBlock` paths that require a fast path will reject specs
without one with a clear error.
