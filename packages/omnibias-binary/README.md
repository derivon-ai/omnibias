# omnibias-binary

**Status: Alpha (0.1.0a1).**

Closed-form, deterministic gradient kernels for binary / ternary / k-bit
quantized neural-network training. The **forward** pass is a hard quantizer;
the **backward** pass uses the exact derivative of a smooth `tanh(beta * z)`
surrogate (which converges to `sign(z)` as `beta -> infinity`) via the Riccati
identity:

```
tanh'(z) = 1 - tanh(z)^2
```

Every order is a polynomial in `t = tanh(z)` from
`omnibias.core.polynomials.tanh_polynomial_coeffs` — no per-backend coefficient
forks, no straight-through estimator.

## Public API

| Function | Forward | Backward surrogate |
|----------|---------|-------------------|
| `binarize(z, beta=10)` | `sign(z)` in `{-1, +1}` (`sign(0)=+1`) | `beta * tanh'(beta z)` |
| `binarize01(z, beta=10)` / `heaviside` | Heaviside step in `{0, 1}` (`H(0)=1`) | `beta * sigmoid'(beta z) = beta * s(1-s)` |
| `ternarize(z, beta=10, delta=0.5)` | `{-1, 0, +1}` dead-zone | derivative of `0.5(tanh(beta(z-delta))+tanh(beta(z+delta)))` |
| `kbit_quantize(z, bits=2, lo=-1, hi=1, beta=10)` | uniform `2**bits` levels on `[lo, hi]` | sum of tanh-step Riccati derivatives at internal thresholds |
| `riccati_tanh_derivative(t, order=1)` | — | evaluates `T_order(t)` (`{-1,+1}` Legendre tower) via Horner |
| `riccati_sigmoid_derivative(s, order=1)` | — | evaluates `P_order(s)` (`{0,1}` Eulerian tower) via Horner |

`binarize01` is the `{0, 1}` codomain twin of `binarize`, built from the Eulerian
`sigmoid_polynomial_coeffs` tower. The two are affinely conjugate --
`binarize01(z, beta) == (binarize(z, beta / 2) + 1) / 2` in both forward and
backward -- so you can pick the natural codomain for your gates (`{0,1}` for
AND/OR/Reed-Muller logic, `{-1,+1}` for XOR/Walsh logic) without changing the math.

Backends: `omnibias.binary.torch.ops` and `omnibias.binary.jax.ops`.

## Usage

```python
import torch
from omnibias.binary.torch.ops import binarize, ternarize, kbit_quantize

z = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
y = binarize(z, beta=20.0)
loss = y.pow(2).mean()
loss.backward()  # z.grad uses the closed-form tanh-beta Riccati surrogate
```

```python
import jax
import jax.numpy as jnp
from omnibias.binary.jax.ops import binarize

z = jnp.ones((4, 8), dtype=jnp.float64)
y = binarize(z, beta=20.0)
grad = jax.grad(lambda zz: jnp.sum(binarize(zz, beta=20.0)))(z)
```

## Tests

```bash
python -m pytest packages/omnibias-binary/tests -q
```

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
