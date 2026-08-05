# omnibias-spiking

**Status: Alpha (0.1.0a1).**

Spiking-neuron **LIF** and **IF** single-step primitives whose forward pass is a
hard Heaviside threshold and whose backward pass uses an **exact closed-form
surrogate gradient** from the omnibias derivative tower — not an ad-hoc
surrogate implementation.

## Math

**Leaky integrate-and-fire (one step)**

```
v_pre = decay * v + x
s     = H(v_pre - threshold)          # hard spike, 0/1
v_out = v_pre - s * threshold         # soft reset
```

**Integrate-and-fire** is `lif_step` with `decay=1.0` (`if_step`).

The spike is non-differentiable in the forward pass. In the backward pass we
replace the Heaviside derivative with the closed-form derivative of a smooth
dictionary activation evaluated at `u = surrogate_scale * (v_pre - threshold)`:

| `surrogate`      | backward factor `d sigma / du`                         |
|------------------|--------------------------------------------------------|
| `fast_sigmoid`   | `sigmoid'(u) = s(1-s)`, `s = sigmoid(u)` (`P_1` coeffs)|
| `gaussian`       | `exp(-u^2/2) / sqrt(2 pi)` (Gaussian bump, `He_0`)   |

The chain rule gives `d s / d v_pre = surrogate_scale * d sigma / du`.

Coefficients come from `omnibias.core.polynomials` (`sigmoid_polynomial_coeffs`,
`hermite_coeffs`), so torch and JAX backends are bit-identical by construction.

## Usage

### PyTorch

```python
import torch
from omnibias.spiking.torch import ops

v = torch.zeros(4, dtype=torch.get_default_dtype(), requires_grad=True)
x = torch.tensor([0.5, 1.1, 0.2, 2.0], dtype=torch.get_default_dtype())
s, v_next = ops.lif_step(v, x, decay=0.9, threshold=1.0, surrogate_scale=4.0)
loss = s.sum()
loss.backward()
```

### JAX

```python
import jax
import jax.numpy as jnp
from omnibias.spiking.jax import ops

jax.config.update("jax_enable_x64", True)

def step(v, x):
    return ops.lif_step(v, x, decay=0.9, threshold=1.0, surrogate_scale=4.0)

v = jnp.zeros(4)
x = jnp.array([0.5, 1.1, 0.2, 2.0])
s, v_next = step(v, x)
grad_v, grad_x = jax.grad(lambda v, x: step(v, x)[0].sum(), argnums=(0, 1))(v, x)
```

## Public API

Both backends expose the same symbols under `omnibias.spiking.torch.ops` and
`omnibias.spiking.jax.ops`:

- `heaviside_spike`
- `surrogate_derivative`
- `lif_step`
- `if_step`

Optional `surrogate` may be an `omnibias.core.spec.ActivationSpec` whose
`derivative` or `fastpath(..., 1)` supplies the closed-form surrogate.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
