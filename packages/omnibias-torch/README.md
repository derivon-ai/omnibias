# omnibias-torch

PyTorch backend for the omnibias closed-form n-th derivative framework.

## Why this is fast

All numbers float64, identical answers to autodiff up to `≤ 10⁻¹⁵`. Full
derivation in [`docs/complexity.md`](../../docs/complexity.md).

- Laplacian overhead is **`O(1)` in input dimension `D`** (0.167 → 0.211 ms
  at `D = 3 → 240` on GPU, `H = 256`, `B = 4096`).
- At `D = 240`, the closed-form Laplacian is **199× faster than
  `torch.func.hessian` + trace** and uses **108× less memory**.
- Iterated Laplacian `Δᵏ` is flat in `k` and `D`: closed-form stays at ~0.1 ms
  where folx-nested OOMs at `k = 4`.
- Bit-identical to `omnibias-jax` and `omnibias-keras` (same shared
  `omnibias.core` polynomial coefficients).

## Install

```bash
pip install omnibias-torch
# or:
pip install omnibias-torch[examples,test]
```

`omnibias-torch` depends on `omnibias-core` (pure-Python math) and
`torch>=2.0`.

## Public API

```python
import torch
from omnibias.torch import (
    OMBU, OperatorBlock, cmbLinear, cmbConv1d, cmbConv2d,
    GrowableOMBU, get_activation, list_activations, register_activation,
    BankSpec, BiasScan, MultiPackUnit,
)

# Trainable scalar-operator primitive (drop-in for an activation):
ombu = OMBU(num_channels=4, K=2, base="tanh")
out = ombu(torch.zeros(3, 4))

# Operator-typed block. Six roles: identity | grad | laplacian | derivative
# | band | integral. grad/laplacian/derivative are closed-form sigma^(n);
# integral is the closed-form antiderivative window S(z+b_hi)-S(z+b_lo), S'=sigma.
block = OperatorBlock(channels=8, op="grad", base="sigmoid")

# CmbLinear: drop-in for nn.Linear with an inline operator block:
linear = cmbLinear(in_features=128, out_features=64, op="identity", base="tanh")

# 23 registered activations, every Riccati-class one with closed-form
# derivatives at every order:
print(list_activations())
```

Gated Wave-1 primitives (not shipped): `MultiPackUnit` (heterogeneous Birkhoff
packs, 01-01) and `BiasScan` / `BankSpec` (transverse scan along `w`, 01-02).
`BiasScan` templates reuse the six `OperatorBlock` roles; equivariance is an
interior lattice shift, not a circular wrap. Soft-argmax `gamma` is not
`delta -> 0`. See [docs/api/multipack.md](../../docs/api/multipack.md) and
[docs/api/scan.md](../../docs/api/scan.md).

See [docs/theory.md](../../docs/theory.md) and the cookbook for end-to-end
PINN, CmbNet, and CvxLayer examples.

## Activation dictionary

23 real-valued activations registered, plus 3 complex-valued (NQS).
See `omnibias.STABILITY.md` (sanitized for the public docs site as
`docs/stability.md`) for the full table of supported derivative orders
per activation.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
