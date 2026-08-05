<!--
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2026 Derivon
-->
# omnibias-shape

Differentiable **soft shape / occupancy fields** and **soft-coverage** (soft-OR /
log-sum-exp union) operators, with a **closed-form derivative tower** and
bit-identical PyTorch + JAX twins.

A soft axis-aligned box is a separable product of sigmoid-pair interval indicators;
a soft *cover* is the probabilistic-OR union
`C = 1 - prod_k (1 - alpha_k m_k)` of `K` such shapes with existence gates `alpha`.
Because every shape is built purely from `sigmoid`, all center-derivatives are
closed-form polynomials in `sigmoid` via the shared Riccati tower
(`omnibias.core.polynomials.sigmoid_polynomial_coeffs`). This gives the **exact
gradient and Hessian** of a coverage energy with respect to the shape centers --
no autodiff -- which is what makes a discrete geometric-covering problem
second-order-optimizable.

## Why

Turn *"cover every 1-pixel of a binary image with the fewest fixed-size squares"*
(a hard discrete set-cover) into a smooth energy on continuous square centers +
existence gates, then minimise it with a curvature-aware optimiser and anneal the
sharpness `beta` toward the discrete solution. See
[`examples/min_square_cover`](../../examples/min_square_cover) for the full study.

## Operator surface

```python
import torch
from omnibias.shape.torch import ops as shape   # or: omnibias.shape.jax

M = N = 12
axes = [torch.linspace(0.0, 1.0, M, dtype=torch.float64),
        torch.linspace(0.0, 1.0, N, dtype=torch.float64)]
centers = torch.tensor([[0.3, 0.3], [0.7, 0.7]], dtype=torch.float64)
side, beta = 0.3, 20.0
gates = torch.ones(2, dtype=torch.float64)               # per-square existence gates
ones_mask = torch.zeros(M, N, dtype=torch.float64)       # the cells that must be covered
ones_mask[3:6, 3:6] = 1.0

occ  = shape.soft_box(axes, centers, side, beta)          # (K, M, N) occupancy
cov, _ = shape.soft_or_coverage(occ, gates)               # (M, N) union in (0, 1)
E    = shape.coverage_energy(occ, gates, ones_mask, loss="softplus", lam=0.1)
g    = shape.coverage_energy_grad(axes, centers, side, beta, gates, ones_mask)
H    = shape.coverage_energy_hessian(axes, centers, side, beta, gates, ones_mask)
```

- **occupancy**: `soft_interval`, `soft_box`, `soft_box_grad`, `soft_box_hessian`,
  `soft_disk`, `soft_polytope`.
- **coverage**: `soft_or_coverage`, `lse_coverage`, `coverage_energy`,
  `coverage_residual`, `coverage_energy_grad`, `coverage_energy_hessian`.
- **cardinality**: `l0_surrogate`, `anneal_lambda`, `prune_inactive`.

`coverage_energy_grad` / `coverage_energy_hessian` are validated bit-close
(`< 1e-10` in float64) against autodiff; the derivation is in
[`examples/min_square_cover/HESSIAN.md`](../../examples/min_square_cover/HESSIAN.md).

## Install

```bash
pip install -e "packages/omnibias-shape[torch]"     # or [jax], or [all]
```

Depends only on `omnibias-core` (pure-Python coefficients); torch / jax are optional
backends. Status: **alpha**.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
