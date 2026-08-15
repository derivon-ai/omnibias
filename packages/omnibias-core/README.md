# omnibias-core

Backend-agnostic mathematical core of the omnibias framework.

This package is the **bottom of the dependency graph**. It ships the pure-Python
polynomial coefficient generators that power every closed-form `σ^(n)` kernel,
plus the generic `ActivationSpec` protocol that `omnibias-torch`,
`omnibias-jax`, and `omnibias-keras` specialise. There is **no** torch / jax /
numpy / tensorflow dependency — both backends import from here, so the
Eulerian / Legendre / Hermite recurrences produce **bit-identical** coefficient
sequences by construction.

## Install

```bash
pip install omnibias-core
```

## Quick start

```python
from omnibias.core.polynomials import (
    sigmoid_polynomial_coeffs,
    tanh_polynomial_coeffs,
    hermite_coeffs,
)

# Coefficients of the degree-(n+1) polynomial P_n such that
#   sigma^(n)(z) = P_n(sigma(z))   (Riccati class)
print(sigmoid_polynomial_coeffs(3))  # 3rd derivative of sigmoid
print(tanh_polynomial_coeffs(4))
print(hermite_coeffs(5))             # probabilist's Hermite (Gaussian)
```

```python
# ActivationSpec is a frozen dataclass of shared metadata. Backends pin
# TensorT to their array type and fill in forward / fastpath callables.
# Prefer the backend registries rather than constructing one by hand:
#
#   from omnibias.torch import get_activation
#   spec = get_activation("tanh")
#   spec.fastpath(z, n)   # closed-form sigma^(n)
```

## Public surface (highlights)

| Module | Role |
|---|---|
| `omnibias.core.polynomials` | `sigmoid_polynomial_coeffs`, `tanh_polynomial_coeffs`, `hermite_coeffs` |
| `omnibias.core.spec` | `ActivationSpec` — shared activation metadata |
| `omnibias.core.multipack` | `PackSpec` / `MultiPackSpec` — heterogeneous Birkhoff support (theory 01-01, **gated**) |
| `omnibias.core.scan` | `BankSpec` — offset / scale bank for the bias scan (theory 01-02, **gated**) |
| `omnibias.core.mollifier` | `MollifierSpec` / `tail_bound` — pack-as-mollifier algebra; certified exponential tails, not compact support (theory 01-05, **gated**) |
| `omnibias.core.spectral_design` | `BandPlan` / `peak_frequency` — order as a band selector, not Littlewood-Paley completeness (theory 01-07, **gated**) |
| `omnibias.core.frames` | `FrameSpec` / `admissibility_constant` — `sigma'` is not admissible (theory 01-06, **gated**) |
| `omnibias.core.locus` | `EqualitySystem` — constraint manifold, not a PDE solver (theory 01-09, **gated**) |
| `omnibias.core.jets` | `contact_residual` / `is_holonomic` — vocabulary, not a discovery (theory 01-10, **gated**) |
| `omnibias.core.conjugate` | line Hilbert permutation of the dictionary (theory 01-12, **gated**; G5 not in CI `all_passed`) |
| `omnibias.core.hierarchy` | 1-D pack tree; `eta=0` bit-identical to dense (theory 02-07, **gated**) |
| `omnibias.core.tanh_method` | travelling-wave tanh algebra, not a collapse (theory 02-09, **gated**) |
| `omnibias.core.ladder` | Hermite raise/lower; Rodrigues reweight required (theory 02-10, **gated**) |
| `omnibias.core.transfer` | 1-D ABCD stacks; `continuum_claim=False` (theory 02-11, **gated**) |
| `omnibias.core.transforms_pde` | named Cole-Hopf / Miura / Bäcklund / Darboux (theory 02-13, **gated**) |
| `omnibias.core.bell` | Bell polynomials / Faà di Bruno combinatorics |
| `omnibias.core.multi_index` | Multi-index ordering + Cauchy product for multivariate jets |
| `omnibias.core.verified` | Rigorous numerics: `Interval`, Taylor models, Kantorovich, Lohner, … |
| `omnibias.core.proof` | Hash-sealed certificate format v1 + Lean bridge |

## Why a separate package?

Forking the coefficients per backend would silently break bit-identity. Keeping
them in a pure-Python wheel means a JAX-only or torch-only install still shares
the same numbers, and the Lean / verified substrate never has to import a
framework.

## Docs

- Theory primer: [`docs/theory.md`](../../docs/theory.md)
- Operator surface: [`docs/operator-surface.md`](../../docs/operator-surface.md)
- Stability matrix: [`docs/stability.md`](../../docs/stability.md)
- Monorepo overview: [`README.md`](../../README.md)

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
