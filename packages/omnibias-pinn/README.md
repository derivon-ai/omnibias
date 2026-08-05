# omnibias-pinn

**Status: Beta (v0.1.0).**

Physics-informed neural networks (PINNs) with **closed-form n-th
derivative operators**, built on top of `omnibias-core`. While
[DeepXDE](https://github.com/lululxvi/deepxde) and
[NVIDIA Modulus](https://github.com/NVIDIA/modulus-sym) differentiate
through stacked layers via autograd (cost grows exponentially as you nest
operators, and round-off accumulates across the nested graph),
`omnibias-pinn` computes the operators *in closed form* at one
forward-pass cost per order, up to the order the activation supports.

## Why this matters for PINNs

Measured float64, identical answers to autodiff up to `≤ 10⁻¹⁵` — see
[`docs/complexity.md`](../../docs/complexity.md):

- The Laplacian residual you put in your PINN loss is `O(1)` in input
  dimension `D` and **199×** faster than `torch.func.hessian + trace` at
  `D = 240` (and **108×** less memory).
- 4th-order PDEs (biharmonic, Cahn–Hilliard, strain-gradient, Helfrich) and
  6th-order plate / KS variants are routine — `polylaplacian(k)` is **flat**
  in `k`, not nested.
- Cross-backend (torch + jax) bit-parity means PINN residuals are
  reproducible on any GPU, on any framework, ULP-equal.

The package surfaces three layers --

* **Fields** (`fields/`): typed structural backends with closed-form
  derivatives -- `OneLayerVectorField`, `SpectralVectorField`,
  `ChebyshevVectorField`.
* **Ops** (`ops/`): user-facing operator surface (`derivative`,
  `gradient`, `divergence`, `laplacian`, `mixed_partial`,
  `biharmonic`, `polylaplacian`, `hessian`, `jacobian`, `curl`,
  `vorticity`, `strain_rate`, `advection`, `material_derivative`,
  `p_laplacian`, ...).
* **Cage** (`cage/`): hard-conservation layers
  (`StreamfunctionField`, `VectorPotentialField`,
  `HelmholtzProjectionField`, `HardBoundaryField`,
  `MassFluxPotentialField`, plus skew-symmetric advection helpers for
  energy / enstrophy preservation).

-- plus equation-aware modules:

* **Losses** (`losses/`): Sobolev preconditioning, Wang-Perdikaris
  causal weighting, NTK rebalance, entropy-consistent residual.
* **Equations** (`equations/`): prebuilt PDE residuals -- Heat,
  Burgers, KuramotoSivashinsky, CahnHilliard, Biharmonic,
  NavierStokes (3D primitive + 2D vorticity-stream forms). Each
  returns a `NamedTuple` with the residual tensor and a diagnostic
  dict.
* **Diagnostics** (`diagnostics/`):
  `relative_l2_per_time`, `forecast_horizon`, `spectral_fidelity`,
  `derivative_stability`, `autograd_phase_check`.

Both the PyTorch and JAX backends ship in lockstep, with bit-identical
numerics enforced by 620 cross-backend tests.

## Install

```bash
pip install omnibias-pinn[torch]              # PyTorch backend
pip install omnibias-pinn[jax]                # JAX backend
pip install omnibias-pinn[all]                # both
```

## 90-second tour

```python
import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields import SpectralVectorField
from omnibias.pinn.torch.cage import VectorPotentialField
from omnibias.pinn.torch.equations import NavierStokes

# Vector potential -> hard-incompressible 3D Navier-Stokes.
base = SpectralVectorField(
    coordinate_spec=CoordinateSpec(axes=("x", "y", "z", "t"), time_axis="t"),
    components=ComponentSpec(("A1", "A2", "A3", "p")),
    K=8, time_hidden=128, time_depth=3, activation="tanh",
)
cage = VectorPotentialField(
    base=base,
    A_components=("A1", "A2", "A3"),
    velocity_names=("u", "v", "w"),
    passthrough_names=("p",),
)

equation = NavierStokes(
    viscosity=1e-2, form="primitive_3d",
    velocity=("u", "v", "w"), pressure="p",
    incompressibility="hard",        # cage already enforces div(u)=0
)

coords = torch.rand(4096, 4, dtype=torch.float64) * 6.28
state = cage(coords)
out = equation(state)

loss = (out.residual ** 2).mean()
# On a gridded residual instead, `losses.sobolev_residual_loss` /
# `losses.causal_residual_loss` precondition this in the Fourier basis.

# attribute-based DSL -- everything routes back to closed-form ops:
state.u.dt           # ∂u/∂t
state.u.lap          # Δu
state.velocity.curl  # ∇×u
```

The cage passes derivatives through its algebraic identity, so it exposes the
orders that identity defines; higher towers such as `state.u.biharm` (Δ²u) come
from the uncaged spectral field.

## What's in v0.1.0

* **Closed-form spatial derivatives** for Fourier and Chebyshev bases
  at any order: $O(B \cdot K^d)$ per residual evaluation, regardless
  of derivative order.
* **Hard-conservation cages**: `div u = 0` to floating-point
  round-off via `StreamfunctionField` (2D) and `VectorPotentialField`
  (3D). Skew-symmetric advection helpers for energy / enstrophy
  preservation.
* **Cross-backend bit-parity**: every torch op has a JAX twin whose
  output is bit-identical for the same `(seed, weights, coords,
  dtype=float64)`. 620 / 620 cross-backend tests pass.
* **Integration parity**: integration tests verify residual / loss
  parity for 2D Navier-Stokes, Kuramoto-Sivashinsky, and Cahn-Hilliard
  on pinned smoke configs (`tests/integration/`).
* **Headline 3D Navier-Stokes benchmark** lives in the internal benchmark
  archive (not shipped publicly; see [`docs/benchmarks.md`](../../docs/benchmarks.md));
  the package keeps only public APIs, docs, and regression tests.

## Documentation

* API reference: [`docs/api/pinn.md`](../../docs/api/pinn.md).
* Cookbook:
  [`docs/cookbook/pinn-navier-stokes.md`](../../docs/cookbook/pinn-navier-stokes.md),
  [`docs/cookbook/pinn-strict-conservation.md`](../../docs/cookbook/pinn-strict-conservation.md).
* Migration from DeepXDE / Modulus:
  [`docs/migration-pinn.md`](../../docs/migration-pinn.md).
* Math derivations:
  [`docs/pinn-derivations.md`](../../docs/pinn-derivations.md).
* Stability matrix:
  [`docs/stability.md`](../../docs/stability.md) (omnibias-pinn section).

## Tests

```bash
scripts/run_tests.sh fast        # _core + torch + jax unit tests
scripts/run_tests.sh cross       # + cross-backend bit-parity tests
scripts/run_tests.sh integ       # + package integration tests
scripts/run_tests.sh full        # everything (~110 s)
```

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
