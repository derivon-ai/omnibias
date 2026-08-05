# API stability matrix

This page documents which symbols are part of the omnibias **stability
contract**, and what derivative orders each activation supports. The contract
covers the curated public core; each of those packages carries its own version
(see [`packages.md`](packages.md)).

## Contract levels

| level | meaning | breaking-change policy |
|---|---|---|
| **stable** | Documented; tested; in the public ``__init__.py`` | Two-step deprecation: warn for one minor release, remove in the next. |
| **experimental** | Newly added; ABI may shift across patch releases | Patch-release breakage permitted; documented in CHANGELOG.md. |

Every symbol re-exported from the top-level ``omnibias.<backend>``
namespace is **stable**. Every other symbol is implementation detail
unless documented otherwise.

## Curated-core release set (frozen public API)

Eight packages form the **curated core** and are published first under this
stability contract:

``omnibias-core``, ``omnibias-torch``, ``omnibias-jax``, ``omnibias-ferminet``,
``omnibias-keras``, ``omnibias-fields``, ``omnibias-pinn``, ``omnibias-geometry``.

Their top-level public API surface (each package's ``__all__``) is **frozen**
against a committed baseline, ``tests/data/curated_core_public_api.json``, and
enforced in CI by ``tests/test_curated_core_api_surface.py`` (the ``curated_api``
job). The surface may only change by a deliberate, reviewed act: regenerate the
baseline with ``python tests/_regen_curated_api.py`` and record the change in
[``CHANGELOG.md``](https://github.com/derivon-ai/omnibias/blob/main/CHANGELOG.md).
**Removing or renaming an exported symbol is a breaking change** and follows the
two-step deprecation policy above.

Each curated-core distribution is additionally build-, metadata-, and
clean-install-verified in CI (``build_wheels`` + ``twine check``,
``curated_clean_install`` matrix, and the ``wheel_import_smoke`` job that imports
every shipped package from its built wheel).

## Activation dictionary

Both backends register **23 real-valued activations**, each with the
maximum ``n`` for which a closed-form ``sigma^(n)(z)`` kernel is
implemented.

| family | activations | max n | typical use |
|---|---|---|---|
| Riccati / Eulerian (every n) | `sigmoid`, `tanh`, `softplus` | unbounded | core neural-VMC body |
| Eigenfunction of d/dz (every n) | `exp` | unbounded | bound-state envelope |
| Hermite-spectral (every n) | `gaussian` | unbounded | RBF orbital decay |
| Trigonometric (every n) | `sin`, `cos` | unbounded | plane-wave / Bloch states |
| Hyperbolic (every n) | `sinh`, `cosh` | unbounded | evanescent / anti-Bragg |
| Riccati periodic (n &leq; 3) | `tan`, `cot` (alias `ctg`/`ctan`) | 3 | phase-only ansatze |
| Hyperbolic Riccati (n &leq; 3) | `coth` | 3 | finite-T NQS amplitude |
| Soliton bound-state (n &leq; 3) | `sech` | 3 | Poschl-Teller, sine-Gordon, KdV |
| Proximal (n &leq; 2) | `arctan`, `log1pu2`, `softabs`, `smooth_sign` | 2 | robust statistics, NQS smoothing |
| NQS log-amplitude (n &leq; 3) | `log_cosh` | 3 | Carleo-Troyer RBM family |
| Mixed (n &leq; 1) | `huber`, `silu`/`swish`, `gelu`, `relu`, `mish` | 1 | drop-in compat with pretrained backbones |

*Riccati restriction is tight*: an activation has closed-form derivatives
at every order if and only if it is in the Riccati class
(``sigma'(z) = P(sigma(z))`` for some polynomial P) or is the
eigenfunction of ``d/dz``. See ``docs/theory.md`` for the proof.

## Cross-backend parity

For every supported (activation, n) pair, ``omnibias-torch`` and
``omnibias-jax`` produce float64-ULP-equal coefficients (they import
the same pure-Python recurrence from ``omnibias-core``). Verified per
release by ``tests/test_jax_parity.py`` (73 / 73 pass).

## Closed-form Laplacian / Hessian

The ``omnibias.jax.laplacian`` family computes the Laplacian and full
Hessian of a one-layer scalar field

\[ f(x) = b + \sum_h c_h \sigma(W_h \cdot x + \beta_h) \]

in closed form. Contract: agrees with ``jax.hessian`` to
``<= 1e-10 abs / 1e-12 rel`` for every activation that supports n &geq; 2.

## FermiNet bridge

The ``omnibias.ferminet.folx_compat.forward_laplacian`` adapter is
*bit-identical* to FermiNet's ``default`` autograd Laplacian on
closed-shell systems. Verified for Be (4 electrons, 4 walkers,
8-determinant FermiNet body): rel\_err = 0.0 to ULP versus default,
``<= 5.07e-15`` versus folx.

## omnibias-pinn (v0.1.0)

`omnibias-pinn` is published as **Beta** for v0.1.0. The
public surface is

* `omnibias.pinn._core.{coords, components, state, ...}` (shared schemas)
* `omnibias.pinn.<backend>.{fields, ops, cage, losses, equations, diagnostics}`
  for `backend in ("torch", "jax")`.

| Submodule | Stability | Notes |
| --- | --- | --- |
| `_core.coords` / `_core.components` | stable | `CoordinateSpec`, `ComponentSpec`, `FieldState`. |
| `fields.OneLayerVectorField` | stable | Drop-in MLP with autograd; keeps the pure baseline. |
| `fields.SpectralVectorField` | stable | Closed-form Fourier derivatives, omnibias temporal MLP. |
| `fields.ChebyshevVectorField` | experimental | Non-periodic; Chebyshev-T basis. |
| `ops.{basic, vector, nonlinear, high_order}` | stable | All cross-backend bit-parity verified. |
| `cage.{Streamfunction, VectorPotential}Field` | stable | Hard incompressibility cages. |
| `cage.{Helmholtz, HardBoundary, MassFluxPotential}Field` | experimental | Newer cages -- API may shift. |
| `losses.{sobolev, causal, ntk, entropy}` | stable | Public loss surface with package-level tests. |
| `equations.{Heat, Burgers, KuramotoSivashinsky, CahnHilliard, Biharmonic, NavierStokes}` | stable | `NamedTuple` outputs; integration-tested with pinned smoke fixtures. |
| `diagnostics.{relative_l2_per_time, forecast_horizon, spectral_fidelity}` | stable | Backend-agnostic NumPy. |
| `diagnostics.field_stability` | experimental | autograd-vs-closed-form benchmarks. |

### Cross-backend bit-parity (omnibias-pinn)

Every `omnibias.pinn.torch.<X>` symbol has a JAX twin under
`omnibias.pinn.jax.<X>` whose outputs are bit-equal to the torch
output for the same `(seed, weights, coordinates, dtype=float64)`.
Verified per release by
`packages/omnibias-pinn/tests/cross_backend/` (620 / 620 pass at
v0.1.0).

### Closed-form derivative orders (omnibias-pinn)

| Field | Spatial | Time |
| --- | --- | --- |
| `OneLayerVectorField` | unbounded (Riccati activations) | unbounded (single hidden layer) |
| `SpectralVectorField` | unbounded (diagonal Fourier multipliers) | closed-form for `time_depth=1`; `torch.func.jacrev`+`vmap` (or `jax.jacrev`+`vmap`) for `time_depth>1` |
| `ChebyshevVectorField` | unbounded (banded differentiation matrices) | same as `SpectralVectorField` |

## Extension packages

The following packages are product-facing scientific extensions, but
they are not part of the curated-core stability contract. They follow
their own alpha/beta gates until they satisfy the same documentation,
test, and cross-backend parity standards as the curated core.

### omnibias-fields (v0.1.0, Beta)

`omnibias-fields` is the shared backend-agnostic substrate consumed by
`omnibias-pinn`, `omnibias-geometry`, `omnibias-score`, and the
gauge / qpinn certified stacks. It is published as **Beta** for v0.1.0.

| Submodule | Stability | Notes |
| --- | --- | --- |
| `_core.{state, components, coordinates, view, registry, base, quadrature}` | stable | `FieldState`, `ComponentSpec`, `CoordinateSpec`, `ComponentView`, `VectorView`, `SigmaCache`, `FieldBase`, `ops_registry`. |
| `torch.ops.{value, derivative, gradient, divergence, laplacian, hessian, jacobian, curl, integrate, inner_product, l2_norm, sobolev_norm, tensor_divergence, dz, dzbar}` | stable | All closed-form, dispatching through the `_omnibias_dispatch` class marker. |
| `jax.ops.{...}` | stable | Bit-identical twin of the torch op surface; verified to float64 ULP tolerance. |

#### Cross-backend bit-parity (omnibias-fields)

Every torch op has a bit-identical JAX twin verified per release by
`packages/omnibias-fields/tests/test_cross_backend.py`. Same `omnibias.core`
polynomial coefficients on every backend, by construction.

### omnibias-geometry (v0.2.0, Beta)

`omnibias-geometry` provides differential geometry on manifolds with
torch + JAX bit-parity, built on `omnibias-fields`. Promoted from Alpha to
**Beta** in v0.2.0.

| Submodule | Stability | Notes |
| --- | --- | --- |
| `_core.{ManifoldSpec, MetricSpec, ChartSpec}` | stable | The pullback-metric spec `g = JᵀhJ` of an analytic-or-neural chart `φ: ℝᵈ → ℝⁿ`. |
| `torch.ops.{metric, inverse_metric, volume_element, christoffel, covariant_derivative, laplace_beltrami, riemann_tensor, ricci_tensor, scalar_curvature, geodesic_rhs}` | stable | Closed-form sigma-tower for field derivatives; exact forward-mode autodiff for metric derivatives. See [Scope & guarantees § 1](scope-and-guarantees.md#1-the-three-kinds-of-exact). |
| `torch.exterior.{exterior_derivative, wedge, hodge_star, codifferential}` | stable | Riemannian exterior calculus; tested against analytic identities. |
| `jax.{ops, exterior}` | stable | Bit-identical twin of the torch surface; cross-backend parity at `rtol = 1e-9`. |

#### Validation (omnibias-geometry)

Every operator is checked against (1) the analytic round sphere
(`scalar curvature = 2/R²`, `Ricci = g/R²`, eigenfunction
`Δ_{S²} cos(θ) = -2 cos(θ)`), (2) a sympy symbolic reference, and
(3) torch ↔ JAX cross-backend parity at `rtol = 1e-9` in float64.

### omnibias-qpinn (v0.0.2a1, Alpha)

`omnibias-qpinn` builds quantum PINN residuals and hard-conservation
cages on top of `omnibias-pinn`.

| Submodule | Stability | Notes |
| --- | --- | --- |
| `_core.complex`, `_core.spinor`, `_core.units` | experimental | Backend-agnostic encodings for complex wavefunctions, spinors, and atomic units. |
| `torch.equations` / `jax.equations` | experimental | TISE, TDSE, NLS / Gross-Pitaevskii, Helmholtz, Klein-Gordon, and Dirac residuals. |
| `torch.cage` / `jax.cage` | experimental | Norm, Bloch-periodic, and Hermitian-operator constraints. |
| `torch.diagnostics` / `jax.diagnostics` | experimental | Energy, variance, norm drift, probability-current, and continuity diagnostics. |
| `torch.eigensolvers` | experimental, torch-only | Galerkin eigensolver helpers; document as backend-specific until a JAX twin exists. |

Alpha status means API names can shift between patch releases. Public
promotion requires locked benchmark numbers and parity coverage for any
backend-specific helpers.

### omnibias-curvature (v0.1.0a1, Alpha)

`omnibias-curvature` exposes closed-form parameter-gradient, Hessian,
Gauss-Newton Fisher, Newton-step, and KFAC-factor helpers for one-layer
Riccati fields.

| Submodule | Stability | Notes |
| --- | --- | --- |
| `one_layer` | experimental | Validated against `jax.grad` / `jax.hessian` for the one-layer field case. |

Alpha status means the one-layer API is usable for research, but the
package is not stable until multi-layer assembly, optimizer integration,
and backend coverage are settled.

## Reproducibility

Release validation and large benchmark archives are kept outside the public package tree. The public stability contract is enforced by package tests, cross-backend parity tests, and documented API tiers.
