# omnibias-qpinn

**Quantum-physics-informed neural networks with closed-form n-th
derivative operators**, built on top of `omnibias-pinn`. Cross-backend
(PyTorch + JAX) residuals for Schrodinger / Gross-Pitaevskii /
Helmholtz / Klein-Gordon / Dirac, plus hard-conservation cages for
norm, Bloch periodicity, and operator Hermiticity.

The package surfaces three layers:

| Layer | Purpose | Example |
| --- | --- | --- |
| **_core** | Backend-agnostic encoding helpers + spinor / atomic-units constants | `make_psi_components`, `make_spinor_components`, `gamma_matrices` |
| **equations** | Prebuilt PDE residuals returning `NamedTuple` outputs | `TISE`, `TDSE`, `NLS`, `Helmholtz`, `KleinGordon`, `Dirac` |
| **cage** | Hard-conservation layers (norm, Bloch, Hermitian) + soft loss helpers | `NormConservationField`, `BlochPeriodicField`, `hermitian_projection` |

Plus diagnostics:

* **Diagnostics** (`diagnostics/`): variational `expected_energy`,
  `expectation_value`, `energy_variance`, probability current /
  divergence / continuity residual, and `norm_squared` / `norm_drift`.

## Encoding choices

* **Complex wavefunctions**: every complex :math:`\psi(x) =
  \psi_R(x) + i\,\psi_I(x)` is encoded as two **real** components
  ``psi_re``, ``psi_im`` bundled in a single :class:`ComponentSpec`
  group. Build the spec with [`make_psi_components`][omnibias.qpinn.make_psi_components].
* **Spinors**: an n-spinor is encoded as :math:`2n` real components.
  Build the spec with [`make_spinor_components`][omnibias.qpinn.make_spinor_components].
  The Pauli / gamma matrices live in
  :mod:`omnibias.qpinn._core.spinor`.

This keeps the existing `omnibias.pinn._core.{ComponentSpec, FieldState}`
unchanged; the closed-form derivative path is reused without
modification.

## 30-second tour (PyTorch backend)

```python
import torch
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.equations import TDSE
from omnibias.qpinn.torch.cage import make_norm_conservation_field

coord = CoordinateSpec(("x", "t"))
spec = make_psi_components(name="psi")
base = OneLayerVectorField(
    coordinate_spec=coord, components=spec, hidden=64, base="gaussian",
)

quad_x = torch.linspace(-5.0, 5.0, 401, dtype=torch.float64)
quad_t = torch.zeros_like(quad_x)
quad_grid = torch.stack((quad_x, quad_t), dim=-1)
quad_w = torch.full((401,), 10.0 / 401, dtype=torch.float64)
field = make_norm_conservation_field(
    base=base, quadrature_coords=quad_grid, quadrature_weights=quad_w,
)

xs = torch.linspace(-3.0, 3.0, 33, dtype=torch.float64)
ts = torch.linspace(0.0, 1.0, 9,  dtype=torch.float64)
grid = torch.cartesian_prod(xs, ts)

state = field(grid)
out = TDSE(potential=lambda s: 0.5 * s.coords[..., 0] ** 2)(state)
loss = (out.residual ** 2).sum(dim=-1).mean()
```

The JAX backend has an identical surface under :mod:`omnibias.qpinn.jax`;
cross-backend tests in `packages/omnibias-qpinn/tests/cross_backend/`
assert numerically identical residuals (``rtol = 1e-9``, ``atol = 1e-12``
in float64).

## Top-level public API

::: omnibias.qpinn._core
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]

## Equations

The PyTorch and JAX backends expose the same names under their
respective root namespaces (`omnibias.qpinn.torch.equations` /
`omnibias.qpinn.jax.equations`).

::: omnibias.qpinn.torch.equations
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]

## Cages

::: omnibias.qpinn.torch.cage
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]

## Diagnostics

::: omnibias.qpinn.torch.diagnostics
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]

## Molecular electronic structure

The `molecular` module wires the **closed-form** Born-Oppenheimer pieces into a
usable local-energy surface: the bare Coulomb potential (electron-nucleus,
electron-electron, nucleus-nucleus), the drift-form local kinetic energy
:math:`T_L = -\tfrac12(\nabla^2\log|\psi| + \lVert\nabla\log|\psi|\rVert^2)`, and
`log_psi_derivatives`, which returns the exact :math:`(\nabla, \nabla^2)\log|\psi|`
of an MLP through the closed-form multivariate jet tower (`mlp_jet_mv` — no
autodiff, no finite differences). The hard [`NuclearCuspField`][omnibias.qpinn.torch.cage.NuclearCuspField]
cage (under **Cages**, above) multiplies any base ansatz by a Padé factor
:math:`\exp(\sum_a -Z_a s_a/(1+b_a s_a))` that enforces the Kato electron-nucleus
cusp :math:`u'(0) = -Z` in closed form.

```python
import jax.numpy as jnp
from omnibias.qpinn.jax import molecular as M

# Hydrogen-atom 1s orbital psi = exp(-Z r): log|psi| = -Z r.
Z, r_vec = 2.0, jnp.array([0.5, -0.2, 0.7])
r = jnp.linalg.norm(r_vec)
grad = -Z * r_vec / r          # grad log|psi|
lap = -2.0 * Z / r             # laplacian log|psi| (3-D)
E_L = M.molecular_local_energy(grad, lap, jnp.zeros(3), r_vec, jnp.array([Z]), n_e=1)
# E_L == -Z**2 / 2 exactly, at every electron position (zero local-energy variance).
```

!!! warning "Closed-form scope"
    The kinetic term, Coulomb potential, Padé-Jastrow derivatives, and cusp cage
    are closed-form. The *variational solution* of the many-electron Schrödinger
    equation is **not** claimed here: VMC Monte-Carlo sampling, SCF / Hartree-Fock
    / CI / coupled-cluster self-consistency, and Gaussian-basis electron-repulsion
    integrals stay iterative / stochastic `numerical` (see
    [scope & guarantees](../scope-and-guarantees.md) §6). The direct Galerkin
    eigensolver is a bounded numerical Rayleigh-Ritz quotient, bit-exact only in
    its analytic-basis limit.

::: omnibias.qpinn.torch.molecular
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]

## JAX twin

`omnibias.qpinn.jax` carries bit-identical numerics with the PyTorch
backend. Use whichever backend matches the rest of your code:

| Need | Backend |
| --- | --- |
| Drop-in with `jax.jit` / `optax` / `flax` | `omnibias.qpinn.jax` |
| Drop-in with `torch.optim` / `torch.compile` | `omnibias.qpinn.torch` |
| Cross-validation between the two | both |
