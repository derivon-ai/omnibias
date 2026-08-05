# omnibias-qpinn

> **Quantum-physics-informed neural networks with closed-form n-th derivative operators.**

`omnibias-qpinn` is the quantum sibling of [`omnibias-pinn`](../omnibias-pinn): the
same typed `FieldState` / `ops` / cage surface, specialised to the partial
differential equations of quantum physics. Every equation residual reduces to
closed-form derivatives of the wavefunction via `omnibias.pinn` fields, so the
kinetic operator `T = -hbar^2/(2m) * Laplacian` is bit-stable at every order
and the higher-order time / mass / coupling terms in nonlinear / relativistic
extensions inherit the same stability.

## Status

**Alpha (0.0.2a1).** Iterating toward Beta. The plan and promotion-gate
checklist live alongside the package layout. Until a stable release, public API
shape may shift between patch releases.

## What ships in the current alpha

| Equation | Class | Function | Notes |
|---|---|---|---|
| Time-independent Schrodinger | `TISE` | `tise` | with optional learnable eigenvalue E |
| Time-dependent Schrodinger   | `TDSE` | `tdse` | split-real `(psi_re, psi_im)` encoding |
| Nonlinear Schrodinger / GP   | `NLS`  | `nls`  | adds `g |psi|^2 psi` to TDSE |
| Helmholtz                    | `Helmholtz` | `helmholtz` | `(Laplacian + k^2) psi = 0` |
| Klein-Gordon                 | `KleinGordon` | `klein_gordon` | optional `phi^4` self-interaction |
| Dirac                        | `Dirac` | `dirac` | 4-spinor; Dirac and Weyl representations |

Cages and Hermitian helpers (hard-conservation layers, by-construction
invariants):

| Cage / helper | API | Invariant |
|---|---|---|
| Norm conservation | `make_norm_conservation_field` (hard) / `norm_loss` (soft) | `integral |psi|^2 = 1` |
| Bloch periodic    | `make_bloch_periodic_field`                                | `psi(x+a) = e^(ika) psi(x)` (up to 2nd-order derivatives in the current alpha) |
| Nuclear cusp      | `make_nuclear_cusp_field`                                  | Kato electron-nucleus cusp `u'(0) = -Z` (Pade factor, closed-form to 2nd order) |
| Hermitian helpers | `hermitian_projection(M)` / `hermiticity_loss(M)`          | `O = 0.5 (M + M^T)` projection or soft loss |

### Molecular electronic structure (`omnibias.qpinn.{torch,jax}.molecular`)

Closed-form Born-Oppenheimer local-energy pieces: the bare Coulomb potential
(`coulomb_potential`), the drift-form local kinetic energy
`T_L = -1/2 (lap log|psi| + |grad log|psi||^2)` (`local_kinetic_energy` /
`local_energy` / `molecular_local_energy`), a `MolecularHamiltonian`, and
`log_psi_derivatives` (exact `(grad, lap) log|psi|` of an MLP via the closed-form
jet tower). The hydrogen-atom oracle `E_L = -Z^2/2` and harmonic-trap oracle
`E_L = D*w/2` are reproduced exactly. **Not** closed-form (never claimed): VMC
sampling, SCF/HF/CI/CC self-consistency, and Gaussian-basis ERI stay `numerical`;
the direct Galerkin eigensolver is a bounded numerical Rayleigh-Ritz quotient.

## Install

```bash
pip install omnibias-qpinn[torch]    # PyTorch backend
pip install omnibias-qpinn[jax]      # JAX backend
pip install omnibias-qpinn[all]      # both
```

## 30-second tour (torch backend)

```python
import torch
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.equations import TDSE

coord = CoordinateSpec(("x", "t"))
spec = make_psi_components(name="psi")                       # ("psi_re", "psi_im") + group "psi"
field = OneLayerVectorField(
    coordinate_spec=coord, components=spec,
    hidden=32, base="gaussian", dtype=torch.float64,
)

xs = torch.linspace(-3.0, 3.0, 33, dtype=torch.float64)
ts = torch.linspace(0.0, 1.0, 9,  dtype=torch.float64)
grid = torch.cartesian_prod(xs, ts)

state = field(grid)
residual = TDSE(hbar=1.0, mass=1.0, potential=lambda s: 0.5 * s.coords[..., 0]**2)
out = residual(state)                                        # TDSEOutput(residual, diag)
loss = (out.residual ** 2).sum(dim=-1).mean()
```

The `omnibias.qpinn.jax` namespace mirrors this surface byte-for-byte;
`tests/cross_backend/` asserts the residuals are numerically identical
(`rtol = 1e-9`, `atol = 1e-12` in float64).

## Why omnibias-qpinn?

Every quantum PDE on this list is a 1st-order-in-time, 2nd-order-in-space (or
biharmonic, or higher) equation acting on a possibly complex wavefunction.
`omnibias-pinn` already provides bit-stable closed-form derivatives of every
order via the Riccati / Hermite / trigonometric activation towers. The only
missing pieces were:

1. A natural way to encode complex psi via two real channels (`omnibias.qpinn._core.complex`).
2. Spinor structure with Pauli / gamma matrices (`omnibias.qpinn._core.spinor`).
3. The equation-residual + cage modules wired through `state.ops.*`.

That is exactly what this package adds.

## Conservative tier (no FermiNet, no ferminet bridge)

This package depends only on `omnibias-core` and `omnibias-pinn`. A future
`omnibias-qpinn[ferminet]` extra will plumb the many-body antisymmetric branch
through to the FermiNet stack; that is *not* in the current alpha.

## Documentation

- `docs/api/qpinn.md` - API reference (mkdocstrings).
- `docs/cookbook/qpinn-tise-qho.md` - 1D harmonic-oscillator ground state (TISE).
- `docs/cookbook/qpinn-tdse.md` - free-particle Gaussian wavepacket (TDSE).
- `docs/cookbook/qpinn-gp-soliton.md` - dark-soliton Gross-Pitaevskii equation.
- `docs/cookbook/qpinn-headline-demos.md` - the two Science-worthy
  S-tier demos (NH3 inversion-tunneling splitting + 2D Abrikosov
  vortex lattice) available in the internal benchmark archive (not
  shipped publicly; see [`docs/benchmarks.md`](../../docs/benchmarks.md)).
- `QPINN_DERIVATIONS.md` - per-equation split-real residual derivations.

## Compute discipline

Tests are layered so the dev box never runs anything heavy:

| Tier | Where | Budget | Hardware |
|---|---|---|---|
| Unit | `tests/{_core,torch,jax}/` | <=5 s each | CPU float64 |
| Cross-backend parity | `tests/cross_backend/` | <=15 s each | CPU float64 |
| Smoke integration | `tests/integration/` | <=60 s each | CPU float64 (loose tolerances) |
| Full benchmark | internal benchmark archive | minutes-hours | GPU |

Headline-quality numbers always go to a GPU cluster; the in-package
smoke tests exist only to detect regressions, not to publish.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
