# 02-06 Potential theory and BEM-Net

## 1. Thesis and status

Treat `sigma'` as a smoothed Green kernel and the `integral` role as a
single-layer potential: a boundary-integral network that solves exterior
Laplace and Helmholtz problems by learning **densities on surfaces** rather than
fields in volumes, with the layer potentials evaluated in closed form.

- **Status**: concept
- **Depends on**: 01-05, 01-12, 02-07
- **Blocks**: 05-01

## 2. Where it lands

`packages/omnibias-pinn/src/omnibias/pinn/bem/` with torch and jax twins. A
submodule: same audience as the PDE solvers, and it depends on the field
substrate.

## 3. Prior art in omnibias

- `docs/operator-surface.md` — the `integral` role, an antiderivative window
  `S(z + b_hi) - S(z + b_lo)` with `S' = sigma`, giving closed-form line
  integrals of the activation.
- `omnibias.core.verified.line` and `hardy_line` — the Poisson kernel
  `p_a(x) = a/(x^2+a^2)`, the conjugate `q_a(x) = x/(x^2+a^2)`, their Hilbert
  relations and derivatives; the Poisson kernel *is* the half-plane Green
  function's normal derivative, so half of classical potential theory is already
  present in verified form.
- `omnibias.fields` — `integrate`, `inner_product`, and the field-operator
  surface.
- `omnibias.pinn.domain` — SDF machinery, which supplies the surface geometry
  (normals, distances) a boundary method needs.
- Spec 01-12's conjugate tower — exact `H` on the line, which is precisely the
  Dirichlet-to-Neumann operator for the half-plane.

**Confirmed gap.** No boundary-integral or layer-potential machinery exists. All
PDE solvers in the repo are volumetric collocation or weak-form methods.

## 4. Mathematics

### Layer potentials

For the Laplace equation with free-space Green function `G`, the single- and
double-layer potentials of a density `phi` on a surface `Gamma` are

```
(S phi)(x) = integral_Gamma G(x - y) phi(y) ds(y)
(D phi)(x) = integral_Gamma dG/dn_y (x - y) phi(y) ds(y)
```

Any `S phi` or `D phi` satisfies the PDE **exactly** in the complement of
`Gamma`, for any `phi`. That is the structural advantage: the differential
equation is satisfied by construction, and only the boundary condition has to be
imposed. A volumetric PINN spends most of its capacity satisfying the PDE; a
boundary method spends none.

### The omnibias connection

Two links, both concrete.

**The half-plane case is already verified.** For the upper half-plane, the
Poisson kernel `p_a` is the harmonic extension kernel and the Dirichlet-to-
Neumann map is the Hilbert transform. Both are in
`omnibias.core.verified.line` and `hardy_line` with exact relations
(`H[p_a] = q_a`, `H[q_a] = -p_a`), and spec 01-12 extends them to every
derivative order. So for the half-plane, BEM-Net is not an approximation: the
operators are closed form and interval-enclosable.

**Smoothed kernels from the tower.** For general surfaces, `sigma'` at scale
`alpha` is a mollified point source (spec 01-05), so

```
G_alpha(r) = (the antiderivative window applied to a radial argument)
```

is a regularized Green kernel with a known mollification order. Regularization is
what makes the boundary integral non-singular, which is normally the hardest
part of a BEM implementation. Here the regularization has a *stated order* from
the mollifier moments, so the regularization error is not a tuning parameter but
a computed quantity.

### Density representation

The density `phi` lives on the surface. Two natural parameterizations:

1. **Scan bank along the surface parameter** (spec 01-02), which is a
   translation-equivariant density basis.
2. **Multi-pack at chosen surface locations** (spec 01-01), which places
   resolution where the geometry has features (corners, edges).

Both give closed-form derivatives of the density, which the hypersingular
operators need.

### Cost

A dense boundary-integral operator is `O(N^2)` for `N` surface points, which is
why BEM needs a fast method. Spec 02-07's hierarchical pack tree is exactly the
near-field / far-field split that a fast multipole method performs, and it is
the reason 02-07 is a dependency here rather than a nicety.

### Where it applies and where it does not

BEM applies to **linear, constant-coefficient, homogeneous** problems where a
Green function exists: Laplace, Helmholtz, Stokes, linear elasticity. It does
not apply to nonlinear or variable-coefficient problems without extra
machinery (volume potentials, which reintroduce the volume). State this plainly:
BEM-Net is a specialist, and its value is being excellent on exterior problems
in unbounded domains, where volumetric methods are worst.

## 5. Worked example

Exterior Dirichlet problem for Laplace in 2-D: find `u` harmonic outside the
unit disc with `u = g` on the circle and `u -> 0` at infinity.

Take `g(theta) = cos(theta)`. The exact solution is

```
u(r, theta) = cos(theta) / r
```

(the dipole field), since `cos(theta)/r` is harmonic and equals `cos(theta)` at
`r = 1`.

Single-layer representation: `u = S phi` with `phi` on the circle. For this
geometry the single-layer operator diagonalizes in Fourier modes: for
`phi = e^{i n theta}`,

```
(S phi)(r, theta) = ( 1 / (2 |n|) ) r^{-|n|} e^{i n theta},   n != 0
```

So `phi = 2 cos(theta)` gives exactly `u = cos(theta)/r`. The density is a
single Fourier mode, one parameter.

A BEM-Net with a scan bank of 16 offsets along `theta` recovers this density to
machine precision because the mode is in the span, and the resulting field
satisfies Laplace **exactly everywhere outside the disc** — not to `1e-6`, but
exactly, because every single-layer potential is harmonic by construction.

Contrast with a volumetric PINN on the same problem. The exterior domain is
unbounded, so the PINN needs a truncation radius `R` and an artificial boundary
condition there. At `R = 10` the truncation error in `u` is `O(1/R) = 0.1` near
the artificial boundary, and the PDE residual is never exactly zero anywhere.
The comparison is not close, and it is the honest reason to build this: **the
exterior problem is where volumetric methods are structurally weakest.**

The catch, stated: this example is analytically diagonalizable. On a general
surface, the operator is dense and the win depends entirely on the fast method
of spec 02-07 working.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/pinn/bem/_core.py
@dataclass(frozen=True)
class Surface:
    sdf: SDF                     # reuse omnibias.pinn.domain
    parameterization: Literal["arclength", "sdf_projection"]
    quadrature: QuadratureSpec

@dataclass(frozen=True)
class KernelSpec:
    equation: Literal["laplace", "helmholtz", "stokes"]
    dimension: int
    wavenumber: complex | None = None
    regularization: MollifierSpec | None = None   # spec 01-05; None = exact
```

```python
# omnibias/pinn/bem/torch.py  (and jax twin)
class BEMNet(nn.Module):
    def __init__(
        self, surface: Surface, kernel: KernelSpec, *,
        density: Literal["scan", "multipack"] = "scan",
        layers: Literal["single", "double", "combined"] = "combined",
        dtype=None,
    ) -> None: ...
    def density(self, s: Tensor) -> Tensor: ...
    def evaluate(self, x: Tensor) -> Tensor:
        """Field at off-surface points. Satisfies the PDE by construction."""
    def boundary_residual(self, s: Tensor) -> Tensor:
        """The only thing that has to be trained."""

def half_plane_dtn(u_boundary: Tensor, *, order: int = 0) -> Tensor:
    """Dirichlet-to-Neumann for the half plane via the exact conjugate tower
    (spec 01-12). Closed form, no quadrature."""
```

## 7. Practical use cases

1. **Exterior scattering.** Acoustic and electromagnetic scattering in unbounded
   domains, where truncation is the dominant error for volumetric methods.
2. **Free-space electrostatics and magnetostatics**, including capacitance and
   inductance extraction.
3. **Stokes flow around bodies**, where the boundary-integral formulation is the
   classical method of choice.
4. **Shape optimization.** The unknowns live on the surface, so shape
   derivatives are natural and cheap.
5. **Half-plane and layered problems**, where the exact conjugate tower makes
   the Dirichlet-to-Neumann operator closed form and enclosable.

## 8. Acceptance gates

Baselines: a volumetric PINN with a truncated domain and an absorbing or
Dirichlet artificial boundary, at matched parameter count and wall time; and a
classical dense BEM with the same number of surface unknowns.

- **G1 exact PDE satisfaction.** Off-surface, the PDE residual of `evaluate` is
  at rounding level (`<= 1e-13` relative to the field magnitude) for every
  density, including untrained random ones. This is a structural property and
  must be tested as one.
- **G2 boundary accuracy.** On the exterior Dirichlet disc problem, relative
  `L2` error of the field on a test annulus `<= 1e-8`, with skill `> 0` against
  the zero predictor.
- **G3 exterior win.** On an unbounded-domain problem, BEM-Net beats the
  truncated volumetric PINN by at least `100x` in far-field relative error at
  matched cost, over five seeds.
- **G4 regularization order.** The measured regularization error decays at the
  order predicted by the mollifier's moment conditions, over three halvings.
- **G5 half-plane exactness.** `half_plane_dtn` matches the analytic
  Dirichlet-to-Neumann map to `<= 4 ulp` on the test set, using the conjugate
  tower rather than quadrature.

## 9. Benchmark plan

- `benchmarks/bem_net.py`: three arms, three geometries (disc, ellipse,
  two-body), plus a scaling study on `N` surface points that records where the
  dense `O(N^2)` cost becomes the bottleneck.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/bem/`.

## 10. Honesty and scope

- Kernels are built from the founding bias collapse (`delta -> 0`) and the
  antiderivative window. No temperature collapse appears.
- **BEM applies only to linear, constant-coefficient, homogeneous problems.**
  This is a hard structural restriction, not a current limitation. Say it in the
  first paragraph of any user-facing document.
- "Satisfies the PDE exactly" means *off the surface, by construction*. It is
  true and worth saying, and it must always be paired with "the boundary
  condition is what is approximated", or it reads as a solved-PDE claim.
- The disc example is analytically diagonalizable and is a validation case, not
  evidence of general performance. General surfaces need spec 02-07 to be
  affordable.
- Certificate tier: sound enclosure for the half-plane operators via the
  verified line modules; everything else is empirical.

## 11. Open questions and risks

- **Dense operator cost.** Without a fast method this is `O(N^2)` and will lose
  to volumetric methods at moderate `N`. The dependency on 02-07 is real, and if
  02-07 fails, this spec is limited to small surfaces.
- **Corners and edges.** Boundary-integral operators lose regularity at
  geometric singularities; graded densities (multi-pack placement) help but do
  not eliminate the problem.
- **Helmholtz resonances.** Single-layer formulations fail at interior
  resonance wavenumbers; the combined-layer formulation is standard and must be
  the default rather than an option.
- **Falsifier.** If, on a realistic exterior problem at matched cost, the
  truncated volumetric PINN gets within `10x`, the structural advantage is not
  translating into practice.

## 12. Implementation checklist

- [ ] `packages/omnibias-pinn/src/omnibias/pinn/bem/_core.py`
- [ ] torch and jax twins with a parity test
- [ ] Reuse `omnibias.pinn.domain` SDFs for geometry; no second surface library
- [ ] Reuse the verified line kernels for the half-plane path
- [ ] Structural test: PDE residual at rounding level for random densities
- [ ] Regularization-order rate test
- [ ] Combined-layer default with a resonance test
- [ ] `benchmarks/bem_net.py` plus smoke JSON and a scaling table
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
