# 02-11 Transfer-matrix networks for layered media

## 1. Thesis and status

A stack of parallel hyperplanes is a **layered medium**: propagation through it
is a product of `2 x 2` transfer matrices, so a network whose layers are
scattering matrices inherits exact physics (energy conservation, reciprocity,
Bloch band structure) as algebraic identities rather than as learned
approximations.

- **Status**: concept
- **Depends on**: 01-02, 02-05
- **Blocks**: 05-01, 07-07

## 2. Where it lands

`packages/omnibias-pinn/src/omnibias/pinn/layered/` with torch and jax twins.
The transfer-matrix algebra itself is small and pure, so
`packages/omnibias-core/src/omnibias/core/transfer.py` holds it.

## 3. Prior art in omnibias

- Spec 02-05's `MultiInterfaceField` — parallel interfaces with transmission
  conditions. This spec is its frequency-domain, wave-propagation counterpart.
- `omnibias.geometry.gauge.transfer` — `certified_transfer_matrix_gap`, a
  certified spectral gap for a *fixed* transfer matrix at one lattice spacing in
  finite dimension, with `continuum_claim = False` already fixed in its honesty
  text. Different physics (lattice gauge theory), same algebraic object, and the
  honesty pattern to copy.
- `omnibias.core.verified.eig_operator` — `lehmann_maehly_lower_bounds`,
  `certified_spectral_gap`, `interval_ldlt_inertia`, for certified band-edge
  statements.
- `omnibias-shape` — soft occupancy fields, useful for layer geometry in an
  inverse-design loop.

**Confirmed gap.** No wave-propagation transfer-matrix machinery for layered
media. The gauge module's transfer matrix is a different object in a different
context.

## 4. Mathematics

### The transfer matrix

For the Helmholtz equation `u'' + k^2(x) u = 0` with `k` piecewise constant,
the field in layer `j` is `u = A_j e^{i k_j x} + B_j e^{-i k_j x}`. Continuity of
`u` and `u'` at an interface gives a `2 x 2` matrix relating `(A, B)` across it,
and propagation through a layer of thickness `d_j` is diagonal:

```
P_j = [[ e^{i k_j d_j}, 0 ], [ 0, e^{-i k_j d_j} ]]
M_{j -> j+1} = (1/2) [[ 1 + r, 1 - r ], [ 1 - r, 1 + r ]],   r = k_j / k_{j+1}
```

The whole stack is `M = M_N P_N ... M_1 P_1`, and reflection and transmission
coefficients read off its entries.

### The identities that come for free

| Property | Algebraic statement | Why it matters |
|---|---|---|
| energy conservation (lossless) | `det M = 1` and a `J`-unitarity relation | `|r|^2 + |t|^2 = 1` exactly |
| reciprocity | a symmetry of `M` | forward and reverse transmission agree |
| Bloch bands (periodic stack) | `|trace M_cell| <= 2` is the pass band | band edges are exactly `|trace| = 2` |
| evanescence | `|trace| > 2` | stop bands |

A network that parameterizes layer properties and composes transfer matrices
**cannot violate these**, because they are properties of the matrix product, not
of the fit. A generic neural surrogate of a scattering problem violates energy
conservation by a few percent and has to be regularized toward it. That
difference is the architectural argument.

### The Bloch condition and band structure

For a periodic stack with cell matrix `M_c`, Floquet-Bloch theory gives

```
cos( q Lambda ) = (1/2) trace M_c
```

with `q` the Bloch wavenumber and `Lambda` the period. So the band structure is a
*scalar function of the trace*, and its derivatives with respect to layer
parameters are closed form. Band-gap optimization becomes differentiation of a
trace, not a spectral solve.

### WKB and the smooth limit

When layer properties vary slowly, the discrete product tends to a WKB
approximation, and the connection formulae at turning points are the classical
Airy matching. This is where the omnibias tower reappears: the smooth profile can
be an OMBU multi-interface field (spec 02-05), and its transverse derivatives —
which the WKB expansion needs to all orders — are closed form. So the
architecture spans the discrete-layer and smooth-profile regimes with one
parameterization.

### Certified band edges

Band edges are where `|trace M_c| = 2`. With interval arithmetic on the matrix
entries, `trace M_c` is enclosed, so a statement like "there is a stop band
covering `[omega_lo, omega_hi]`" becomes a sound enclosure claim. The tooling for
this already exists in `omnibias.core.verified`; what is new is applying it to a
layered-media transfer matrix.

## 5. Worked example

A quarter-wave stack: two materials with refractive indices `n_1 = 1.5`,
`n_2 = 2.5`, each layer a quarter wavelength at the design frequency.

At the design frequency, each layer contributes a phase of `pi/2`, so
`P_j = [[i, 0], [0, -i]]`. For a two-layer cell the standard result is

```
M_c = [[ -n_2/n_1, 0 ], [ 0, -n_1/n_2 ]]
trace M_c = -( n_2/n_1 + n_1/n_2 ) = -( 2.5/1.5 + 1.5/2.5 )
          = -( 1.6666667 + 0.6 ) = -2.2666667
```

`|trace| = 2.2667 > 2`, so the design frequency is inside a **stop band** — which
is exactly what a quarter-wave mirror is for. The check is one arithmetic
expression, no eigensolve.

Band edge: the edges are where `|trace| = 2`. Sweeping frequency `omega` scaled
by the design frequency `omega_0`, the phase per layer is
`phi = (pi/2)(omega/omega_0)` and

```
(1/2) trace M_c = cos^2 phi - (1/2)( n_2/n_1 + n_1/n_2 ) sin^2 phi
                = cos^2 phi - 1.1333333 sin^2 phi
```

Setting this to `-1` (the band edge on the negative side):

```
cos^2 phi - 1.1333333 (1 - cos^2 phi) = -1
2.1333333 cos^2 phi = 0.1333333
cos^2 phi = 0.0625,   cos phi = +-0.25,   phi = 1.3181161 rad
omega / omega_0 = phi / (pi/2) = 0.8390...
```

so the stop band runs from `omega/omega_0 = 0.8390` to `1.1610` by symmetry: a
fractional bandwidth of `0.322`. The classical formula for a quarter-wave stack
gives `(2/pi) arcsin( (n_2 - n_1)/(n_2 + n_1) ) = (2/pi) arcsin(0.25) = 0.1609`
for the half-width, that is `0.3218` full — agreeing to three digits.

The point: a **closed-form band structure, differentiable in `n_1` and `n_2`**,
with no spectral solve anywhere. Optimizing bandwidth is now a two-parameter
differentiable problem.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/core/transfer.py
@dataclass(frozen=True)
class Layer:
    index: complex          # refractive index or wave speed
    thickness: float

def interface_matrix(n_lo: complex, n_hi: complex) -> Matrix2: ...
def propagation_matrix(n: complex, thickness: float, omega: float) -> Matrix2: ...
def stack_matrix(layers: Sequence[Layer], omega: float) -> Matrix2: ...
def reflection_transmission(M: Matrix2) -> tuple[complex, complex]: ...
def bloch_dispersion(cell: Sequence[Layer], omega: float) -> float:
    """(1/2) trace M_cell; |value| <= 1 is a pass band."""
def band_edges(cell, *, omega_range, tol) -> tuple[tuple[float, float], ...]: ...
def certified_band_gap(cell, *, omega_range) -> BandGapCertificate:
    """Interval enclosure of the trace over the range; sound stop-band claim."""
def unitarity_residual(M: Matrix2) -> float:
    """|det M - 1| for lossless stacks; must be at rounding level."""
```

```python
# omnibias/pinn/layered/torch.py  (and jax twin)
class TransferStack(nn.Module):
    """Layer parameters are learnable; the physics identities are structural."""
    def __init__(self, n_layers: int, *, lossless: bool = True, dtype=None) -> None: ...
    def forward(self, omega: Tensor) -> tuple[Tensor, Tensor]:   # (r, t)
        ...
    def band_structure(self, omega: Tensor) -> Tensor: ...
```

Complex arithmetic must respect the default dtype policy: complex64 with a
float32 default, complex128 with float64.

## 7. Practical use cases

1. **Optical coating design.** Antireflection coatings, dielectric mirrors,
   filters: the classical application, now differentiable end to end with exact
   energy conservation.
2. **Photonic and phononic crystals.** Band-gap engineering by differentiating a
   trace, with certified gap statements.
3. **Seismic layered inversion.** Recovering layer velocities and thicknesses
   from surface reflections (spec 05-01).
4. **Acoustic metamaterials.** Same algebra, different constitutive relations.
5. **Thermal and diffusive layered problems**, where the transfer matrix is real
   and the "band" structure is a decay-rate structure.

## 8. Acceptance gates

Baselines: a plain MLP surrogate of the scattering problem, and a dense
finite-element or finite-difference Helmholtz solve at matched cost.

- **G1 structural identities.** `unitarity_residual` is at rounding level for
  every lossless stack, including randomly initialized untrained ones, and
  `|r|^2 + |t|^2 = 1` to `<= 1e-13`. This must hold before training, since it is
  structural.
- **G2 band-structure accuracy.** Computed band edges for the quarter-wave stack
  match the classical closed-form result to `<= 1e-10` relative.
- **G3 certified gaps.** `certified_band_gap` never claims a gap where a dense
  frequency scan plus a random sample finds a propagating mode, and every
  claimed gap is confirmed by the scan.
- **G4 inverse-design win.** On a bandwidth-maximization problem, the
  differentiable trace reaches a better optimum than a gradient-free search at
  `10x` fewer objective evaluations, over five seeds.
- **G5 surrogate comparison, honest.** The MLP surrogate's energy-conservation
  violation is reported alongside its accuracy, so the structural advantage is
  quantified rather than asserted.
- **G6 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/transfer_stack.py`: identity checks, band-structure validation,
  inverse-design study, MLP-surrogate comparison including its conservation
  violation.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/layered/`.

## 10. Honesty and scope

- Neither collapse limit appears in the core transfer-matrix algebra. Where a
  smooth profile is used (the WKB regime), its basis comes from the founding
  bias collapse (`delta -> 0`), and the interface-sharpening scale `alpha` of
  spec 02-05 is a third, distinct limit that is not a collapse.
- **This is one-dimensional layered propagation.** Oblique incidence adds a
  transverse wavenumber and still works; genuinely two- or three-dimensional
  scattering does not reduce to a `2 x 2` product, and the spec must not imply
  it does.
- Structural identities hold for **lossless, reciprocal, linear** media. Loss,
  nonlinearity and gain each break one of them, and the implementation should
  refuse to advertise `unitarity_residual` outside the lossless case.
- Certified band-gap claims are **sound enclosures** over a stated frequency
  range with a stated discretization of the layer parameters. Following the
  pattern already used by `certified_transfer_matrix_gap`, they are
  finite-dimensional statements about a specific stack, and no continuum or
  infinite-stack claim is implied.

## 11. Open questions and risks

- **Numerical conditioning of the product.** Long stacks with evanescent layers
  produce exponentially large and small entries; the scattering-matrix
  formulation is the standard stable alternative and may be required, at the
  cost of a less transparent algebra.
- **Complex differentiability.** Gradients through complex matrix products need
  care in both frameworks; Wirtinger derivatives are available in
  `omnibias.fields`, and the convention must be fixed once.
- **Dispersion.** Frequency-dependent indices make the layer matrices
  frequency-dependent in a way that complicates band-edge finding. Support it
  explicitly or refuse it explicitly.
- **Falsifier.** If the MLP surrogate, regularized toward energy conservation,
  matches the structural model on inverse design at equal cost, the structural
  argument is weaker than claimed.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/src/omnibias/core/transfer.py`
- [ ] `packages/omnibias-pinn/src/omnibias/pinn/layered/` torch and jax twins
- [ ] Structural identity tests on untrained random stacks
- [ ] Band-edge validation against the classical quarter-wave formula
- [ ] Certified band-gap soundness test (dense scan plus random sample)
- [ ] Stability study: transfer versus scattering formulation on long stacks
- [ ] Fixed Wirtinger convention, documented, with a gradient test
- [ ] `benchmarks/transfer_stack.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
