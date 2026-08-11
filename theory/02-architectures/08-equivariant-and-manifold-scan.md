# 02-08 Equivariant and manifold scan

## 1. Thesis and status

Replicate a pack template over an **orbit of normals** to get steerable
derivative features that transform predictably under rotation, and scan in
**chart coordinates** using the pullback metric so the same construction works
on a learned manifold rather than only in flat space.

- **Status**: concept
- **Depends on**: 01-02, 01-06
- **Blocks**: 05-02

## 2. Where it lands

Two pieces:

- `packages/omnibias-torch/src/omnibias/torch/scan_equivariant.py` and the jax
  twin, for the orientation-orbit bank.
- `packages/omnibias-geometry/src/omnibias/geometry/scan/`, for the chart-space
  scan, because it needs the metric machinery.

## 3. Prior art in omnibias

- Spec 01-02's `BiasScan` — the flat, single-direction bank.
- `omnibias-geometry` — metric, Christoffel symbols, covariant derivative,
  Laplace-Beltrami, curvature, geodesics, exterior calculus, and the pullback
  metric of a learned chart `g = J^T h J`.
- `omnibias.geometry.atlas` — the region-wise metric bridge onto
  `omnibias-partition`.
- `omnibias.core.polynomials` — `hermite_coeffs`, relevant because the gaussian
  family's derivative tower is the classical steerable basis.

**Confirmed gap.** No orientation bank, no steerability analysis, and no scan in
chart coordinates. The geometry package computes operators on manifolds but has
no notion of a bias bank living on one.

## 4. Mathematics

### Steerability

A family of filters is *steerable* when a filter at any orientation is a fixed
linear combination of a finite basis. The classical example: derivatives of a
gaussian. For the first derivative in direction `theta` in 2-D,

```
G_theta = cos(theta) G_x + sin(theta) G_y
```

exactly, with two basis filters. For the `n`-th order, the basis has `n + 1`
elements and the interpolation functions are trigonometric polynomials of degree
`n`.

In omnibias terms: with a **gaussian base**, an order-`n` pack along direction
`w` is a directional derivative of a gaussian, so the whole orientation orbit is
spanned by `n + 1` fixed packs. **Steerability is exact, and the coefficients are
closed form.**

For non-gaussian bases (`tanh`, logistic) this fails: `sigma^(n)(w . x)` is not a
separable function of the coordinates, so there is no finite steerable basis.
That is a real distinction and must be stated: **the steerable construction is a
gaussian-family property, not a general OMBU property.**

### The orientation bank

Where exact steerability is unavailable, use a discrete orbit:

```
w_l = R(theta_l) w_0,     l = 1 .. L
r_l(x) = F_T( w_l . x + tau )
```

and read out either the full response vector (equivariant to the discrete
rotation group `C_L`) or an invariant summary (`max`, `sum`, or the modulus of
the first Fourier coefficient along `l`).

Equivariance to `C_L` is exact: rotating the input by `2 pi / L` cyclically
shifts the response. Equivariance to continuous rotation is approximate, with
error controlled by `L` and by the angular bandwidth of the template — the same
sampling argument as spec 01-02's bank spacing, now on the circle.

### Scanning on a manifold

On a manifold with metric `g`, "slide the hyperplane along its normal" needs
three replacements:

| Flat notion | Manifold replacement |
|---|---|
| `w . x + b` | a level set of a coordinate function, or the signed geodesic distance to a hypersurface |
| translation along `w` | flow along the geodesic normal field |
| `d/dz` | the covariant derivative in the normal direction |

The cleanest realization uses the **pullback metric of a learned chart**, which
`omnibias-geometry` already computes as `g = J^T h J`. In chart coordinates the
scan is the flat construction; what changes is that

1. the direction `w` must be raised or lowered with `g` to be a covector or
   vector consistently, and
2. derivative orders beyond the first pick up Christoffel terms, so
   `sigma^(n)` along a geodesic is **not** simply the flat tower unless the
   normal field is geodesic.

The honest statement: for the first order the construction transfers exactly; for
higher orders the closed-form transverse tower is exact **in the chart** and the
conversion to covariant derivatives introduces the metric's own derivatives,
which `omnibias-geometry` computes by forward-mode autodiff of the analytic
metric and labels as such. The composite is therefore "closed-form field
derivatives, autodiff metric derivatives", the same honesty label the geometry
package already uses.

## 5. Worked example

**Steerability with a gaussian base, 2-D, order 1.**

Basis: `G_x(x, y) = -x G(x, y)` and `G_y = -y G(x, y)` with
`G = exp(-(x^2 + y^2)/2)`.

Filter at `theta = 30` degrees, evaluated at `(x, y) = (1, 0.5)`:

```
G(1, 0.5)   = exp(-(1 + 0.25)/2) = exp(-0.625) = 0.5352614
G_x         = -1   * 0.5352614 = -0.5352614
G_y         = -0.5 * 0.5352614 = -0.2676307
G_30 = cos(30) G_x + sin(30) G_y
     = 0.8660254 * (-0.5352614) + 0.5 * (-0.2676307)
     = -0.4635500 - 0.1338154 = -0.5973654
```

Direct evaluation: the directional derivative along
`w = (0.8660254, 0.5)` is `-(w . r) G = -(0.8660254 * 1 + 0.5 * 0.5) * 0.5352614
= -(1.1160254)(0.5352614) = -0.5973654`. Identical, as steerability requires.
Two stored filters give every orientation exactly.

**Non-gaussian failure.** With `sigma = tanh` and order 1, the analogous
question is whether `sigma'(w_theta . x)` lies in the span of
`sigma'(e_1 . x)` and `sigma'(e_2 . x)`. At `x = (1, 0.5)`:

```
sigma'(1)    = 1 - tanh^2(1)    = 0.4199744
sigma'(0.5)  = 1 - tanh^2(0.5)  = 0.7864477
sigma'(w_30 . x) = sigma'(1.1160254) = 1 - tanh^2(1.1160254) = 0.3501261
```

No fixed pair of coefficients reproduces `0.3501261` from `(0.4199744,
0.7864477)` across all `x`, because the function is not separable. The bank is
therefore mandatory outside the gaussian family, and the spec must say so
instead of implying general steerability.

**Chart scan.** Take a learned chart `phi : R^2 -> R^3` with induced metric
`g = J^T J`. A scan along chart direction `e_1` at a point where
`g = diag(4, 1)` covers arc length `2` per unit chart coordinate, so a bank with
chart spacing `0.1` has physical spacing `0.2` in that direction and `0.1` in
the other. Without the metric correction the bank is anisotropic in a way the
model does not know about. The correction is a single scaling by `sqrt(g_11)`,
and it is exactly what makes a chart scan comparable across the manifold.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/torch/scan_equivariant.py  (and jax twin)
@dataclass(frozen=True)
class OrientationBank:
    angles: tuple[float, ...]          # discrete orbit, or
    steerable_order: int | None = None # exact steering (gaussian family only)

class EquivariantScan(nn.Module):
    def __init__(
        self, dim: int, bank: OrientationBank, offsets: BankSpec, *,
        template: MultiPackSpec | OpName = "grad",
        base: str = "gaussian",
        readout: Literal["orbit", "max", "fourier"] = "orbit",
        dtype=None,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

def steerable_basis(order: int, dim: int) -> SteerableBasis | None:
    """Exact steering coefficients; returns None for non-gaussian bases rather
    than silently approximating."""
```

```python
# omnibias/geometry/scan/__init__.py
def chart_scan(
    chart, x: Tensor, direction: Tensor, offsets: BankSpec, *,
    metric_correction: bool = True,
) -> Tensor:
    """Scan in chart coordinates with arc-length-normalized spacing via the
    pullback metric g = J^T h J."""
```

Returning `None` from `steerable_basis` rather than an approximation is
deliberate: an approximate "steerable" basis that silently degrades is worse
than none.

## 7. Practical use cases

1. **Rotation-aware interface detection.** Interfaces have orientations;
   scanning position without orientation misses half the geometry.
2. **Anisotropic media.** Layered materials with tilted strata need the
   orientation orbit to find the stratification direction.
3. **Learned-manifold features.** Once a chart is learned (an autoencoder, a
   diffusion map), scanning in chart coordinates gives features that respect the
   learned geometry.
4. **Curvature-aware sampling.** The metric correction makes bank spacing
   comparable across a curved domain, which matters for any quadrature or
   density estimate built on the bank.
5. **Steerable feature extraction** with a gaussian base, where the exact
   `n + 1`-element basis is a large computational saving over a dense orbit.

## 8. Acceptance gates

Baselines: a plain `BiasScan` with a single learned direction, and a dense
orientation orbit at high `L`.

- **G1 exact steering.** For gaussian bases and orders `1 .. 5`, the steered
  filter matches direct evaluation to `<= 4 ulp` at every angle tested.
- **G2 honest refusal.** `steerable_basis` returns `None` for every non-gaussian
  base, and a test asserts that no approximate fallback is used.
- **G3 discrete equivariance.** Rotating the input by `2 pi / L` cyclically
  shifts the response vector to `<= 4 ulp`; off-orbit equivariance error decays
  as `L` doubles at the predicted rate.
- **G4 metric correction.** On a chart with known anisotropy, arc-length spacing
  after correction is uniform to `<= 1` percent, versus the uncorrected error
  which equals the metric anisotropy.
- **G5 task skill.** On an anisotropic interface-orientation task, the
  orientation bank beats the single-direction scan in angular error, with skill
  `> 0` against a random-angle predictor, over five seeds.

## 9. Benchmark plan

- `benchmarks/equivariant_scan.py`: steering exactness, equivariance error
  versus `L`, metric-correction validation, orientation-estimation task.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/eqscan/`.

## 10. Honesty and scope

- Templates collapse by the founding `delta -> 0` bias collapse. No temperature
  collapse appears unless a `max` readout is softened, in which case the
  sharpness parameter must be labelled as a softmax knob and its `beta -> inf`
  limit named as temperature collapse.
- **Exact steerability is a gaussian-family property.** For `tanh`, logistic and
  other Riccati bases there is no finite steerable basis, and the worked example
  shows why. Never present the orbit bank as exact steering.
- Equivariance to the discrete group `C_L` is exact; continuous rotation
  equivariance is approximate with a measured error.
- On manifolds, **field derivatives are closed form, metric derivatives are
  forward-mode autodiff of the analytic metric.** That is the label the geometry
  package already uses, and it must be carried here.
- Beyond first order, covariant and chart derivatives differ by Christoffel
  terms. Only claim the flat tower in the chart, not covariantly.

## 11. Open questions and risks

- **Cost of the orbit.** `L` orientations multiply cost by `L`. For 3-D the
  orbit is over `SO(3)` and the sampling problem is much harder; the spec is
  written for 2-D and must not be extrapolated silently.
- **Learned charts change during training**, so the metric correction is
  recomputed constantly. Measure the overhead.
- **Interaction with steerability.** A gaussian base gives exact steering but is
  not a Riccati base, so it uses a different (Hermite) tower. Check that the
  polynomial coefficients come from `hermite_coeffs` and stay shared.
- **Falsifier.** If a single learned direction plus depth matches the orbit bank
  on the orientation task, the explicit orbit is unnecessary.

## 12. Implementation checklist

- [ ] `packages/omnibias-torch/src/omnibias/torch/scan_equivariant.py`
- [ ] `packages/omnibias-jax/src/omnibias/jax/scan_equivariant.py`
- [ ] `packages/omnibias-geometry/src/omnibias/geometry/scan/`
- [ ] Exact-steering test for the gaussian family via `hermite_coeffs`
- [ ] Refusal test: `steerable_basis` returns `None` for non-gaussian bases
- [ ] Discrete-equivariance test and off-orbit rate test
- [ ] Metric-correction test on a chart with known anisotropy
- [ ] torch/jax parity test
- [ ] `benchmarks/equivariant_scan.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
