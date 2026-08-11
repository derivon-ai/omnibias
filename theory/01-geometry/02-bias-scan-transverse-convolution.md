# 01-02 Bias scan: transverse convolution and the position knob

## 1. Thesis and status

Sliding a hyperplane along its own normal *is* what the bias already does, so
the convolutional idea is not "slide the plane" but **share one pack template
and evaluate it at many bias offsets**, producing a translation-equivariant
response along `w` at the cost of one activation call per offset.

- **Status**: designed
- **Depends on**: 01-01
- **Blocks**: 01-06, 01-10, 02-01, 02-07, 02-08, 02-11, 03-04, 03-05, 03-08, 05-01, 05-02

## 2. Where it lands

Submodule beside the existing blocks:
`packages/omnibias-torch/src/omnibias/torch/scan.py` plus the jax twin, with the
offset-grid algebra in `packages/omnibias-core/src/omnibias/core/scan.py`.

## 3. Prior art in omnibias

- `packages/omnibias-torch/src/omnibias/torch/blocks/conv.py` — `cmbConv1d`,
  `cmbConv2d`: a standard `nn.Conv*d` followed by a per-channel `OperatorBlock`.
  This is *spatial grid* convolution with an operator-typed nonlinearity.
- `packages/omnibias-torch/src/omnibias/torch/architectures/cmbnet.py` —
  `CmbNet`, the reference operator-typed CNN (edge / blob / scale layers).
- `packages/omnibias-torch/src/omnibias/torch/unit.py` — the OMBU, which is
  already a 1-D convolution of `sigma` with a tap vector along the transverse
  coordinate.

**Confirmed gap.** There is no object that shares one template across a *bank of
bias offsets* and returns the whole response vector. `cmbConv*` convolves over a
pixel grid; the scan convolves over the transverse coordinate itself and needs
no grid at all.

## 4. Mathematics

### Why a lone sliding plane is not convolution

With `H(b) = { x : w . x + b = 0 }`:

- changing `b` slides `H` along `w`: that is the bias, already present;
- translating `x` tangentially leaves the set `H` unchanged, so it produces no
  new feature;
- rotating `w` is a different orientation, not a slide.

Convolution is one pattern, weight-shared, applied at many positions. Along the
transverse coordinate `z = w . x` an OMBU already is a discrete convolution

```
f(z) = sum_k s_k sigma(z + b_k) = (s * sigma)(z)
```

with taps `s` at offsets `b`. Collapse signs make it a derivative kernel; free
signs make it a learned filter.

### The scan operator

Let `T` be a template: a `MultiPackSpec` (spec 01-01) or a single role, with all
means measured relative to a reference. Define the **bias bank** offsets
`tau_1 .. tau_M` and

```
Scan[T](z) = ( F_T(z + tau_1), ..., F_T(z + tau_M) )
```

where `F_T` is the template's collapsed output. Parameters `(w, T)` are shared
across `j`; only `tau_j` varies. Three readouts:

```
response       R_j(x) = F_T(w . x + tau_j)
pooled         P(x)   = sum_j a_j R_j(x)                  (learned taps: a filter)
localized      tau*(x) = sum_j tau_j softmax(gamma R)_j   (soft argmax)
```

### Equivariance

For a shift `x -> x + t w / |w|^2` the pre-activation shifts by `t`, so

```
R_j(x + t w / |w|^2) = R_{j'}(x)  whenever tau_{j'} = tau_j + t
```

exactly when `t` is a multiple of a uniform bank spacing. The scan is therefore
**equivariant to translation along `w`** on the bank lattice, and approximately
equivariant off-lattice with error controlled by the bank spacing and the
template's bandwidth (spec 01-07 gives the bandwidth).

Note what is *not* claimed: nothing here is equivariant to tangential
translation (which acts trivially) or to rotation (spec 02-08 handles
orientation).

### Scale as a second bank axis

Using the `tempered` combinator, `sigma_alpha(u) = sigma(alpha u)` has an exact
tower, so a two-dimensional bank `(tau_j, alpha_m)` sweeps position and width.
With a `band` or `integral` template this is a soft sliding window of tunable
width: the closest structural analogue to a CNN receptive field in this
geometry.

### Cost

`M` distinct offsets cost `M` activation evaluations per channel, and every
order in the template at a given offset is free once that activation value is
known. A bank is therefore cheaper than `M` independent units by exactly the
factor of shared template parameters.

## 5. Worked example

Template: a single `K = 2` collapse of `sigma = tanh`, so `F_T(u) = 1 - tanh^2(u)`,
a bump of unit height at `u = 0`.

Bank: `tau = (-1.0, -0.5, 0.0, 0.5, 1.0)`, input `z = 0.3`.

```
u_j      = z + tau_j = (-0.7, -0.2, 0.3, 0.8, 1.3)
R_j      = 1 - tanh(u_j)^2
         = (0.6469, 0.9610, 0.9151, 0.5219, 0.1849)
soft argmax with gamma = 8:
  weights  ~ (0.0356, 0.5238, 0.4162, 0.0193, 0.0051)
  tau*     ~ -0.283
```

The response peaks at `tau = -0.5` and `tau = 0.0`, bracketing the true
interface at `tau = -z = -0.3`; the soft argmax returns `-0.283`, within one
tenth of a bank spacing. Refining the bank to spacing `0.25` moves the estimate
to `-0.297`. That is the matched-filter localization property the scan exists
for.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/core/scan.py
@dataclass(frozen=True)
class BankSpec:
    offsets: tuple[float, ...]
    scales: tuple[float, ...] = (1.0,)
    def uniform(lo: float, hi: float, n: int) -> "BankSpec": ...
    @property
    def spacing(self) -> float | None: ...   # None when non-uniform
```

```python
# omnibias/torch/scan.py  and  omnibias/jax/scan.py
class BiasScan(nn.Module):
    def __init__(
        self,
        num_channels: int,
        bank: BankSpec,
        *,
        template: MultiPackSpec | OpName = "grad",
        base: str | ActivationSpec = "tanh",
        learnable_offsets: bool = True,
        learnable_scales: bool = False,
        readout: Literal["response", "pooled", "argmax"] = "response",
        gamma: float = 8.0,          # soft-argmax sharpness, not a collapse
        dtype: torch.dtype | None = None,
    ) -> None: ...
    def forward(self, z: Tensor) -> Tensor: ...   # (..., C) -> (..., C, M) or (..., C)

def scan_response(z, offsets, scales, spec, base) -> Tensor: ...
def soft_argmax_offset(response, offsets, *, gamma) -> Tensor: ...
```

Rules: offsets and scales are arrays (traceable); the template's orders are
static ints; `dtype=None` resolves to the framework default; torch and jax must
be bit-identical.

## 7. Practical use cases

1. **Interface localization in a field.** Sweep the bank along a normal and read
   `tau*`: where is the layer boundary, the shock, the wall? Beats training a
   separate regressor because the estimate is differentiable and shares the
   template with the feature extractor.
2. **Matched filtering for a known signature.** If the physics says the
   interface carries a `sigma''` signature, scan that template and take the
   peak. This is classical matched filtering with an exactly differentiable
   kernel.
3. **Scale-space features without a pixel grid.** Point clouds and implicit
   fields have no array to convolve; the `(tau, alpha)` bank still gives a
   multiscale response.
4. **Cheap 1-D probes along rays.** Sampling a learned field along a normal (for
   example along an SDF gradient) and scanning gives a profile descriptor at a
   fraction of the cost of a volumetric CNN.
5. **Front tracking in time.** Scanning in the time coordinate turns the bank
   into a moving-front detector for evolution problems.

## 8. Acceptance gates

Baseline: (a) a single fixed-bias `OperatorBlock` of the same template, and
(b) `cmbConv1d` on a gridded version of the same signal at matched parameters.

- **G1 equivariance.** For on-lattice shifts, the response vector is a pure
  circular shift to `<= 4 ulp`. Off-lattice, the equivariance error decays at
  the predicted rate as the bank spacing halves.
- **G2 localization.** On synthetic interfaces with known position and additive
  noise at 1 percent of signal amplitude, mean absolute error of `tau*` is
  `<= 0.1` bank spacings, with skill `> 0` against predicting the domain
  midpoint, over five seeds.
- **G3 parity.** torch and jax bit-identical.
- **G4 no-grid win.** On a point-cloud interface task with no natural grid, the
  scan beats a voxelized `cmbConv` pipeline in mean absolute position error at
  equal or lower wall time.

## 9. Benchmark plan

- `benchmarks/bias_scan.py` with smoke and `--full` tiers.
- Smoke writes `docs/benchmarks/bias_scan_smoke.json`; full writes under
  `$OMNIBIAS_SCRATCH/scan/`.
- Metrics: equivariance error versus spacing, localization error versus noise,
  wall time versus `M`.
- Standard `gates` block from `benchmarks/_gates.py`.

## 10. Honesty and scope

- The template's internal limit is the founding bias collapse
  (`delta -> 0`, yields `sigma^(n)`). The `gamma` in the soft argmax is a
  sharpness parameter on a softmax readout; if it is ever driven to infinity
  that is **temperature collapse** (a hard argmax), a different limit from bias
  collapse, and must be labelled as such wherever it appears.
- Equivariance is along `w` only, and exact only on the bank lattice. Do not
  describe the scan as a translation-equivariant operator on the input space.
- The scan is not a replacement for grid convolution when a grid exists and is
  cheap; the claim is about grid-free settings and about sharing a typed
  template.

## 11. Open questions and risks

- **Bank size versus cost.** Cost is linear in `M`. Beyond a few dozen offsets a
  hierarchical scheme (spec 02-07) is required, and the flat scan should
  document the crossover measured, not assumed.
- **Learned offsets can collapse onto each other**, wasting bank capacity. Track
  minimum separation during training; consider a spacing prior.
- **Soft argmax is biased** when two interfaces are within one bank spacing. The
  benchmark must include a two-interface case so the failure mode is visible
  rather than hidden.
- **Falsifier.** If localization error never beats a simple thresholded gradient
  detector on realistic noise, the primitive is not earning its cost.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/src/omnibias/core/scan.py`
- [ ] `packages/omnibias-torch/src/omnibias/torch/scan.py`
- [ ] `packages/omnibias-jax/src/omnibias/jax/scan.py`
- [ ] Equivariance regression test (on-lattice exactness, off-lattice decay)
- [ ] Soft-argmax gradient test against finite differences
- [ ] torch/jax parity test
- [ ] `benchmarks/bias_scan.py` plus committed smoke JSON
- [ ] Docs page and mkdocs nav entry
- [ ] Regenerate `__all__` in both backends
- [ ] Index row in `theory/README.md`
