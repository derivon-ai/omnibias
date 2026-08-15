# 01-07 Order as frequency: spectral design of the tower

## 1. Thesis and status

Differentiation multiplies the Fourier transform by `(i xi)^n`, so **pack order
is a frequency-band selector**: choosing `n` and the tempering scale `alpha`
places a channel's sensitivity in a known band, which converts spectral-bias
mitigation from a heuristic into a design calculation.

- **Status**: gated (G1–G2 earned; G3 not in CI `all_passed`; 01-06 frames stay concept)
- **Depends on**: 01-01, 01-06
- **Blocks**: 02-01, 02-05, 02-07, 03-07, 07-03

## 2. Where it lands

`packages/omnibias-core/src/omnibias/core/spectral_design.py` (pure Python
band algebra and initializers) plus initializer helpers in each backend that
consume it. It is a design calculator, not a runtime layer.

## 3. Prior art in omnibias

- `omnibias.core.transforms` — closed-form Fourier transforms for `gaussian`
  and `sech` families.
- `omnibias.core.spec` — the `tempered` combinator with the exact scaling law
  `sigma_alpha^(n)(u) = alpha^n sigma^(n)(alpha u)`.
- `packages/omnibias-pinn/.../MscaleMLP` — multiscale MLP with a fixed set of
  input scalings, the standard MscaleDNN remedy for spectral bias.
- `FourierFeatureMLP` — random Fourier features with a chosen bandwidth.
- `FBPINNField` — finite-basis domain decomposition.
- `benchmarks/spectral_bias_fbpinn.py` — the existing four-gap benchmark arm,
  which records per-arm `wall_seconds` and `lstsq_matched`.

**Confirmed gap.** The multiscale remedies choose scales heuristically (a
geometric ladder). Nothing computes *which band a given `(n, alpha)` channel
actually sees*, and nothing initializes a network from a target spectrum.

## 4. Mathematics

### The band of a channel

Let `hat_sigma(xi)` be the transform of the base. Then

```
hat{sigma^(n)}(xi) = (i xi)^n hat_sigma(xi)
```

and with tempering,

```
hat{sigma_alpha^(n)}(xi) = (i xi)^n (1 / alpha) hat_sigma(xi / alpha) * alpha^n
                         = i^n xi^n alpha^(n-1) hat_sigma(xi / alpha)
```

The **response magnitude** is `R_{n,alpha}(xi) = |xi|^n alpha^(n-1)
|hat_sigma(xi / alpha)|`. This is a band-pass profile: `|xi|^n` suppresses low
frequencies, and the decay of `hat_sigma` suppresses high ones.

### Peak frequency

For the `sech`-family base, `hat_sigma(xi)` decays like `exp(-c |xi|)` for a
base-dependent `c > 0`. Then

```
log R = n log|xi| - c |xi| / alpha + const
d/d|xi|: n / |xi| - c / alpha = 0   =>   xi_peak = n alpha / c
```

So the peak frequency is **linear in the order and linear in the temper scale**:

```
xi_peak(n, alpha) = n alpha / c
```

For a gaussian base with `hat_sigma(xi) ~ exp(-xi^2 / 2)`, the same computation
gives `xi_peak = alpha sqrt(n)`. The two families therefore trade order against
scale differently, which is a real design choice and should be exposed.

### Bandwidth

Expanding `log R` to second order about the peak gives, for the exponential-decay
family,

```
d^2 log R / d xi^2 |_peak = -n / xi_peak^2 = -c^2 / (n alpha^2)
```

so the effective half-width is `sqrt(n) alpha / c`, and the **relative**
bandwidth `Delta xi / xi_peak = 1 / sqrt(n)` narrows as the order grows. High
order means a sharper band, which is the quantitative version of "high-order
packs are selective".

### Design problem

Given a target spectral support `[xi_lo, xi_hi]` and a budget of `C` channels,
choose `{(n_i, alpha_i)}` so that

```
sum_i R_{n_i, alpha_i}(xi)   is bounded above and below on [xi_lo, xi_hi]
```

which is exactly the Littlewood-Paley flatness condition of spec 01-06, now used
as an initializer rather than a diagnostic. Practical recipe:

1. Choose peaks geometrically spaced across `[xi_lo, xi_hi]`, spacing set by the
   relative bandwidth `1 / sqrt(n)` so neighbouring bands overlap at a target
   ratio (a half-power overlap is a reasonable default).
2. Solve `xi_peak(n, alpha) = target` for `alpha` at fixed `n`, in closed form
   from the formulas above.
3. Report the achieved flatness (max over min of the sum on the target band) as
   a number, so the initialization is auditable.

### Why this matters for spectral bias

The standard analysis says a network's neural tangent kernel has a spectrum that
decays with frequency, so high-frequency components are learned slowly. The
MscaleDNN remedy rescales inputs so that a high frequency looks low to some
subnetwork. The formulation here does the same thing but says *exactly* which
band each channel covers before training starts, and lets a residual-driven rule
(spec 03-13) add a channel in the band where the residual spectrum is largest.

## 5. Worked example

Base: `sech`-family with decay constant `c = pi / 2` (the value for
`sech(pi u / 2)`-normalized transforms; the implementation reads the true
constant from `omnibias.core.transforms`).

Target band: `xi in [1, 32]`, budget `C = 4` channels, fixed order `n = 2`.

Relative bandwidth at `n = 2` is `1 / sqrt(2) = 0.707`, so geometric spacing by a
factor of about `2` gives sensible overlap. Peaks at `xi = 2, 4, 8, 16`:

```
alpha_i = c xi_i / n = (pi/2) xi_i / 2 = 0.7854 xi_i
alpha = (1.571, 3.142, 6.283, 12.566)
```

Flatness check of `sum_i R_{2, alpha_i}(xi)` over `[1, 32]`: with these four
channels the sum varies by a factor of about `1.4` across the band, which is
acceptable. Dropping to two channels (peaks at `2` and `16`) makes the ratio
about `7.9`, a visible spectral hole at `xi ~ 6`, and that hole is exactly where
a PINN would stall.

Order sweep at fixed peak `xi = 8`: `alpha = c * 8 / n`, so `n = 1` needs
`alpha = 12.57`, `n = 4` needs `alpha = 3.14`. The higher-order channel achieves
the same peak with a four-times smaller temper scale and a relative bandwidth of
`0.5` instead of `1.0`: sharper, and less prone to the saturation that large
`alpha` causes in the activation argument.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/core/spectral_design.py
def response_profile(base: str, order: int, alpha: float, xi: Sequence[float]) -> tuple[float, ...]:
    """|xi|^n alpha^(n-1) |hat_sigma(xi / alpha)|, closed form where available."""

def peak_frequency(base: str, order: int, alpha: float) -> float: ...
def relative_bandwidth(base: str, order: int) -> float: ...
def alpha_for_peak(base: str, order: int, xi_peak: float) -> float: ...

@dataclass(frozen=True)
class BandPlan:
    orders: tuple[int, ...]
    scales: tuple[float, ...]
    flatness: float          # max/min of the summed response on the target band

def design_band_plan(
    base: str, *, xi_lo: float, xi_hi: float, channels: int,
    order: int | Sequence[int] = 2, overlap: float = 0.5,
) -> BandPlan: ...
```

Backends consume a `BandPlan` in an initializer:

```python
# omnibias/torch/init.py  (and jax twin)
def init_from_band_plan(module, plan: BandPlan, *, generator=None) -> None: ...
```

Bit-identical across backends given the same plan and seed policy.

## 7. Practical use cases

1. **PINN initialization for known solution spectra.** Wave and Helmholtz
   problems come with a wavenumber; put channels where the physics is instead of
   hoping a geometric ladder covers it.
2. **Diagnosing training stalls.** Compute the residual spectrum, compare with
   the network's band coverage, and see the hole. This turns "the PINN will not
   converge" into a measurement.
3. **Choosing order instead of width.** When saturation limits how large
   `alpha` can be, raising `n` reaches the same band at smaller `alpha`.
4. **Adaptive refinement** (spec 03-13): spawn a channel at the peak of the
   residual spectrum, with `alpha` solved in closed form.
5. **Fair benchmarking.** Reporting flatness alongside accuracy stops an
   architecture from winning purely by accidentally better band coverage.

## 8. Acceptance gates

Baselines: `MscaleMLP` with its default geometric scales, and `FourierFeatureMLP`
with a tuned bandwidth, both at matched parameter count.

- **G1 formula correctness.** `peak_frequency` matches the numerically located
  peak of `response_profile` to `<= 1e-6` relative across the supported bases and
  orders `1 .. 8`.
- **G2 spectral prediction.** For a channel with plan `(n, alpha)`, the measured
  transfer magnitude on random band-limited inputs matches `R_{n,alpha}` to
  `<= 2` percent across the band.
- **G3 task skill.** On the existing spectral-bias arm
  (`benchmarks/spectral_bias_fbpinn.py`), a band-planned initialization reaches
  the arm's absolute error gate in at least `2x` fewer steps than the tuned
  `MscaleMLP` baseline, over five seeds, with `lstsq_matched` recorded as the
  benchmark already requires.
- **G4 hole detection.** For a deliberately holed plan, the diagnostic flags the
  hole before training, and training does in fact stall in that band.

## 9. Benchmark plan

- Extend `benchmarks/spectral_bias_fbpinn.py` with a `band_plan` arm rather than
  creating a competing benchmark, keeping the shared gates in
  `benchmarks/_gates.py` and the existing per-arm `wall_seconds` and
  `lstsq_matched` fields.
- Add `benchmarks/spectral_design.py` for the pure formula validation (cheap,
  CPU, always in smoke).

## 10. Honesty and scope

- The tower comes from the founding bias collapse (`delta -> 0`). No temperature
  collapse appears in this spec.
- The band formulas are **exact for bases with closed-form transforms** and are
  otherwise numerical; `response_profile` must say which regime it is in rather
  than silently switching.
- The peak and bandwidth formulas are asymptotic in the sense of a Laplace
  approximation about the maximum. For small `n` (especially `n = 1`) the
  quadratic bandwidth estimate is crude; report the numerically measured
  half-power width alongside it.
- Faster convergence in a benchmark is an empirical claim tied to that
  benchmark's gate. It is not a theorem about spectral bias.

## 11. Open questions and risks

- **Composition.** These are single-channel band statements. What a *deep*
  composition sees is not the product of the bands, and the spec must not
  pretend otherwise; the honest object for depth is the jet composition, not a
  transfer function.
- **Nonlinearity.** The transfer reading is a linearization; on large-amplitude
  inputs the activation saturates and the band shifts. Measure the amplitude at
  which the prediction degrades.
- **Falsifier.** If band-planned initialization does not beat the tuned
  geometric ladder at matched cost, the calculator is a diagnostic only, and the
  spec should say so.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/src/omnibias/core/spectral_design.py`
- [ ] `init_from_band_plan` in torch and jax with a parity test
- [ ] Peak and bandwidth tests against numerically located extrema
- [ ] Transfer-magnitude test on band-limited inputs
- [ ] New arm in `benchmarks/spectral_bias_fbpinn.py`
- [ ] `benchmarks/spectral_design.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
