# 01-06 OMBU wavelet frames

## 1. Thesis and status

`sigma'` is an admissible mother wavelet, the bias is its translation parameter
and tempering is its dilation parameter, so a bias-scan bank over `(offset,
scale)` is a **wavelet frame with an exact derivative tower on every atom**.

- **Status**: concept
- **Depends on**: 01-02, 01-05
- **Blocks**: 01-07, 02-01, 02-07, 02-08, 02-10, 03-05, 05-01

## 2. Where it lands

A submodule of `omnibias-core` for the frame algebra
(`omnibias/core/frames.py`) plus a thin bank constructor in the backends
reusing `BiasScan` from spec 01-02. No new package: this is a way of *reading*
the scan, not a new dependency tier.

## 3. Prior art in omnibias

- `omnibias.core.spec` — the `tempered` combinator, giving
  `sigma_alpha(u) = sigma(alpha u)` with an **exact** tower
  `sigma_alpha^(n)(u) = alpha^n sigma^(n)(alpha u)`. This is the dilation
  operator, already present and already exact.
- `omnibias.core.transforms` — closed-form Fourier transforms for the `gaussian`
  and `sech` families, which is what an admissibility computation needs.
- `omnibias.core.probability` — `is_cdf_activation`, `cdf_normalization`.
- `packages/omnibias-torch/src/omnibias/torch/architectures/cmbnet.py` — the
  scale layer in `CmbNet` is a fixed multi-scale bank; it is a special case.

**Confirmed gap.** No frame vocabulary: no admissibility constant, no frame
bound estimation, no reconstruction operator, no statement of what redundancy
a bank has.

## 4. Mathematics

### Atoms

With `psi = sigma'` (or any collapsed pack output), define

```
psi_{a,b}(u) = (1 / sqrt(a)) psi( (u - b) / a ),      a > 0, b in R
```

The bank of spec 01-02 with offsets `tau_j` and tempered scales `alpha_m`
realizes exactly `psi_{1/alpha_m, -tau_j}` up to normalization, so the bank *is*
a discrete wavelet system; nothing new is computed, only named.

### Admissibility

A mother wavelet is admissible when

```
C_psi = integral_0^inf |hat_psi(xi)|^2 / xi  d xi  <  inf
```

which requires `hat_psi(0) = 0`, that is `integral psi = 0`. For
`psi = sigma'` with `sigma` a CDF, `integral sigma' = 1 != 0`, so **`sigma'` is
not admissible as-is**. Two clean fixes, both already available:

1. **Use `sigma''` or higher.** For `n >= 2`, `integral sigma^(n) = 0` because
   `sigma^(n-1)` vanishes at both infinities. So `psi_n = sigma^(n)` is
   admissible for every `n >= 2`, and by spec 01-07 its transform is
   `(i xi)^n hat_sigma(xi)`, which vanishes to order `n` at the origin: `psi_n`
   has `n` vanishing moments.
2. **Use a difference of scales.** `phi_eps - phi_{2 eps}` has zero integral by
   construction; this is the multi-pack design of spec 01-05 read as a wavelet.

Route 1 is the natural one here: **derivative order equals number of vanishing
moments.** That single sentence connects the tower to approximation theory: a
tower of order `n` annihilates polynomials of degree `< n`, which is why
high-order packs are blind to smooth trends and sensitive to local structure.

For the `sech` family the admissibility constant is computable in closed form
from the known transform, which is why the frame bounds below are not merely
estimated.

### Frame bounds

A discrete system `{ psi_{a_m, b_{j}} }` is a frame with bounds `A, B` when

```
A ||f||^2  <=  sum_{m,j} |<f, psi_{a_m, b_j}>|^2  <=  B ||f||^2
```

For a dyadic scale grid `a_m = 2^{-m}` and uniform offsets with spacing `b_0`,
the standard sufficient condition is a Littlewood-Paley estimate on

```
Sum(xi) = sum_m |hat_psi(2^{-m} xi)|^2
```

being bounded above and below away from zero, plus `b_0` small enough. With a
closed-form `hat_psi` this is a computable check rather than an assumption, and
the resulting `A, B` can be enclosed with interval arithmetic.

The redundancy `B / A` tells you how much the bank overcounts; a well-designed
bank should report it rather than leave it implicit.

### What the tower adds over a classical wavelet

Every atom carries its own exact derivative tower:

```
d^k/du^k psi_{a,b}(u) = a^{-k - 1/2} sigma^(n + k)( (u - b) / a )
```

so differentiating a wavelet expansion term-by-term is exact and free. Classical
wavelet bases (Daubechies and friends) have limited smoothness and no closed
derivative formula; the price paid here is the loss of exact compact support
(spec 01-05's honesty note applies) and of orthogonality.

The trade is explicit: **omnibias atoms are analytic, exactly differentiable to
all orders, and redundant; classical wavelets are compactly supported,
orthogonal, and only finitely smooth.** Neither dominates; state which property
a use case needs.

## 5. Worked example

Base `sigma = tanh`, so `psi_2 = sigma'' = -2 t (1 - t^2)` with `t = tanh(u)`, an
odd function with `integral psi_2 = 0` (admissible) and vanishing moments up to
order 2 by parity plus construction.

Vanishing moments check:

```
integral psi_2(u) du     = [sigma']_{-inf}^{inf}       = 0     (moment 0)
integral u psi_2(u) du   = -integral sigma'(u) du      = -1    (moment 1, nonzero)
```

So `psi_2` has exactly one vanishing moment in the classical sense; in terms of
polynomial annihilation, `<psi_2, 1> = 0` but `<psi_2, u> = -1`. The number of
vanishing moments of `sigma^(n)` is `n - 1` for the raw derivative, and reaches
`n` when the atom is normalized against its own first nonzero moment.

Dilation exactness, at `u = 0.3`, `alpha = 2`:

```
sigma_alpha''(0.3) = alpha^2 sigma''(0.6) = 4 * ( -2 tanh(0.6) (1 - tanh(0.6)^2) )
tanh(0.6) = 0.5370496,   1 - t^2 = 0.7115777
sigma''(0.6) = -2 * 0.5370496 * 0.7115777 = -0.7642976
sigma_alpha''(0.3) = 4 * (-0.7642976) = -3.0571904
```

Direct evaluation of `d^2/du^2 tanh(2u)` at `u = 0.3` gives the same value to
machine precision. The `alpha^n` scaling is exact, which is what makes the scale
axis of the bank free of any approximation.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/core/frames.py
@dataclass(frozen=True)
class FrameSpec:
    base: str
    order: int                 # n >= 2 for admissibility
    scales: tuple[float, ...]
    offset_spacing: float

def admissibility_constant(base: str, order: int) -> Interval | None:
    """Closed form where the transform is known; None when it is not."""
def vanishing_moments(base: str, order: int) -> int: ...
def littlewood_paley_bounds(spec: FrameSpec, *, grid: int = 4096) -> tuple[Interval, Interval]:
    """Sound (A, B) frame bounds via an enclosed sup/inf of the LP sum."""
def redundancy(spec: FrameSpec) -> Interval: ...
```

Backends: nothing new. A `FrameSpec` compiles to a `BankSpec` plus a template
order for `BiasScan`; that path already exists in spec 01-02.

## 7. Practical use cases

1. **Multiscale feature banks with exact derivatives** for fields sampled off
   any grid: point clouds, meshless PDE collocation, scattered sensors.
2. **Denoising with a stated approximation order.** `n` vanishing moments means
   polynomial trends of degree `< n` pass through untouched, so the atom
   responds to structure rather than to drift.
3. **Sparse coding of interfaces.** A jump in the `k`-th derivative lights up
   atoms of order around `k` at the jump location; the bank is a natural
   interface descriptor for spec 05-01.
4. **Principled bank design.** Instead of guessing scales, choose them so the
   Littlewood-Paley sum is flat, and report the redundancy.
5. **Preconditioning spectral bias.** Spec 01-07 uses the same transforms to
   choose which band each layer sees.

## 8. Acceptance gates

- **G1 admissibility.** `admissibility_constant` matches a high-precision
  numerical integral to `<= 1e-10` relative for every base with a closed-form
  transform, and returns `None` (not a guess) otherwise.
- **G2 frame bounds are sound.** For random test signals, the measured frame
  ratio lies inside `[A, B]` on a dense grid and a random sample, with zero
  violations.
- **G3 dilation exactness.** `sigma_alpha^(n)(u) = alpha^n sigma^(n)(alpha u)`
  holds to `<= 4 ulp` across the supported order range.
- **G4 task skill.** On a denoising task with polynomial trend plus localized
  structure, the order-`n` bank beats a matched-cost Gaussian-derivative bank in
  mean squared error, with skill `> 0` against the identity, over five seeds.

## 9. Benchmark plan

- `benchmarks/ombu_frames.py`: admissibility table, Littlewood-Paley flatness
  versus scale grid, denoising sweep.
- Smoke JSON committed; full sweep under `$OMNIBIAS_SCRATCH/frames/`.

## 10. Honesty and scope

- The founding bias collapse produces the atoms (`delta -> 0`, `K` biases
  coalescing into `sigma^(n)`). No temperature collapse appears here.
- **`sigma'` is not admissible.** Any text that calls the first derivative a
  wavelet without the zero-mean fix is wrong; the admissible atoms start at
  `n >= 2` or at a zero-mean scale difference.
- These systems are **frames, not orthonormal bases**, and the atoms are **not
  compactly supported**. Both facts must appear wherever the word "wavelet" is
  used.
- Frame bounds obtained by enclosing a Littlewood-Paley sum on a finite grid are
  sound only if the enclosure accounts for between-grid variation; use an
  interval sup over subintervals, not a pointwise maximum.

## 11. Open questions and risks

- **No fast transform.** Classical wavelets have `O(N)` algorithms; a redundant
  analytic frame does not. Cost must be measured against a Gaussian-derivative
  bank, the honest competitor.
- **Reconstruction.** The dual frame is not available in closed form; iterative
  reconstruction may be needed, which weakens the "everything is exact" story.
- **Falsifier.** If the exact-derivative property never gets used downstream,
  the frame reading is a rebranding of an existing bank and should stay a
  documented concept.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/src/omnibias/core/frames.py`
- [ ] `FrameSpec -> BankSpec` compiler in both backends
- [ ] Admissibility tests against high-precision integrals
- [ ] Frame-bound soundness test (dense grid plus random sample)
- [ ] Dilation exactness test across orders
- [ ] `benchmarks/ombu_frames.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
