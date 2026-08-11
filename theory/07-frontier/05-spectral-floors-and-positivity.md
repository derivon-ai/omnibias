# 07-05 Spectral floors and positivity

## 1. Thesis and status

Certified eigenvalue *lower* bounds and certified positivity are the two
workhorses behind almost every rigorous claim in mathematical physics, and both
are variational — so both are limited by the trial space, which is exactly what
multi-pack and arrangement bases are good at supplying.

- **Status**: designed
- **Depends on**: 01-01, 01-03, 01-11, 07-01
- **Blocks**: 07-04, 07-06

## 2. Where it lands

`omnibias.core.verified.eig_operator` for the spectral side and `omnibias.sos`
for positivity. Both exist and both are the right home; the additions are trial
spaces and basis generators, not new engines.

## 3. Prior art in omnibias

The engines are shipped and are the strong part of the library.

- `omnibias.core.verified.eig_operator` — `lehmann_maehly_lower_bounds` with
  `LehmannCertificate`, `certified_spectral_gap` with `SpectralGapCertificate`,
  `temple_lower_bound` and `temple_lower_bound_vector`,
  `generalized_eigenvalue_enclosure`, `count_eigenvalues_below`,
  `interval_ldlt_inertia` and `interval_ldlt_pivots`, `is_positive_definite`,
  `ritz_upper_bound`, `operator_comparison_bounds`, `EigenvalueLowerBound`,
  `Inertia`.
- `omnibias.sos` — `problem`, `solve`, `certify`, `positivstellensatz`,
  `monomials`, `rounding`, `auxiliary`, `honesty`, `formal`, `proofmachine`. The
  interval LDL^T PSD certificate can earn `theorem_prover_verified`.
- `omnibias.core.verified.{Interval,affine,TaylorModel,TaylorModelMV}` — the
  substrate.
- `omnibias.convex` — log-barrier interior point with closed-form Hessian, KKT
  implicit gradients.

**Confirmed gap.** The trial spaces are supplied by the caller and there is no
principled generator. There is no adaptive trial-space refinement, and the
choice of SOS monomial basis is not informed by the problem's geometry.

## 4. Mathematics

### Why lower bounds are hard and upper bounds are easy

For a symmetric operator `A`, any trial vector `v` gives a Rayleigh quotient
`R(v) = v^T A v / v^T v >= lambda_1`, so **upper** bounds on the lowest
eigenvalue are free. A **lower** bound requires knowing that no eigenvalue hides
below the trial space's reach, which is genuinely harder and is what
Lehmann-Maehly-Goerisch provides, given a shift `rho` that must itself be a
rigorous separator.

The bound's quality depends on the trial space in a specific way: the
Lehmann bound is sharp when the trial space contains the true eigenvectors and
degrades with the angle between them. So a basis that resolves the eigenvector's
*local structure* is worth far more than a larger generic one.

### Why multi-pack bases help

Low-lying eigenfunctions of physically interesting operators are typically
smooth in most of the domain and structured in small regions — a boundary layer,
a potential well, a corner singularity. A multi-pack (spec 01-01) places
different derivative orders at different locations, which is exactly a basis
that is cheap where the function is smooth and rich where it is not.

Concretely: a Schrödinger operator with a deep narrow well needs many Fourier
modes or many uniform elements to resolve the well; a multi-pack with high order
at the well and low order elsewhere resolves it with a handful of functions.
Since the Lehmann bound's cost is cubic in the trial dimension, a `4x` reduction
in dimension is a `64x` reduction in certification cost, before any tightness
improvement.

### Why arrangements help for SOS

A Positivstellensatz certificate writes `p = sum_j sigma_j g_j` with each
`sigma_j` a sum of squares. The monomial basis for the `sigma_j` determines both
feasibility and cost, and the standard choice — all monomials up to a degree —
scales combinatorially and is agnostic to where `p` is nearly zero.

If `p`'s near-zero set is understood geometrically — a cell of a hyperplane
arrangement, say — the basis can be adapted to it. The connection is the same
one as in the spectral case: **positivity is hardest where the function is
smallest**, so basis richness should be concentrated there.

### The certified chain

For both, the chain is:

```
trial space  ->  interval matrix assembly  ->  interval LDL^T  ->  rigorous bound
             ->  finite rational obligation  ->  Lean kernel
```

Every step after the first is already implemented. The obligation extracted at
the end is a **finite rational** statement — "this enclosed quantity is
strictly positive" — which is exactly the Mathlib-free kernel's scope.

## 5. Worked example

**A one-dimensional well, and the dimension it takes to certify.**

Take `A = -d^2/dx^2 + V(x)` on `[-1, 1]` with Dirichlet conditions and

```
V(x) = -50 exp( -100 x^2 )
```

a well of depth `50` and width about `0.2`. The ground state is concentrated in
the well.

*Uniform basis.* Sine modes `sin(k pi (x+1) / 2)`. Resolving a feature of width
`0.2` on a domain of length `2` needs wavenumbers up to about `2 / 0.2 = 10`
oscillations, so roughly `k <= 30` for a few digits, and the Lehmann shift must
separate the ground state from the rest, which needs enough modes that the
computed second eigenvalue is trustworthy. Call it `n = 40`.

Cost of the interval Lehmann computation: the LDL^T is `O(n^3)` in interval
arithmetic, so `40^3 = 64 000` interval operations, each several times the cost
of a float operation.

*Multi-pack basis.* Place one pack of order `0 ... 6` at the well centre
(`7` functions, tempered scale matched to the well width `0.2`, so
`alpha = 10`), plus `8` low-order functions spread over the rest of the domain.
Total `n = 15`.

Cost: `15^3 = 3 375`, a **19x reduction**, and the basis is better aligned with
the eigenfunction because the tower at scale `alpha = 10` reproduces exactly the
Gaussian-like local structure the well imposes.

*The check that matters.* Alignment is an assumption until measured. The
benchmark must report, for both bases, the angle between the trial space and the
numerically computed ground state:

```
sin(theta) = || (I - P_V) v_1 ||
```

If the multi-pack basis at `n = 15` has a smaller `sin(theta)` than the sine
basis at `n = 40`, the claim holds; if not, the dimension reduction is buying
nothing and the bound will be looser despite being cheaper. **Reporting
`sin(theta)` alongside the bound is what separates this from wishful basis
selection**, and it costs one projection.

*The honest expectation.* For this problem — a smooth well matched by a tempered
Gaussian-family basis — the multi-pack should win clearly, because the basis was
chosen knowing the answer's shape. For a problem whose eigenfunction has
structure the pack family cannot express (an oscillatory ground state, a corner
singularity with a non-integer exponent), the uniform basis will win, and the
benchmark suite must include such a case rather than only favourable ones.

## 6. Proposed API

```python
# omnibias/core/verified/trial_spaces.py    -- new, pure Python
def multipack_trial_space(
    locations: Sequence[float], orders: Sequence[int], scales: Sequence[float],
    *, domain: tuple[float, float],
) -> TrialSpace:
    """Multi-pack basis for a variational bound. Pure core: no backend imports."""

def trial_space_alignment(trial: TrialSpace, reference_vector) -> float:
    """sin(theta) between the trial space and a reference eigenvector.
    Required in every reported bound."""

def adaptive_trial_refinement(
    operator, trial: TrialSpace, *, target_width: float, max_dim: int,
) -> TrialSpace:
    """Residual-driven refinement, reusing the birth/growth moves of spec 03-13."""

# omnibias/sos/monomials.py                 -- extension
def arrangement_adapted_basis(problem, arrangement, *, degree: int): ...
```

Pure-Python, in `omnibias.core.verified`, consistent with the pure-core rule.

## 7. Practical use cases

1. **Cheaper certified spectral gaps** for the gauge transfer matrices of spec
   07-04, which consume this directly.
2. **Certified ground-state energies** for model quantum systems, with lower
   bounds rather than variational upper bounds alone — the pairing gives a
   two-sided enclosure of a physical quantity.
3. **Certified stability floors** for validated dynamics (spec 07-06), where a
   spectral gap bounds a contraction rate.
4. **Tighter SOS certificates** at lower degree, which is the binding constraint
   on SOS applicability.
5. **Two-sided enclosures for eigenvalue problems in PDE analysis**, where the
   upper bound is easy and the lower bound is the whole difficulty.

## 8. Acceptance gates

Baselines: uniform spectral (sine or Chebyshev) trial spaces at matched
dimension, and the standard total-degree monomial basis for SOS.

- **G1 alignment reported.** Every certified bound reports `sin(theta)` against
  a numerically computed reference eigenvector. No bound is accepted without it.
- **G2 dimension reduction.** On a suite of at least `10` problems with
  localized eigenfunctions, the multi-pack basis achieves a bound within `5%` of
  the uniform basis's at **one quarter the dimension or less**, over the whole
  suite.
- **G3 adversarial case included.** The suite contains at least `3` problems
  whose eigenfunctions the pack family cannot express well (oscillatory ground
  state, corner singularity, discontinuous coefficient), and the results are
  reported without exclusion. Losing on these is expected and acceptable;
  omitting them is not.
- **G4 soundness.** The certified lower bound never exceeds the true eigenvalue
  across `1000` synthetic problems with known spectra. A single violation is a
  bug.
- **G5 SOS degree reduction.** Arrangement-adapted bases certify positivity at a
  degree at least `2` lower than the total-degree basis on at least half of a
  named problem set, with the failures reported.
- **G6 kernel obligation.** The extracted positivity obligation passes
  `lake build` and `theorem_prover_verified` is set by the kernel, never
  asserted. Asserted by test.

## 9. Benchmark plan

- `benchmarks/spectral_trial_spaces.py`: the `10`-problem suite plus the `3`
  adversarial cases, alignment and dimension-versus-tightness curves, soundness
  over synthetic spectra.
- `benchmarks/sos_adapted_basis.py`: degree reduction on a named problem set.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/spectral/`.

## 10. Honesty and scope

- **Fixed-operator statements only.** Every bound is about one operator, one
  discretization, one domain. This is `eig_operator`'s own stated scope and is
  not widened.
- A certified **lower** bound plus a variational **upper** bound is a genuine
  two-sided enclosure of the discretized operator's eigenvalue. It says nothing
  about the continuous operator unless a separate discretization-error bound is
  supplied, which this spec does not provide.
- Trial-space choice affects **tightness, never soundness**. A badly chosen
  basis gives a loose but still valid bound. Any bug that lets it give an
  invalid bound is a soundness failure and G4 exists to catch it.
- The founding `delta -> 0` bias collapse supplies the multi-pack basis
  functions. No temperature collapse appears.
- Certificate tier: **sound enclosure**, escalating to
  `theorem_prover_verified` only on a genuine kernel pass. The SOS path can
  additionally reach `mathlib_verified` through `omnibias.formal`, which is a
  distinct tier and is never conflated with the kernel one.

## 11. Open questions and risks

- **Basis selection is problem knowledge in disguise.** Placing packs where the
  eigenfunction is localized requires knowing roughly where that is. The
  adaptive refinement path is the principled answer, and if it does not work the
  method reduces to "a good basis gives a good bound", which is true and not a
  contribution.
- **Conditioning.** Multi-pack bases at high order and matched scale become
  nearly linearly dependent; the interval LDL^T will detect this by failing,
  which is safe but limits usable dimension. Condition numbers must be reported.
- **The Lehmann shift.** The bound needs a rigorous separator `rho`, and
  obtaining one can be harder than the bound itself. A better trial space does
  not help with this and the spec should not imply otherwise.
- **SOS basis adaptation may not reduce degree.** Positivity certificates can
  require a degree set by the algebraic structure rather than the geometry, in
  which case no basis choice helps. G5 allows for this by requiring only half
  the problem set.
- **Falsifier.** If G2 fails on the localized-eigenfunction suite — the case
  constructed to favour multi-packs — then the basis contributes nothing and the
  spec reduces to the alignment diagnostic, which is a small but real
  contribution and should be shipped as exactly that.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/src/omnibias/core/verified/trial_spaces.py`,
      pure Python
- [ ] `trial_space_alignment` reported in every bound
- [ ] Adaptive refinement reusing spec 03-13's birth and growth moves
- [ ] `10`-problem localized suite plus `3` adversarial problems, none excluded
- [ ] `1000`-problem soundness test against known spectra
- [ ] Condition numbers reported; interval LDL^T failure handled as a safe
      outcome, not an error to suppress
- [ ] `arrangement_adapted_basis` in `packages/omnibias-sos/src/omnibias/sos/monomials.py`
- [ ] Kernel obligation wired with a test that the flag cannot be forged
- [ ] `benchmarks/spectral_trial_spaces.py` and `benchmarks/sos_adapted_basis.py`
      plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parents: the Yang-Mills mass gap (through spec 07-04, which consumes these
bounds) and, more broadly, spectral-gap questions in mathematical physics
including quantum many-body gap conjectures.**

Every bound produced here is about a **single fixed finite-dimensional
operator**. The parents are statements about families — a continuum limit, a
thermodynamic limit, an infinite lattice — and a bound on one member of a family
constrains nothing about the family's limit without a uniformity argument that
this machinery does not supply and does not attempt.

The general obstruction is worth naming precisely: the parents are typically
**undecidable-in-general or open-in-general** statements about limits, while
`eig_operator`'s scope, in its own words, is fixed-operator statements. That gap
is structural, not a matter of tighter constants or larger trial spaces.

This spec does not claim, imply, or provide evidence for any spectral-gap
conjecture about a limit of operators.
