# 01-05 Mollifier and distribution calculus

## 1. Thesis and status

A collapsing pack is a **mollifier**: the family `f_K(z; delta)` converges to the
`(K-1)`-st distributional derivative of a Dirac functional, so an OMBU is not
only a smooth function but a legitimate test-function generator, which is what a
weak-form method needs.

- **Status**: gated (G1–G3 earned; G4 deferred to 02-04 VPINN)
- **Depends on**: 01-01
- **Blocks**: 01-06, 01-10, 02-04, 02-06, 02-14, 03-04, 03-06, 07-02

## 2. Where it lands

`packages/omnibias-core/src/omnibias/core/mollifier.py` — pure Python: moment
conditions, admissibility tests, exact antiderivative bookkeeping. The
tensor-side test-function assembly lives with the weak-form architecture
(spec 02-04) in `omnibias-fields`.

## 3. Prior art in omnibias

- `omnibias.core.spec` — `ActivationSpec` metadata, including whether a base is a
  CDF and what its normalization is.
- `omnibias.core.probability` — `cdf_normalization`, `is_cdf_activation`. When
  `sigma` is a CDF, `sigma'` is a probability density, which is exactly the
  mollifier condition `integral phi = 1`.
- `omnibias.core.transforms` — closed-form Fourier and related transforms for
  the `gaussian` and `sech` families.
- `docs/operator-surface.md` — the `integral` role: an antiderivative window
  `S(z + b_hi) - S(z + b_lo)` with `S' = sigma`. omnibias already has closed-form
  antiderivatives, not only derivatives.

**Confirmed gap.** There is no mollifier or test-function vocabulary anywhere:
no moment conditions, no admissibility check, no statement of what a pack
converges to as a distribution. The `integral` role is the raw material; the
calculus around it is missing.

## 4. Mathematics

### The mollifier family

Let `sigma` be a CDF-type base, so `phi = sigma'` is a density: `phi >= 0`,
`integral phi = 1`. Define the scaled family

```
phi_eps(u) = (1 / eps) phi(u / eps)
```

Then `phi_eps -> delta` weakly as `eps -> 0+`: for continuous bounded `psi`,
`integral phi_eps(u) psi(u) du -> psi(0)`.

Now the collapse statement in distribution form. A `K`-pack with central
stencil, spread `delta`, applied against a test function `psi`:

```
integral f_K(u; delta) psi(u) du  ->  (-1)^(K-1) psi^(K-1)(0)   as delta -> 0
```

after normalization, which is precisely the action of `delta^(K-1)`, the
`(K-1)`-st distributional derivative of the Dirac functional. **The founding
bias collapse is a statement about mollifiers converging to derivatives of a
point functional.**

### Moment conditions and accuracy

A mollifier `phi` is of **order `m`** when

```
integral phi = 1,     integral u^j phi(u) du = 0 for j = 1 .. m-1,
                      integral u^m phi(u) du != 0
```

Convolution with an order-`m` mollifier reproduces polynomials of degree `< m`
exactly, and the smoothing error on a `C^m` function is `O(eps^m)`. Symmetric
bases (`gaussian`, `sech`, the derivative of `tanh`) automatically kill all odd
moments, so they are at least order 2.

Higher order comes free from multi-packs: a linear combination
`sum_g c_g phi(u - mu_g)` can be designed to annihilate a chosen set of moments,
and the coefficients solve the same Vandermonde system as spec 01-04. So
**order-`m` mollifiers are a multi-pack design problem**, solved exactly over
the rationals.

### Admissibility for weak forms

A Petrov-Galerkin test function `v` must satisfy:

| Requirement | Why | How the pack supplies it |
|---|---|---|
| `v in C^s` for the form's order `s` | integration by parts | `sigma` is smooth |
| compact or rapidly decaying support | boundary terms vanish | `sigma'` decays; `band` is effectively local |
| known `integral v`, `integral v'`, ... | exact quadrature | closed-form antiderivative `S` |
| non-degenerate family | well-posed discrete system | bank of offsets, spec 01-02 |

The critical one is the third: for a bump built from the `integral` role, the
required integrals are differences of `S` at the window edges, so **integration
by parts is exact, not quadrature-approximated**. That is the property spec
02-04 monetizes.

### Boundary terms

For a window `[a, b]` and a `k`-th order form,

```
integral_a^b u^(k) v = [ sum of boundary terms ] + (-1)^k integral_a^b u v^(k)
```

The boundary terms involve `v, v', ..., v^(k-1)` at `a` and `b`. With a
`sigma'`-type bump these are not exactly zero (the tails are exponentially
small, not compactly supported), so the correct statement is: **boundary terms
are bounded by an explicit exponentially small quantity**, and that quantity can
be enclosed with `omnibias.core.verified.Interval`. Do not claim exact
compact support for an analytic base; claim a certified tail bound.

The `band` and `integral` roles with a genuinely compact base (a spline-like
tempered construction) would give exact compact support; that is a separate
design branch to be measured, not assumed.

## 5. Worked example

Base `sigma = tanh`-CDF form, that is `phi(u) = (1/2) sech^2(u)` normalized so
`integral phi = 1`. Check: `integral (1/2) sech^2 = (1/2) [tanh]_{-inf}^{inf}
= (1/2)(2) = 1`. Good.

Moments: `phi` is even, so `M_1 = 0`. The second moment is

```
M_2 = integral u^2 (1/2) sech^2(u) du = pi^2 / 12 = 0.822467
```

nonzero, so `phi` is an order-2 mollifier.

Build an order-4 mollifier as a two-scale multi-pack:

```
psi(u) = a phi(u) + b (1/2) phi(u/2) / ... 
```

Simpler and exact: use the same `phi` at two scales `eps` and `2 eps` with
weights chosen to kill `M_2`. With `phi_eps` having second moment
`eps^2 M_2`, the combination

```
psi = (4/3) phi_eps - (1/3) phi_{2 eps}
```

has total mass `4/3 - 1/3 = 1` and second moment
`(4/3) eps^2 M_2 - (1/3)(4 eps^2) M_2 = (4/3 - 4/3) eps^2 M_2 = 0`.

So `psi` is an order-4 mollifier (the third moment vanishes by symmetry). This is
Richardson extrapolation read as mollifier design, and it costs exactly two
activation evaluations.

Numerical check, smoothing `f(u) = u^2` at `u = 0` with `eps = 0.1`:

```
(phi_eps * f)(0)  = eps^2 M_2            = 8.2247e-3      (order-2 error)
(psi * f)(0)      = 0                    to rounding      (order-4: exact on u^2)
```

## 6. Proposed API

Does not exist yet.

```python
# omnibias/core/mollifier.py
@dataclass(frozen=True)
class MollifierSpec:
    base: str                     # activation name, must be CDF-type
    scale: float                  # eps
    packs: tuple[PackSpec, ...]   # reuse spec 01-01's PackSpec
    @property
    def order(self) -> int: ...   # first nonvanishing moment index

def moments(spec: MollifierSpec, up_to: int) -> tuple[float, ...]:
    """Closed form where the base allows it; otherwise certified quadrature."""
def is_admissible(spec: MollifierSpec, *, form_order: int) -> bool: ...
def design_order(base: str, order: int) -> MollifierSpec:
    """Solve for pack weights that annihilate moments 1..order-1. Exact rational
    solve, shared with `omnibias.difference` irregular stencils."""
def tail_bound(spec: MollifierSpec, *, half_width: float) -> Interval:
    """Outward-rounded bound on the mass outside the window, so weak-form
    boundary terms are enclosed rather than assumed zero."""
```

Pure Python. Tensor-side test functions are assembled in spec 02-04.

## 7. Practical use cases

1. **Weak-form PINNs** (spec 02-04) need a test-function family with exact
   integrals; this is the family and the admissibility proof.
2. **Distributional data.** Sources that are genuinely singular (point forces,
   vortex sheets, delta initial data) can be represented by the pack family with
   a stated convergence order instead of an ad hoc smoothed bump.
3. **Consistent smoothing in discovery.** `omnibias.symbolic` smooths noisy
   fields before reading derivatives; an order-`m` mollifier smooths without
   biasing polynomials of degree `< m`, which directly reduces coefficient bias.
4. **Certified quadrature weights** (spec 03-06) are moment conditions in
   disguise; the same exact solve serves both.
5. **Honest error accounting near interfaces**, where the tail bound tells you
   how much of a neighbouring layer leaked into the window.

## 8. Acceptance gates

- **G1 moment correctness.** Computed moments match closed-form values for
  `gaussian`, `sech`-type and logistic bases to `<= 1e-12` relative, and match
  certified quadrature for bases without closed form.
- **G2 order verification.** An order-`m` designed mollifier reproduces
  polynomials of degree `< m` to `<= 1e-12` and shows an empirical `O(eps^m)`
  smoothing error over at least three halvings.
- **G3 tail soundness.** `tail_bound` upper-bounds the true outside mass on a
  dense grid and a random sample, with zero violations.
- **G4 downstream.** With these test functions, the weak-form residual of spec
  02-04 on a problem with known solution reaches relative `L2 <= 1e-8`, at least
  two orders better than the same residual assembled with Gauss quadrature at
  matched cost.

## 9. Benchmark plan

- `benchmarks/mollifier_calculus.py`: moment table, smoothing-order sweep, tail
  bound tightness versus window width.
- Smoke JSON committed; full sweep under `$OMNIBIAS_SCRATCH/mollifier/`.

## 10. Honesty and scope

- This is the founding bias-collapse register. The limit is `delta -> 0` (or
  equivalently `eps -> 0`) with `K` biases coalescing; there is no `beta -> inf`
  temperature collapse anywhere in this spec.
- **No exact compact support** for analytic bases. Say "certified exponentially
  small tail", never "compactly supported", unless a genuinely compact base is
  used.
- Moment conditions give an *order*, which is an asymptotic statement in `eps`.
  Report the constant, not only the rate.
- The tail bound is a sound enclosure (`Interval`), not a theorem-prover tier.

## 11. Open questions and risks

- **Sign.** Higher-order mollifiers necessarily take negative values, so they
  are no longer densities. Anything downstream that assumed positivity (mass
  interpretations, probabilistic readings) breaks at order `> 2`; flag it in the
  type.
- **Conditioning.** Moment-annihilating weights grow with order, so an order-8
  mollifier may be numerically useless even though it is exactly derived.
- **Falsifier.** If, on realistic weak forms, plain Gauss quadrature at the same
  cost matches the exact-integral path, the main advertised win evaporates and
  the spec becomes a vocabulary contribution only.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/src/omnibias/core/mollifier.py`
- [ ] Shared exact rational solve with `omnibias.difference` irregular stencils
- [ ] Moment tests against closed-form values
- [ ] Order-verification test (polynomial reproduction plus rate)
- [ ] Tail-bound soundness test (dense grid plus random sample)
- [ ] Docs page and nav entry
- [ ] Regenerate `__all__` in `omnibias/core/__init__.py`
- [ ] Index row in `theory/README.md`
