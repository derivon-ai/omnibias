# 01-04 Irregular and Birkhoff stencils with certified error

## 1. Thesis and status

Generate **exact rational finite-difference weights for arbitrary node sets and
arbitrary per-node order sets** by solving a confluent Vandermonde system over
the rationals, with a certified truncation bound, so the multi-pack construction
has a rigorous numerical twin.

- **Status**: gated (G1–G4 earned)
- **Depends on**: 01-01
- **Blocks**: 01-11, 03-06, 07-03

## 2. Where it lands

A submodule of `omnibias-difference`:
`packages/omnibias-difference/src/omnibias/difference/_core/irregular.py`, with
public re-exports from `omnibias.difference`. Same domain (the `delta -> 0`
register), same tier, same audience as the existing stencil code.

## 3. Prior art in omnibias

- `packages/omnibias-difference/src/omnibias/difference/_core/stencil.py` —
  `signs_exact(order, delta: Fraction) -> tuple[Fraction, ...]`,
  `offsets_exact(order, delta, stencil="central"|"forward")`,
  `stencil_signs`, `stencil_offsets`, `accuracy_order(stencil)`.
- `omnibias.difference` — `finite_difference_estimate`, `certified_fd_error`,
  `certified_fd_error_general(f_float, deriv_bound, z, order, delta, stencil)`,
  `certified_derivative_enclosure`, `FiniteDifferenceCertificate`.
- `packages/omnibias-torch/src/omnibias/torch/stencil.py` — the tensor-side
  uniform stencil helpers.
- `omnibias.core.verified.interval` — outward-rounded `Interval` arithmetic.

**Confirmed gap.** Every generator in the repo is a **uniform** forward or
central stencil parameterized by `(order, delta)`. There is no API for irregular
node sets, no per-node order sets, and no weight system for lacunary (Birkhoff)
data. `certified_fd_error_general` certifies an arbitrary function but still
evaluates the same regular stencil.

## 4. Mathematics

### The weight system

Let nodes `t_1 .. t_m` (distinct, relative to the expansion point) each carry an
order set `O_i` (which derivatives of `f` are available at `t_i`). Write the
available data as pairs `(i, p)` with `p in O_i`, and let `N` be their number.
We want weights `a_{i,p}` such that

```
sum_{(i,p)} a_{i,p} * f^(p)(t_i)  =  f^(q)(0) + O(h^r)
```

for a target order `q`. Applying the requirement to the monomial basis
`f(t) = t^j / j!` for `j = 0 .. N-1` gives the **confluent Vandermonde system**

```
sum_{(i,p)} a_{i,p} * t_i^{j-p} / (j-p)!  =  [j == q],      j = 0 .. N-1
```

with the convention that terms with `j < p` vanish. The matrix is square and
rational whenever the nodes are rational, so the solve can be done exactly with
`fractions.Fraction` and Gaussian elimination over `Q`.

Three consequences.

1. **Exact rational weights.** No floating-point weight generation, so the
   weights themselves are certifiable objects (spec 01-11).
2. **Poisedness is a rank condition.** The scheme is poised exactly when the
   confluent Vandermonde matrix is nonsingular. Over `Q` this is a decidable,
   exact test, unlike the numerical rank test that spec 01-01 falls back to.
   This spec therefore supplies the *authoritative* poisedness oracle.
3. **Order of accuracy.** Write the nodes in units of the scale, `t_i = c_i h`
   with `c_i` rational. The first unmatched monomial is `j = N`, and its
   residual is the leading error:

```
E = C_N * h^(N-q) * f^(N)(xi),      C_N = sum_{(i,p)} A_{i,p} c_i^(N-p) / (N-p)!
```

where `A_{i,p} = h^q a_{i,p}` are the scale-free weights. The generic order is
`r = N - q`, and it is higher when `C_N` vanishes by symmetry, which is exactly
why the uniform central stencil gets `O(h^2)` instead of `O(h)`.

### Certified truncation bound

Given an interval bound `M_N` on `|f^(N)|` over the hull of the nodes,

```
|E| <= |C_N| * h^(N-q) * M_N
```

with `C_N` computed exactly over `Q` and converted outward to an `Interval`.
This slots directly into the existing `FiniteDifferenceCertificate` shape.

### Conditioning

The weights grow like `h^{-q}` and the condition number of the Vandermonde
system grows rapidly with `N` and with node clustering. The generator must
report a conditioning diagnostic (an exact rational bound on the weight
magnitudes, plus the ratio `max|a| / min node separation^q`) so a caller can
tell a well-posed request from a hopeless one *before* spending float
evaluations on it.

### Relationship to multi-pack collapse

Spec 01-01's multi-pack unit is the `delta -> 0` limit of exactly these
stencils, restricted to the case where the data at node `i` is a single order
`n_i`. Running the generator at finite `h` gives the collapsing family; taking
`h -> 0` gives the closed form. Having both means every closed-form claim has a
numerical cross-check with a certified error bar.

## 5. Worked example

Target: `f'(0)` from **value at `-h`, value at `0`, derivative at `h`**. This is
a genuine Birkhoff scheme with a gap: no derivative at `-h` or `0`, no value at
`h`. No uniform stencil in the repo can express it.

Data pairs `(c=-1, p=0)`, `(c=0, p=0)`, `(c=1, p=1)`, so `N = 3`, `q = 1`.
Imposing the conditions on `f(t) = t^j / j!` for `j = 0, 1, 2`:

```
j = 0:   f = 1,        f' = 0        ->  a1 + a2               = 0
j = 1:   f = t,        f' = 1        ->  a1 (-h)        + a3   = 1
j = 2:   f = t^2 / 2,  f' = t        ->  a1 (h^2 / 2) + a3 h   = 0
```

From row 0, `a2 = -a1`; from row 2, `a3 = -a1 h / 2`; substituting into row 1
gives `-3 a1 h / 2 = 1`, hence

```
a1 = -2 / (3h),   a2 = 2 / (3h),   a3 = 1/3

f'(0) ~= ( -2 f(-h) + 2 f(0) ) / (3h)  +  f'(h) / 3
```

All three weights are exact rationals in `h`, as required.

Leading error coefficient, from the `j = 3` monomial:

```
C_3 = A_1 (-1)^3 / 3!  +  A_3 (1)^2 / 2!      with A_1 = -2/3, A_3 = 1/3
    = (-2/3)(-1/6) + (1/3)(1/2) = 1/9 + 1/6 = 5/18
```

so `E ~= (5/18) h^2 f'''(0)`, an `O(h^2)` scheme.

Numerical check on `f = exp`, where `f'(0) = 1` and `f''' = 1`:

| `h` | estimate | error | `(5/18) h^2` |
|---|---|---|---|
| 0.1 | 1.0028076 | 2.808e-3 | 2.778e-3 |
| 0.01 | 1.0000281 | 2.810e-5 | 2.778e-5 |
| 0.001 | 1.0000003 | 2.778e-7 | 2.778e-7 |

The error tracks the predicted coefficient to three digits and the rate is
exactly 2. That is the whole deliverable: an unusual data pattern, exact
weights, a predicted constant, and a bound that holds.

The trap this example also encodes: if the nodes are given at `O(1)` spacing
rather than as multiples of `h`, the "order" is meaningless, because there is no
small parameter. The generator therefore takes nodes in units of `h` and returns
weights as exact rationals in `h`.

## 6. Proposed API

Gated. `omnibias.difference._core.irregular`, re-exported from
`omnibias.difference`. Scale-free weights use `A_{i,p} = a_{i,p} h^{q-p}`
(the `A = h^q a` form in the worked-example arithmetic below only closes
after that correction).

```python
# omnibias/difference/_core/irregular.py
@dataclass(frozen=True)
class StencilRequest:
    nodes: tuple[Fraction, ...]            # in units of h
    orders: tuple[tuple[int, ...], ...]    # available orders per node
    target_order: int                      # q

@dataclass(frozen=True)
class IrregularStencil:
    request: StencilRequest
    weights: tuple[tuple[Fraction, ...], ...]   # a_{i,p}, matching `orders`
    accuracy: int                                # r, asymptotic in h
    leading_coeff: Fraction                      # C_N
    weight_magnitude: Fraction                   # conditioning diagnostic

def solve_irregular_stencil(request: StencilRequest) -> IrregularStencil | None:
    """Exact rational solve. Returns None when the confluent Vandermonde system
    is singular, that is when the Birkhoff scheme is not poised."""

def is_poised_exact(request: StencilRequest) -> bool:
    """Authoritative poisedness oracle: exact rank over the rationals."""

def polya_screen(request: StencilRequest) -> bool:
    """Cheap necessary condition; False proves not poised, True proves nothing."""

def certified_irregular_error(
    stencil: IrregularStencil, *, h: Fraction, deriv_bound: Interval
) -> FiniteDifferenceCertificate:
    """|E| <= |C_N| h^r * bound, outward rounded."""
```

Pure Python, no tensor imports, consistent with the rest of `omnibias-difference`.

## 7. Practical use cases

1. **The authoritative poisedness oracle for spec 01-01.** Multi-pack
   representation claims need an exact answer, not a numerical rank guess.
2. **Non-uniform grids.** Compactified coordinates (as used in the CCF line
   problems) produce non-uniform nodes where uniform stencils lose order
   silently. An exact generator restores the stated accuracy.
3. **Mixed data.** Problems where values and fluxes are known at different
   points (sensors, interface conditions) are Birkhoff data by construction.
4. **Certified quadrature and differentiation** (spec 03-06) needs exact weights
   with rigorous remainders, not float weights with hoped-for accuracy.
5. **Lean-checkable identities** (spec 01-11): the exact weights make the
   consistency conditions finite rational statements.

## 8. Acceptance gates

- **G1 reproduction.** For uniform node sets with single orders, the generator
  reproduces `signs_exact` and `offsets_exact` exactly (as `Fraction`
  equalities, not float comparisons).
- **G2 order verification.** For every generated stencil, the empirical
  convergence rate measured on a battery of analytic functions matches the
  reported `accuracy` to within `0.15` over at least three halvings of `h`.
- **G3 certificate soundness.** `certified_irregular_error` bounds the observed
  error on a dense grid **and** a random sample, with zero violations.
- **G4 poisedness correctness.** On a curated set of known-poised and
  known-unpoised Birkhoff schemes from the interpolation literature,
  `is_poised_exact` is correct on every case, and `polya_screen` never returns
  `False` for a poised scheme.

## 9. Benchmark plan

- `benchmarks/irregular_stencils.py`: convergence-rate table, conditioning
  versus node clustering, and wall time versus `N`.
- Smoke tier committed as JSON; full sweep under
  `$OMNIBIAS_SCRATCH/stencils/`.
- Because this is a rigorous primitive, the soundness test (dense grid plus
  random sample) is a unit test, not only a benchmark.

## 10. Honesty and scope

- This is the founding `delta -> 0` bias-collapse register: finite differences
  becoming derivatives. No temperature collapse appears.
- Order claims are **asymptotic in the node scale `h`**. A stencil with nodes at
  `O(1)` spacing has no meaningful order; the worked example above shows exactly
  that trap.
- The certificate bounds truncation only. Rounding error in the *float*
  evaluation of the stencil is separate and is exactly the cancellation the
  closed-form tower exists to avoid; both must be reported.
- Poisedness is decidable here; do not export a float-rank test as if it were
  authoritative.

## 11. Open questions and risks

- **Exact rational solves get expensive** as `N` grows (coefficient blowup in
  Gaussian elimination over `Q`). Measure; add a modular or fraction-free
  algorithm if needed.
- **Conditioning versus exactness.** Exact weights can still be catastrophically
  ill-conditioned when *applied* in floating point. The diagnostic must be
  prominent, and the closed-form path preferred whenever it exists.
- **Falsifier.** If, on every realistic node set, the uniform stencil plus
  interpolation is as accurate and cheaper, the generator is a nicety rather
  than a capability.

## 12. Implementation checklist

- [x] `packages/omnibias-difference/src/omnibias/difference/_core/irregular.py`
- [x] Public re-exports from `omnibias.difference` with regenerated `__all__`
- [x] Reproduction test versus `signs_exact` / `offsets_exact`
- [x] Convergence-rate test battery
- [x] Certificate soundness test (dense grid plus random sample)
- [x] Poisedness test set from the interpolation literature
- [x] `benchmarks/irregular_stencils.py` plus smoke JSON
- [x] Docs page and nav entry
- [x] Index row in `theory/README.md`
