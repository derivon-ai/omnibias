# 01-08 Tropical-log homotopy

## 1. Thesis and status

`logsumexp_beta` is a smooth path from the log semiring to the max-plus
(tropical) semiring, and because omnibias differentiates that path exactly at
every `beta`, the whole tropical picture of a hyperplane arrangement becomes a
differentiable object with a closed-form gap to its hard limit.

- **Status**: gated (G1–G3 CI; G4 `--full` only; reuses `logsumexp_gap_bound`; cost gates smoke-earned, not in CI `all_passed`)
- **Depends on**: 01-03
- **Blocks**: 02-02, 03-01, 03-02, 03-03, 03-05

## 2. Where it lands

A submodule of `omnibias-struct`
(`packages/omnibias-struct/src/omnibias/struct/_core/tropical.py`), extending
the existing semiring driver rather than adding a package. The geometry side
(Newton polytopes) sits with spec 01-03 in `omnibias-partition`.

## 3. Prior art in omnibias

- `packages/omnibias-struct/src/omnibias/struct/_core/` — the semiring /
  hypergraph driver with `MaxPlusSemiring`, `LogSemiring` and
  `CountingSemiring`, the
  `logsumexp_beta` relaxation, and `logsumexp_gap_bound` giving the closed-form
  `log(N) / beta` gap.
- `omnibias.struct.certify_soft_dp` — the sealed soft-to-hard dynamic-programming
  gap certificate.
- `omnibias.{torch,jax}.jet` — exact differentiation of the softplus and sigmoid
  tower, which is what makes the `beta`-path differentiable to all orders.
- `omnibias.discrete` — `AnnealSchedule` and `anneal_descent`, the temperature
  axis as an optimization device.

**Confirmed gap.** The semiring driver treats `beta` as a knob to be annealed.
Nothing treats the family `{ S_beta }` as a **homotopy of algebraic structures**
with geometry attached (Newton polytopes, tropical hypersurfaces, the dual
subdivision), and nothing connects that geometry to the arrangement picture of
spec 01-03.

## 4. Mathematics

### The homotopy

For `beta > 0` define on `R`:

```
a (+_beta) b = (1 / beta) log( exp(beta a) + exp(beta b) )
a (*_beta) b = a + b
```

`(R, +_beta, *_beta)` is a commutative semiring for every finite `beta`, and

```
beta -> inf :  a (+_beta) b -> max(a, b)      (max-plus / tropical)
beta -> 0+  :  beta * (a (+_beta) b) -> log(exp(0) + exp(0)) -> counting-like behaviour
```

so the family interpolates between the tropical semiring and the log semiring in
a single parameter. The gap is uniform and closed form: for `N` terms,

```
0  <=  logsumexp_beta(x)  -  max(x)  <=  log(N) / beta
```

which is exactly `logsumexp_gap_bound` in the existing driver. That bound is the
reason this homotopy is certifiable rather than merely convenient.

### Derivatives along the path

`d/dx_i logsumexp_beta(x) = softmax(beta x)_i`, and every higher derivative is a
polynomial in the softmax values, so the tower is closed form and available from
`omnibias.{torch,jax}.jet` without autodiff graphs. Two consequences:

1. **Exact curvature on the path.** The Hessian of a `beta`-relaxed objective is
   available in closed form, so second-order methods work at every temperature.
2. **Exact `beta` sensitivity.** `d/d beta` of the relaxed value is also closed
   form, which turns "choose a schedule" into a differentiable design problem
   (spec 03-07 uses this).

### The tropical geometry attached

A tropical polynomial `max_i ( a_i + <m_i, x> )` over exponent vectors `m_i` has:

- a **Newton polytope**, the convex hull of the `m_i`;
- a **tropical hypersurface**, the locus where the max is attained twice;
- a **dual subdivision** of the Newton polytope, induced by the coefficients.

Now the connection that makes this a geometry spec: the tropical hypersurface of
a *linear* tropical polynomial is exactly the set where two affine functions
tie, that is the **equality locus** of spec 01-09, and the cells of the induced
subdivision are exactly the **arrangement cells** of spec 01-03. The three
pictures are one picture:

| Picture | Object | Hard limit | Soft version |
|---|---|---|---|
| arrangement (01-03) | cells of `sign h_i` | `beta -> inf` | product of gates |
| tropical (01-08) | argmax regions of `h_i` | `beta -> inf` | softmax weights |
| equality locus (01-09) | `h_i = h_j` | tie set | small `|h_i - h_j|` |

The tropical reading is the one that gives *combinatorial* tools: the
subdivision is computable, the number of cells is bounded by the Newton
polytope's faces, and the tie set is codimension one by construction.

### Where the certificate comes from

Because the gap `log(N) / beta` is uniform, any statement proved about the soft
object transfers to the hard object with an explicit additive slack. That is
precisely the pattern `certify_soft_dp` already implements. Extending it to
arrangement and LP objects means: **soft optimum plus `log(N)/beta` sandwiches
the hard optimum**, so a decode-and-certify pipeline gets a bound for free.

## 5. Worked example

Three affine functions on `R`:

```
h_1(x) = 0,     h_2(x) = x,     h_3(x) = 2x - 1
```

Tropical value `T(x) = max(h_1, h_2, h_3)`. Tie points: `h_1 = h_2` at `x = 0`;
`h_2 = h_3` at `x = 1`; `h_1 = h_3` at `x = 0.5`, but there `h_2 = 0.5 > 0`, so
that tie is not on the tropical hypersurface (it is dominated). The hypersurface
is `{0, 1}`, two points, dividing `R` into three cells, matching the three
argmax regions.

Soft value at `x = 0.6`, `beta = 5`:

```
h = (0, 0.6, 0.2),   beta h = (0, 3, 1)
logsumexp(beta h) = log(1 + 20.0855 + 2.71828) = log(23.8038) = 3.16969
soft value = 3.16969 / 5 = 0.633938
hard value = max(h) = 0.6
gap        = 0.033938
bound      = log(3) / 5 = 0.219722          (sound, roughly 6.5x loose here)
```

At `beta = 50` the gap falls to `0.00368` and the bound to `0.02197`. The bound
is uniform and therefore loose away from ties; near a tie (say `x = 1.0`, where
`h_2 = h_3`) the gap approaches its worst case and the bound becomes tight. That
is the correct behaviour for a certificate: loose where the problem is easy,
tight where it is hard.

Softmax weights at `x = 0.6`, `beta = 5`: `(0.0420, 0.8438, 0.1142)`. These are
the differentiable cell memberships, and their derivatives in `x` are closed
form.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/struct/_core/tropical.py
@dataclass(frozen=True)
class TropicalLinear:
    coeffs: FloatArray     # a_i,  shape (n,)
    exponents: FloatArray  # m_i,  shape (n, D)   -> here, the normals

def tropical_value(poly: TropicalLinear, x: FloatArray) -> FloatArray: ...
def relaxed_value(poly, x, *, beta: float) -> FloatArray: ...
def relaxed_weights(poly, x, *, beta: float) -> FloatArray: ...     # softmax
def homotopy_gap_bound(poly, *, beta: float) -> float:
    """log(n) / beta, reusing `logsumexp_gap_bound`."""
def newton_polytope(poly) -> tuple[tuple[float, ...], ...]:
    """Vertices of the convex hull of the exponent vectors."""
def dual_subdivision(poly) -> tuple[tuple[int, ...], ...]:
    """Cells of the regular subdivision induced by the coefficients."""
def tie_locus_samples(poly, box, *, n: int) -> FloatArray:
    """Points where the argmax is (numerically) attained twice."""
def certify_tropical_gap(poly, x, *, beta) -> TropicalGapCertificate: ...
```

`relaxed_value` and `relaxed_weights` get bit-identical torch and jax twins; the
polytope combinatorics stay pure Python.

## 7. Practical use cases

1. **Certified relaxations of piecewise-linear problems.** Shortest path, LP,
   scheduling: solve soft, decode hard, and report the `log(n)/beta` sandwich.
   This is the pattern `omnibias.struct` already proves works for dynamic
   programming, generalized to arrangement-shaped problems.
2. **Second-order annealing.** Exact Hessians along the temperature path let a
   trust-region method follow the solution branch instead of taking small
   gradient steps at each temperature.
3. **Schedule design.** Closed-form `d/d beta` makes the annealing schedule a
   differentiable object; spec 03-07 turns that into a geodesic formulation.
4. **Complexity bookkeeping.** The Newton polytope bounds how many distinct
   linear pieces the model can express, which is a capacity statement with real
   content, unlike raw parameter counts.
5. **Diagnosing degeneracy.** A near-tie means the softmax weights are spread,
   which is precisely when a hard decode is unstable. The tie locus tells you
   where to distrust the decode.

## 8. Acceptance gates

- **G1 gap soundness.** On randomized instances, `homotopy_gap_bound` upper
  bounds the measured soft-minus-hard difference on a dense grid **and** a
  random sample, with zero violations, across `beta` spanning three decades.
- **G2 subdivision correctness.** For `n <= 10` and `D <= 3`, the computed dual
  subdivision matches a brute-force reference, and its cells agree with the
  arrangement cells found by spec 01-03's sampler.
- **G3 derivative exactness.** Closed-form first and second derivatives of
  `relaxed_value` match high-precision finite differences to `<= 1e-10`
  relative.
- **G4 downstream win.** On a piecewise-linear optimization family, the
  second-order path-following method reaches the same decoded objective as
  `anneal_descent` in at least `2x` fewer objective evaluations, over five seeds,
  with the certified gap reported in both cases.

## 9. Benchmark plan

- `benchmarks/tropical_homotopy.py`: gap soundness sweep, subdivision
  correctness, path-following versus `anneal_descent`.
- Smoke JSON committed; full sweep under `$OMNIBIAS_SCRATCH/tropical/`.

## 10. Honesty and scope

- The limit here is **temperature collapse**: `beta -> inf` hardens a softmax
  into an argmax, a 0/1 feasibility-sense collapse. It is **not** the founding
  bias collapse, which is the `delta -> 0` coalescence of `K` biases producing a
  smooth `sigma^(K-1)`. Any module added under this spec that carries a
  relaxation must include the cross-reference note and be registered in
  `PENALTY_FILES` in
  `packages/omnibias-core/tests/test_concept_terminology.py`.
- The gap bound is **uniform and therefore loose** away from ties. Report both
  the bound and the measured gap; never present the bound as the error.
- Tropical geometry gives capacity and structure statements. It does not give
  complexity-theoretic conclusions: nothing here bears on P versus NP, and a
  certified sandwich on one instance is one instance.
- Tie-locus sampling is numerical and returns *samples*, never a claim to have
  found the whole hypersurface.

## 11. Open questions and risks

- **Subdivision cost** is exponential in `D`; the API must refuse large inputs
  rather than run for hours.
- **Numerical ties.** Deciding a tie in floating point needs a tolerance, and
  the tolerance leaks into the combinatorics. Interval arithmetic is the right
  fix and should be used for any certified statement.
- **Falsifier.** If path-following never beats plain annealing at equal cost,
  the exact-derivative advantage is not materializing and the spec is a
  vocabulary contribution.

## 12. Implementation checklist

- [ ] `packages/omnibias-struct/src/omnibias/struct/_core/tropical.py`
- [ ] torch and jax twins for `relaxed_value` and `relaxed_weights`
- [ ] Reuse `logsumexp_gap_bound`; do not fork the constant
- [ ] Gap soundness test (dense grid plus random sample, three decades of `beta`)
- [ ] Subdivision correctness test versus brute force
- [ ] Cross-reference terminology note plus `PENALTY_FILES` registration
- [ ] `benchmarks/tropical_homotopy.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
