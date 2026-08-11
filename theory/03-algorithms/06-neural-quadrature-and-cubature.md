# 03-06 Neural quadrature and cubature

## 1. Thesis and status

A pack's moments are closed form, so a bank of packs is a quadrature rule whose
nodes and weights can be *solved* for exactness on a prescribed function space
— and, unlike a classical rule, the resulting rule is differentiable in its own
design parameters.

- **Status**: designed
- **Depends on**: 01-04, 01-05, 02-04
- **Blocks**: 03-13, 07-02

## 2. Where it lands

`packages/omnibias-fields/src/omnibias/fields/_core/quadrature.py` (extend the
existing `QuadratureSpec` surface) plus a solver in
`packages/omnibias-core/src/omnibias/core/cubature.py`.

## 3. Prior art in omnibias

- `packages/omnibias-fields/src/omnibias/fields/_core/quadrature.py` —
  `QuadratureSpec`, consumed by the `integrate` and `inner_product` field ops.
- `docs/operator-surface.md` — the `integral` role's antiderivative window.
- `omnibias.measure` — layer-cake and simple-function primitives, the measure
  integral.
- `omnibias.variational` — action integrals with rigorous action enclosures,
  which need quadrature and currently rely on standard rules.
- Spec 01-04's confluent Vandermonde solver — the same linear-algebra core, on
  derivative values rather than integrals.
- Spec 01-05's moment conditions — the mathematics of pack moments.

**Confirmed gap.** `QuadratureSpec` describes standard rules. There is no rule
*construction*, no moment solving, and no differentiable quadrature design.

## 4. Mathematics

### Moments of a pack

For a tempered pack `p_g(x) = alpha_g sigma^(n_g)( alpha_g (x - mu_g) )`, the
`j`-th moment is

```
M_j(g) = integral x^j p_g(x) dx
```

By repeated integration by parts and the exact scaling law
`sigma_alpha^(n)(u) = alpha^n sigma^(n)(alpha u)`, this reduces to a finite
combination of the base activation's moments at the pack's centre and scale.
For `n_g >= 1` the pack integrates to zero, its first moment is
`-mu_g`-weighted, and so on — the point being that every moment is a **closed-form
function of `(mu_g, alpha_g, n_g)`**, not a quadrature.

### Constructing a rule

A quadrature rule `integral f ~ sum_g w_g f(x_g)` is exact on a space `V` when
it reproduces a basis of `V`. Writing the exactness conditions for the monomials
up to degree `d`:

```
sum_g w_g x_g^j = M_j(target measure),    j = 0 .. d
```

a Vandermonde system in the nodes and weights, exactly the confluent structure
spec 01-04 solves. With free nodes it is nonlinear (Gauss-type); with fixed
nodes it is linear (Newton-Cotes-type).

**What packs add.** Instead of point evaluations, use pack functionals:

```
integral f ~ sum_g w_g < p_g, f >
```

Each `< p_g, f >` is a *smoothed* sample of `f` (spec 01-05: a mollified
evaluation or derivative). Two consequences:

1. The rule is exact on the space spanned by functions whose pack moments match,
   which includes derivative information — a Birkhoff-type quadrature.
2. The functionals are **smooth in the nodes**, so the rule is differentiable
   with respect to node positions, which a point-evaluation rule is not (a node
   moving past a discontinuity of `f` is not differentiable).

### Differentiable rule design

Since the moment conditions are closed form in `(mu, alpha, n, w)`, the
exactness residual

```
R(theta) = sum_j | sum_g w_g M_j(g) - M_j^target |^2
```

is a smooth function of the design parameters, minimizable by the same
second-order machinery the repo already has. So designing a rule for a *given
integrand family* becomes an optimization rather than a table lookup. That is
the practical content: a rule specialized to the functions you actually
integrate.

### Error bound

For a rule exact to degree `d`, the Peano kernel theorem gives

```
| error |  <=  ( 1 / (d+1)! ) * || K_d ||_1 * || f^(d+1) ||_inf
```

and the Peano kernel `K_d` is computable in closed form from the pack moments.
Combined with an interval bound on `f^(d+1)` — which, for an omnibias field, is
available from the tower — the quadrature error becomes a **certified
enclosure** rather than an estimate.

That is the strongest statement in this spec and the one most worth building:
`integrate` returning an `Interval` instead of a float.

### Cubature in higher dimensions

Tensor products work and cost `O(M^D)`. Sparse grids (Smolyak) reduce that at
the cost of a worse constant. Genuinely `D`-dimensional cubature rules are hard
and largely tabulated; the honest position is to support tensor and sparse
constructions and to say plainly that non-product cubature design is out of
scope.

## 5. Worked example

**A two-node rule exact on cubics, derived by hand.**

Target: `integral_{-1}^{1} f(x) dx` with point nodes (the classical case, as a
validation of the machinery before adding packs).

Two nodes `+-x_0` with equal weights `w` by symmetry. Exactness on `1` and `x^2`:

```
j = 0:  2 w        = 2         =>  w = 1
j = 2:  2 w x_0^2  = 2/3       =>  x_0^2 = 1/3,  x_0 = 0.5773503
```

Odd moments vanish by symmetry, so the rule is exact on cubics: this is
two-point Gauss-Legendre, recovered from the moment system.

**Check on `f(x) = x^4`.** Exact integral `2/5 = 0.4`. Rule gives
`2 * (1/3)^2 = 2/9 = 0.2222222`. Error `0.1777778`.

Peano bound with `d = 3`: `f^(4) = 24`, and for two-point Gauss on `[-1, 1]` the
classical error constant is `(b-a)^5 / 4320 * f^(4)(xi) = 32/4320 * 24 =
0.1777778`. **The bound is attained exactly**, because `f^(4)` is constant —
which is the right sanity check that the error machinery is correct rather than
merely plausible.

**Now the pack version.** Replace point evaluation at `x_0` by a pack functional
`< p, f >` with `p` a normalized order-0 bump of width `1/alpha` centred at
`x_0`. The rule is no longer exact on cubics: the bump's second moment
contributes a bias

```
< p, f >  =  f(x_0) + ( 1 / (2 alpha^2) ) c_2 f''(x_0) + O(alpha^-4)
```

where `c_2` is the bump's normalized second moment. So the pack rule has an
extra `O(alpha^-2)` term, and either

- `alpha` is chosen large enough that the term is below the target tolerance, or
- the moment conditions are re-solved to *absorb* it, giving a corrected rule
  that is again exact on cubics with finite `alpha`.

The second option is the interesting one and is what "solve for exactness"
means here: the smoothing is not an error to be minimized but a known term to be
cancelled. For `c_2 = 1` and `alpha = 10` the bump's second moment is `0.01`, so
on `f = x^4` (with `f'' = 12 x^2 = 4` at the nodes) the uncorrected bias is
`0.005 * 4 = 0.02` per node, `0.04` in total against an exact value of `0.4`: a
ten percent error. Large enough that cancellation is clearly worthwhile, which
is the point of the example.

## 6. Proposed API

Extends the existing quadrature surface.

```python
# omnibias/core/cubature.py
@dataclass(frozen=True)
class MomentSystem:
    degree: int
    measure: Literal["lebesgue", "gaussian", "custom"]
    target_moments: tuple[float, ...]

def pack_moment(pack: PackSpec, j: int) -> float:
    """Closed form. Exact rational where the base allows."""

def solve_rule(
    system: MomentSystem, *, nodes: int, free_nodes: bool = True,
    functional: Literal["point", "pack"] = "point",
    pack_scale: float | None = None,
) -> QuadratureRule: ...

def peano_kernel(rule: QuadratureRule, *, degree: int) -> Callable[[float], float]: ...
def certified_error(rule, *, deriv_bound: Interval, degree: int) -> Interval:
    """Sound enclosure of the quadrature error."""
```

```python
# omnibias/fields/_core/quadrature.py  (additions)
def integrate_certified(field, region, rule: QuadratureRule) -> Interval:
    """Returns an enclosure, not a float. The derivative bound comes from the
    field's own tower."""

def design_rule(integrand_family, *, nodes: int, optimizer) -> QuadratureRule:
    """Differentiable rule design against a specific integrand family."""
```

## 7. Practical use cases

1. **Certified integration in the variational package**, upgrading action
   enclosures from a standard rule plus an estimate to a rule plus a bound.
2. **Weak-form assembly** (spec 02-04), where the quadrature error currently
   competes with the discretization error.
3. **Specialized rules** for integrand families that appear repeatedly — for
   example integrals of activation products, which are ubiquitous here.
4. **Differentiable domain integrals** where the region moves: pack functionals
   are smooth in node position, so the integral is differentiable in the
   geometry.
5. **Adaptive refinement** (spec 03-13) driven by the certified error rather
   than by a heuristic indicator.

## 8. Acceptance gates

Baselines: Gauss-Legendre, Clenshaw-Curtis, and (for higher dimensions) sparse
grids, all at matched node count.

- **G1 classical recovery.** `solve_rule` with point functionals reproduces
  Gauss-Legendre nodes and weights to `<= 1e-13` for `nodes = 2 .. 12`.
- **G2 moment exactness.** Constructed rules integrate their design space
  exactly, to `<= 1e-13` relative, including the pack-functional corrected case.
- **G3 certified error soundness.** `certified_error` contains the true error on
  a dense grid **and** a random sample of integrands, with zero violations, and
  is attained (to within a small factor) on the extremal case, as the worked
  example demonstrates.
- **G4 design win.** A rule designed for a specific integrand family beats
  Gauss-Legendre at matched node count on that family by at least `10x` in
  error, over five seeds of family parameters.
- **G5 honest scope.** The dimension at which tensor-product cost becomes
  prohibitive is measured and recorded; no non-product cubature is claimed.

## 9. Benchmark plan

- `benchmarks/neural_quadrature.py`: classical recovery, moment exactness,
  certified-error tightness, designed-rule comparison, dimension scaling table.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/quadrature/`.

## 10. Honesty and scope

- Pack functionals come from the founding bias collapse (`delta -> 0`). No
  temperature collapse appears.
- **Gauss-Legendre is very hard to beat on generic smooth integrands.** The win
  in G4 is specifically on a *specialized* family; any comparison on generic
  integrands should be reported even where the classical rule wins.
- Pack functionals introduce an `O(alpha^-2)` smoothing bias unless it is
  explicitly cancelled by the moment conditions. Which path was taken must be
  recorded in the rule.
- Certified error requires a bound on `f^(d+1)`. For an omnibias field the tower
  provides it; for an arbitrary integrand it does not, and the function must
  refuse rather than assume.
- Non-product cubature in high dimensions is **out of scope**, stated plainly.
- Certificate tier: sound enclosure for the quadrature error.

## 11. Open questions and risks

- **Nonlinear moment systems** (free nodes) can be ill-conditioned or have no
  real solution for some target spaces. Detect and report rather than returning
  a poor local minimum.
- **Pack functional cost.** Each functional is a small integral itself; if it is
  not closed form, the rule is more expensive than a point rule and the
  advantage evaporates. Only families with closed-form pack moments should be
  supported.
- **Peano kernel computation** for pack functionals is more involved than for
  point rules; verify it numerically before trusting it.
- **Falsifier.** If designed rules never beat Gauss-Legendre by a useful margin
  on realistic integrand families, the remaining value is the certified error,
  which is real but much narrower than the spec's framing.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/src/omnibias/core/cubature.py`
- [ ] Extend `omnibias/fields/_core/quadrature.py` with `integrate_certified`
- [ ] Reuse spec 01-04's confluent Vandermonde solver; do not fork it
- [ ] Classical-recovery test against tabulated Gauss-Legendre values
- [ ] Certified-error soundness test, including the extremal attainment case
- [ ] Smoothing-bias cancellation test for the pack-functional path
- [ ] Refusal path when no derivative bound is available
- [ ] Dimension-scaling table in the benchmark
- [ ] `benchmarks/neural_quadrature.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
