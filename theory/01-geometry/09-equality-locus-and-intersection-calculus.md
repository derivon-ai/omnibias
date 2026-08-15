# 01-09 Equality-locus and intersection calculus

## 1. Thesis and status

Force two or more collapsed units to **agree**, and their shared level set
`f_i(x) = f_j(x)` becomes a codimension-`m` manifold whose exact Jacobian and
Hessian the derivative tower supplies in closed form, so the locus can be
followed by Newton without autodiff and its existence can be certified on a box.

- **Status**: gated (G1–G5 CI; G6 parity; constraint manifold, not a PDE solver)
- **Depends on**: 01-01
- **Blocks**: 01-10, 02-12, 02-13, 03-03, 03-08, 07-02

## 2. Where it lands

Two homes, matching the pure-core rule:

- `packages/omnibias-core/src/omnibias/core/locus.py` — the pure-Python
  geometry: residual and Jacobian assembly from an `ActivationSpec` and a
  `MultiPackSpec`, transversality tests, branch bookkeeping.
- `packages/omnibias-fields/src/omnibias/fields/locus/` — the tensor-side
  solver and its torch and jax twins, because the consumers (PINN interfaces,
  free boundaries) live in the field substrate.

## 3. Prior art in omnibias

- `packages/omnibias-pinn/src/omnibias/pinn/domain/_core/sdf.py` — `Halfspace`,
  `r_intersect_sdf`, `r_union_sdf`, `r_conjunction`, `RCompose`. Smooth exact set
  algebra on implicit surfaces (R-functions). This is the right machinery for
  combining loci and must be reused, not reinvented.
- `packages/omnibias-pinn/src/omnibias/pinn/_core/constrained.py` — the closest
  precedent for tying outputs together: the relative `periodic()` condition
  (which relates a field's value at two points) and the switching algebra used
  to blend constrained expressions.
- `packages/omnibias-core/src/omnibias/core/verified/kantorovich.py` —
  `certify_zero_radii`, `krawczyk_certificate`: rigorous existence and
  uniqueness of a zero in a box, given an interval enclosure of the Jacobian.
- `packages/omnibias-convex/.../qp_layer` — differentiating through a solved
  system by the implicit function theorem on the KKT conditions. Same technique,
  different equations.

**Confirmed gap.** Nothing in the repo ties two unit outputs together as a
constraint and treats the resulting set as a first-class object. There is no
equality locus, no Newton-on-the-locus, no transversality check, and no branch
bookkeeping.

## 4. Mathematics

### Setup

Let unit `i` have normal `w_i`, bias `b_i`, order `n_i`, outer weight `c_i`,
and write `z_i(x) = w_i . x + b_i`. After collapse,

```
f_i(x) = c_i sigma^(n_i)( z_i(x) )
```

For a family of `m + 1` units, define the residual map `F : R^D -> R^m` by

```
F(x) = ( f_1(x) - f_2(x), f_2(x) - f_3(x), ..., f_m(x) - f_{m+1}(x) )
```

and the **equality locus** `M = { x : F(x) = 0 }`.

### Closed-form derivatives

Because `d/dz sigma^(n) = sigma^(n+1)`, the chain rule through an affine
pre-activation is a single extra order:

```
grad f_i(x)   = c_i sigma^(n_i + 1)( z_i ) * w_i
Hess f_i(x)   = c_i sigma^(n_i + 2)( z_i ) * w_i w_i^T
```

Both are **exact and closed form**, computed from the same `sigma(z_i)` value
that the forward pass already produced. So the Jacobian `DF` (an `m x D` matrix
of differences of the gradients above) and every second-order object on the
locus come free. This is the crux: Newton's method on `F = 0` needs no autodiff
graph, no finite differences, and no extra activation evaluations.

### Regularity and transversality

By the regular value theorem, if `DF(x)` has full rank `m` at every `x` in `M`,
then `M` is a smooth embedded submanifold of dimension `D - m`, with tangent
space `ker DF(x)`. Full rank is checkable pointwise in closed form, and a
rigorous version follows by enclosing the singular values of `DF` over a box
with `Interval` arithmetic.

Rank deficiency is not a bug to be hidden; it is the geometry telling you the
constraints are tangent (two interfaces osculating) or redundant.

### The order-matched case is exactly affine

Suppose two units share order and weight: `n_1 = n_2 = n`, `c_1 = c_2`. Then
`F = 0` reads `sigma^(n)(z_1) = sigma^(n)(z_2)`. If `sigma^(n)` is injective the
locus is exactly

```
(w_1 - w_2) . x + (b_1 - b_2) = 0
```

an **affine hyperplane**: the bisector of the two units. If `sigma^(n)` is even
(true for `tanh` at odd `n`, since `sigma'` is even) the locus also contains the
mirror branch `z_1 = -z_2`, that is `(w_1 + w_2) . x + (b_1 + b_2) = 0`. So

> **Order-matched agreement produces flat loci; curvature comes from order
> mismatch.**

That is a clean, checkable statement, and it is the reason the interesting cases
use different orders on the two sides.

### The order-mismatched case is a genuine curved manifold in closed form

With `n_1 != n_2`, `F = 0` is a transcendental equation whose solution set is a
curved hypersurface. It is described in closed form (the equation itself), it is
differentiable in closed form (above), and it can be followed numerically by
Newton with quadratic convergence. This is the object worth having: a curved
constraint manifold with exact derivatives at no extra cost.

### Newton on the locus

Given a point `x_0` near `M`, the Gauss-Newton projection step is

```
x_{k+1} = x_k - DF(x_k)^+ F(x_k)
```

with `DF^+` the pseudoinverse (`m < D`, so this is the minimum-norm correction,
which moves normal to the locus and leaves the tangential position alone).
Because `DF` is a sum of at most `m + 1` rank-one terms `c_i sigma^(n_i+1) w_i`,
the pseudoinverse can be formed from a small `m x m` Gram matrix rather than a
`D`-sized decomposition, so the cost is `O(m^2 D)`.

### Differentiating through the locus

If the units carry parameters `theta` and a downstream loss depends on a point
`x*(theta)` on the locus, the implicit function theorem gives

```
dx* / d theta = - DF^+ (partial F / partial theta)
```

with `partial F / partial theta` also closed form. This is precisely the
technique `omnibias.convex.torch.qp_layer` uses on a KKT system, applied to a much
smaller and better conditioned system. Spec 02-12 turns this into a layer.

### Rigorous existence

`krawczyk_certificate` and `certify_zero_radii` in
`omnibias.core.verified.kantorovich` prove that a zero of `F` exists and is
unique in a box, given interval enclosures of `F` and `DF`. Both enclosures are
available because `sigma^(n)` has rigorous interval versions in
`omnibias.core.verified`. So a statement of the form

> there is exactly one interface point in this box, and it lies in
> `[x_lo, x_hi]`

is a **sound enclosure**, not an estimate. That is a genuinely new capability:
certified interface location.

### Set algebra

Loci compose: intersections of loci are handled by stacking residuals; smooth
unions and intersections of the *regions* they bound reuse the R-function CSG
already in `omnibias.pinn.domain` (`r_intersect_sdf`, `r_union_sdf`,
`RCompose`). Do not write a second CSG.

## 5. Worked example

Base `sigma = tanh`. Recall

```
sigma'   = 1 - t^2
sigma''  = -2 t (1 - t^2)
sigma''' = -2 (1 - t^2)(1 - 3 t^2)        with t = tanh(z)
```

**Case A, order-matched (flat locus).** Unit 1: order 1, `w_1 = (1, 0)`,
`b_1 = 0`. Unit 2: order 1, `w_2 = (0, 1)`, `b_2 = 0`, same `c`. Then

```
F(x, y) = sigma'(x) - sigma'(y) = tanh^2(y) - tanh^2(x) = 0  <=>  y = x  or  y = -x
```

Both branches appear because `sigma'` is even. The locus is the pair of
diagonals, exactly as the order-matched lemma predicts, and the branch
bookkeeping is not optional: a solver that ignores it will jump between
diagonals.

**Case B, order-mismatched (curved locus).** Unit 1: order 1, `w_1 = (1, 0)`,
`c_1 = 1`. Unit 2: order 2, `w_2 = (0, 1)`, `c_2 = -2`. Then

```
F(x, y) = sigma'(x) + 2 sigma''(y)
```

Find the locus point with `x = 0`. There `sigma'(0) = 1`, so we need
`sigma''(y) = -0.5`, that is `-2 t (1 - t^2) = -0.5` with `t = tanh(y)`:

```
t - t^3 = 0.25    ->    t = 0.269595...    ->    y = atanh(t) = 0.276390...
```

Check the gradient at `(0, 0.27639)`:

```
dF/dx = sigma''(0)      = 0
dF/dy = 2 sigma'''(y)   = 2 * ( -2 (1 - t^2)(1 - 3 t^2) )
      t^2 = 0.0726817,  1 - t^2 = 0.9273183,  1 - 3 t^2 = 0.7819549
      sigma'''(y) = -2 * 0.9273183 * 0.7819549 = -1.4500922
dF/dy = -2.9001844
```

`grad F = (0, -2.9002) != 0`, so the point is regular: near it the locus is a
smooth curve with tangent direction `(1, 0)`. All of these numbers came from one
`tanh` evaluation per unit and polynomial arithmetic; no autodiff, no finite
differences.

A Newton step from the off-locus point `(0, 0.20)`:

```
t = tanh(0.20) = 0.1973753,  sigma''(0.20) = -2 t (1 - t^2) = -0.3793903
F  = 1 + 2(-0.3793903) = 0.2412194
sigma'''(0.20) = -2 (1 - t^2)(1 - 3 t^2) = -2 (0.9610431)(0.8831293) = -1.6974128
dF/dy = -3.3948256
y_1 = 0.20 - 0.2412194 / (-3.3948256) = 0.2710530
```

One step lands within `5.3e-3` of the true `0.2763896`; the next step gives
`0.2763859`, and the third is exact to float64. Quadratic convergence, as
promised.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/core/locus.py
@dataclass(frozen=True)
class UnitTerm:
    order: int
    weight: float            # c_i
    normal: tuple[float, ...]
    bias: float

@dataclass(frozen=True)
class EqualitySystem:
    terms: tuple[UnitTerm, ...]     # m + 1 units -> m equations
    base: str = "tanh"
    @property
    def codimension(self) -> int: ...

def residual(sys: EqualitySystem, x) -> FloatArray: ...
def jacobian(sys: EqualitySystem, x) -> FloatArray:     # closed form, exact
    ...
def hessian_blocks(sys: EqualitySystem, x) -> FloatArray: ...
def is_transversal(sys, x, *, tol: float = 1e-10) -> bool: ...
def branch_signature(sys, x) -> tuple[int, ...]:
    """Which solution branch of each even/odd order relation the point sits on."""
def affine_locus(sys: EqualitySystem) -> tuple[AffineSet, ...] | None:
    """Exact closed-form answer for the order-matched case; None otherwise."""
```

```python
# omnibias/fields/locus/  (torch and jax twins)
def newton_project(sys, x0, *, max_iter: int = 20, tol: float = 1e-12) -> Tensor:
    """Gauss-Newton projection onto the locus. Differentiable through the
    implicit function theorem, not through the unrolled iteration."""

def locus_tangent(sys, x) -> Tensor: ...          # basis of ker DF
def certify_locus_point(sys, box: Box) -> KrawczykCertificate:
    """Wraps `omnibias.core.verified.kantorovich.krawczyk_certificate`."""
```

Rules: closed-form derivatives only (no autodiff inside `jacobian`); torch and
jax bit-identical; the gradient of `newton_project` uses the implicit function
theorem so memory does not scale with iteration count.

## 7. Practical use cases

1. **Shock and interface location.** A shock is where two states agree on a
   conserved flux; writing that agreement as an equality locus makes its position
   a differentiable, certifiable output rather than a post-hoc contour trace.
2. **Free boundaries.** Obstacle and Stefan problems are defined by an equality
   between two expressions (contact pressure, temperature at the phase front).
   The locus *is* the free boundary.
3. **Characteristic surfaces.** For first-order PDEs the characteristic
   condition is an equality between a normal speed and a wave speed; expressed as
   a locus, characteristics can be followed with closed-form derivatives.
4. **Certified interface location** for inverse problems: "exactly one interface
   in this box, located in `[x_lo, x_hi]`" is a sound statement, and it is what a
   nondestructive-testing customer actually wants.
5. **Constraint manifolds for optimization.** Any equality constraint built from
   omnibias units gets exact projection and exact tangent spaces, so
   Riemannian-style methods become available cheaply.
6. **Exact ansatz solutions** (spec 02-12): within a named ansatz class, the
   locus condition can *be* the solution of a differential equation, and the
   closed-form derivatives let you verify that claim symbolically rather than by
   fitting.

## 8. Acceptance gates

- **G1 affine lemma.** For order-matched pairs, `affine_locus` returns the
  predicted hyperplane(s), and sampled points satisfy `|F| <= 1e-14` on both
  branches when the base derivative is even.
- **G2 derivative exactness.** `jacobian` and `hessian_blocks` match
  high-precision finite differences to `<= 1e-10` relative on random systems with
  orders up to 6.
- **G3 Newton convergence.** From starting points within a stated basin, Newton
  reaches `|F| <= 1e-12` in `<= 5` iterations with an empirically quadratic rate,
  on at least 1000 random systems.
- **G4 certificate soundness.** `certify_locus_point` never claims existence
  where a dense scan plus a random sample finds no root, and every box it
  certifies does contain the root found by high-precision Newton.
- **G5 implicit gradient.** The implicit-function-theorem gradient matches an
  unrolled-Newton autodiff gradient to `<= 1e-8` relative, at constant memory.
- **G6 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/equality_locus.py`: Newton convergence statistics, transversality
  behaviour near tangency, certificate soundness sweep, implicit-versus-unrolled
  gradient accuracy and memory.
- Smoke JSON committed; full sweep under `$OMNIBIAS_SCRATCH/locus/`.

## 10. Honesty and scope

- The units are formed by the founding bias collapse (`delta -> 0`, `K` biases
  coalescing into `sigma^(K-1)`). There is **no** temperature collapse in this
  spec: nothing is annealed and no 0/1 step appears.
- **An equality locus is a constraint manifold.** It is not, by itself, a
  solution to any differential equation. The step from "locus" to "solution"
  requires a named ansatz class and a verification, which is spec 02-12's job
  and carries its own gates.
- Branch multiplicity is real. When `sigma^(n)` is not injective, the locus has
  several components; any claim about "the" locus must state which branch.
- Certified statements are **sound enclosures** (`Interval` and Krawczyk tier).
  They are not theorem-prover verified unless a genuine Lean kernel pass is
  produced, and the enclosure alone never earns that flag.
- Transversality is a hypothesis, not a guarantee. Near tangency the manifold
  claim fails and the solver must report that rather than return a point.

## 11. Open questions and risks

- **Global structure.** Newton is local. Finding *all* components of a locus is a
  global problem; sampling plus certified boxes gives a lower bound on the
  component count, never completeness.
- **Ill-conditioning near tangency.** `DF^+` blows up as the rank drops. The API
  should return a conditioning number with every projection so downstream code
  can refuse.
- **Parameter drift during training.** As `theta` moves, the locus can change
  topology (components merging or vanishing). Any layer built on this must
  detect topology change rather than silently following the wrong branch.
- **Falsifier.** If, on the target problems, a plain level-set network with
  autodiff derivatives matches this at equal cost, the closed-form-Jacobian
  advantage is not material and the spec collapses to a convenience.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/src/omnibias/core/locus.py`
- [ ] `packages/omnibias-fields/src/omnibias/fields/locus/` with torch and jax twins
- [ ] Reuse `r_intersect_sdf` / `RCompose` for set algebra; no second CSG
- [ ] Reuse `krawczyk_certificate`; no second existence test
- [ ] Affine-lemma test including the mirror branch
- [ ] Jacobian and Hessian exactness tests versus high-precision references
- [ ] Newton convergence statistics test
- [ ] Implicit-gradient versus unrolled-autodiff test with memory assertion
- [ ] Certificate soundness test (dense scan plus random sample)
- [ ] torch/jax parity test
- [ ] `benchmarks/equality_locus.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
