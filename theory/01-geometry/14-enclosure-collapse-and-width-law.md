# 01-14 Enclosure Collapse and the Width Law

> Informally called interval collapse; the public name is Enclosure
> Collapse because `Interval`, conformal slabs, and IA wrapping are
> different objects.

## 1. Thesis and status

Enclosure Collapse is the `width -> 0` limit of a **sound enclosure**
`[lo, hi]`. The output is a point plus a proof, or `Inconclusive`. The
closed-form object is the Width Law: the leading term of `w(r)` is a
tower derivative. The operation is squeeze: contract a named enclosure
until a certificate fires, or return `Inconclusive`. Forcing `lo = hi`
by clamping is unsound and forbidden.

- **Status**: gated
- **Depends on**: 03-08, 07-02, 07-06, 08-04, 09-18
- **Blocks**: none

### Operator card

- **Benefit.** One named third collapse, one Width Law, one squeeze API
  that actually solves six already-gated problems.
- **How it works.** Algebra in `omnibias.core.verified.enclosure_collapse`;
  product facade in `omnibias.verify.enclosure_collapse` wrapping existing
  engines.
- **Strength.** Predicted `w(r)` from `sigma^(n+1)`, not a clamp.
- **When to use.** When a sound enclosure must shrink until a certificate
  fires.
- **When not.** Not a derivative. Not a 0/1 step. Not the integral window
  `S(z+b_hi)-S(z+b_lo)`. Not a new package.
- **Accuracy floor.** Outward rounding; Lemma Floor (`w >= 2 ulp` after a
  non-exact sound evaluation).

## 2. Where it lands

Algebra: `omnibias.core.verified.enclosure_collapse` (Apache-2.0).
Product facade: `omnibias.verify.enclosure_collapse` (AGPL). No new
package. A permissive package must never depend on a copyleft one, so
the Width Law stays in `omnibias.core.verified` and the six-problem
driver stays in `omnibias.verify`.

## 3. Prior art in omnibias

Confirmed present:

- `Interval.width`, outward `nextafter` (`omnibias.core.verified.interval`)
- `AffineForm` dependency cancellation
- `TaylorModel` / `TaylorModelMV` remainder
- `WidthBudget(truncation, jacobian, wrapping, rounding)` with `.dominant`
  in `omnibias.core.verified.jet_flow` — missing a recommended *action*
  and a Width Law prediction
- `WidthReport` in `omnibias.pinn.certified.weak_form` (07-02) — a related
  split; not a second budget type
- `krawczyk_certificate`, `radii_polynomial_certificate`,
  `kantorovich_accept_step`
- `certify_peak` / `Inconclusive` (03-08)
- `certified_interior_residual`, `adaptive_certified_interior_residual`,
  `aposteriori_error_certificate`
- `certify_trained_global_min` — `f_lower > 0` Lean path exists;
  missing a first-class `squeeze_identifiability`
- `remainder_loss` / `RemainderTrainConfig` (09-18) — missing a
  `WidthBudget.truncation` tag
- PCI `width_cap` reject; 08-09 certified step filter — cite, do not fork

**Confirmed gap.** No named third collapse, no Width Law theorems, no
single `SqueezeReport` / `squeeze(kind, ...)` driver through which all
six problems are solvable.

## 4. Mathematics

This is **not** the founding bias collapse (`delta -> 0`, `K` biases,
output `sigma^(K-1)`). It is **not** temperature collapse
(`beta -> inf`, a 0/1 feasibility step). Enclosure Collapse is the
`width -> 0` limit of a *sound enclosure*.

### Definition 1 (Sound enclosure)

An interval `I = [lo, hi]` is a sound enclosure of a real quantity `q`
iff `lo <= q <= hi` after outward rounding. Width `w(I) = hi - lo >= 0`.
Identifying endpoints (`lo := hi`) is **not** an allowed operation.

### Definition 2 (Width budget)

For a computed enclosure of `Q` over a box of radius `r` (or a flow of
horizon `T`),

```
w = w_true + w_wrap + w_trunc + w_round
```

where `w_true` is the oscillation of the true `Q` on the domain,
`w_wrap` is dependency / wrapping overestimation, `w_trunc` is the
jet / Taylor-model remainder `|R_N|`, and `w_round` is the
outward-rounding floor (ulps).

This is the existing `WidthBudget` split (`truncation`, `jacobian`,
`wrapping`, `rounding`), with `jacobian` the structured part of wrapping
from `DF`. `.total` is an *upper* split, not an equality in exact reals.
`w_true` is not a `WidthBudget` field.

### Definition 3 (Enclosure Collapse)

Enclosure Collapse is any sound procedure that strictly decreases a
named width (or a named certified gap `f_upper - f_lower`) and stops
when a certificate predicate holds or the budget is exhausted
(`Inconclusive`). The output is a point plus a proof, or
`Inconclusive`. It is a limit of **certificates**, not of biases and
not of temperature. There is no operator `Collapse([lo,hi]) -> point`
analogous to `Collapse(K biases) -> sigma^(K-1)`.

### Lemma Floor (rounding floor)

**Statement.** There exist constants `q` and IEEE-754 evaluations such
that every sound machine enclosure of `q` satisfies `w >= 2 ulp`.
In particular, `lo = hi` is not a sound general target.

**Proof sketch.** `Interval` pushes each endpoint one `nextafter`
outward on a non-exact operation. `Interval.point(1.0) / 3.0` is the
shipped witness. `Interval.from_value(float)` is an exact point and is
**not** the witness.

**Consequence.** Training that “forces `lo = hi`” is either unsound
or is secretly shrinking the *box*, not the enclosure.

### Lemma Constant Residual

**Statement.** If `Q` is identically `c` on a box `B`, then
`w_true(B) = 0` and any positive width of a sound enclosure of `Q(B)`
is wrapping + truncation + rounding. Subdivision of `B` drives wrapping
to the rounding floor for inclusion-isotonic interval extensions of
compositions of elementary maps (Moore).

**Worked witness.** `u = cos(w · x)` on the unit square, Helmholtz
residual `Δu + |w|^2 u` is identically 0.
`certified_interior_residual` width decreases as `splits` increases
(`1 -> 1.60`, `4 -> 0.936`, `16 -> 0.257`, `64 -> 0.065`). The facade
exposes this as `squeeze_residual(..., until="ulps" | width_cap=...)`.

### Lemma Centered Form

**Statement.** If `f` is C^1 on a convex box `X` and `f'(X)` is a sound
enclosure of `{f'(x) : x in X}`, then

```
f(X)  ⊆  f(c) + f'(X) · (X - c)     for any c in X.
```

If `‖f'(X)‖ <= L` then `w(f(X)) = O(r)` with leading factor `2L`
(1-D). At a critical point `0 ∈ f'(X)` the first-order form does not
certify a unique minimizer; order 2 is required.

**omnibias delta.** `f'` is the **closed-form** verified tower
(`sigma_tower_interval` / interval `sigma^(n+1)`), not nested autodiff.
`certified_minimize` already uses this.

### Theorem Width Law

**Statement.** Let `σ` be a Riccati activation whose derivative tower
is closed-form. Let `f = σ^(n)` (or a finite linear combination of
tower terms). Let `I(r) = [c-r, c+r]` and let `[f](I(r))` be the
mean-value interval extension that uses the closed-form enclosure of
`f'`. Then as `r -> 0+`,

```
w([f](I(r))) / (2r)  ->  |f'(c)| = |σ^(n+1)(c)|.
```

If `f'(c) = 0` and `f''(c) ≠ 0`, then (1-D factor locked to `1` in
the implementation)

```
w([f](I(r))) / r^2  ->  |f''(c)|
```

using the second-order form `f(c) + f''(I) · (I-c)^2`.

For a Taylor remainder of order `N` using a sound enclosure of
`f^{(N+1)}` on `I(r)`,

```
w(R_N(r)) / r^{N+1}  ->  |f^{(N+1)}(c)| / (N+1)!
```

up to the even/odd factor on `[-r, r]^{N+1}` (2 when `N+1` is odd, 1
when even).

**This is the analog of Lemma Collapse.** Bias collapse produces
`σ^(n)`. The Width Law produces the **leading coefficient of enclosure
width** from `σ^(n+1)`.

## 5. Worked example

Mean-value Width Law at `sigmoid^(0)` about `c = 0.5`, `r = 1e-4`:

```
f = sigmoid, n = 0
predicted_leading ≈ |sigmoid'(0.5)|
w_measured / (2r) → predicted_leading
```

Helmholtz constant residual (cookbook numbers):

```
u = cos(w · x), w = (1.3, -0.7), domain = [0,1]^2
splits: 1 → 1.60, 4 → 0.936, 16 → 0.257, 64 → 0.065
```

Lemma Floor: `Interval.point(1.0) / 3.0` has `width >= 2 ulp`.

## 6. Proposed API

No torch/jax twins of the Width Law. Remainder twins already exist.

```python
@dataclass(frozen=True)
class WidthLaw:
    center: float
    order: int
    predicted_leading: Interval
    exponent: int
    activation: str
    kind: Literal["mean_value", "taylor_remainder", "critical_point"]

@dataclass(frozen=True)
class RecommendedAction:
    dominant: Literal["truncation", "jacobian", "wrapping", "rounding"]
    action: Literal["raise_order", "subdivide", "shrink_step", "stop_floor"]
    reason: str

width_law(activation, n, center, *, kind=...) -> WidthLaw
predicted_width(law, r) -> Interval
measured_mean_value_width(activation, n, center, r) -> Interval
diagnose_width(budget: WidthBudget) -> RecommendedAction
rounding_floor_witness() -> Interval

SqueezeKind = Literal["peak", "residual", "identifiability", "existence", "remainder", "flow"]
squeeze(kind, **kwargs) -> SqueezeReport
squeeze_peak / squeeze_residual / squeeze_identifiability
squeeze_existence / squeeze_remainder / squeeze_flow
```

Default dtype is irrelevant (interval algebra). No new package.

## 7. Practical use cases

1. **Residual squeeze.** Drive a Helmholtz / model-problem residual
   enclosure to a width cap without claiming Navier–Stokes regularity.
2. **Peak localization.** First-class path through `certify_peak`
   (03-08), with `Inconclusive` on flat peaks.
3. **Identifiability.** `squeeze_identifiability` *is* the product
   wrapping `certify_trained_global_min` (`f_lower > 0`).
4. **Existence ball.** Kantorovich / Krawczyk / radii; empty ball is a
   reject, not a crash.
5. **Remainder as truncation.** Tag `R_N` as the `WidthBudget.truncation`
   piece (09-18).
6. **Validated flow.** Diagnose an existing Lohner `WidthBudget`; raise
   order, subdivide, shrink the step, or stop at the rounding floor.

## 8. Acceptance gates

- **G1 Width Law.** Predicted vs measured `w(r)` for Riccati `sigmoid` /
  `tanh` at shrinking `r`; the ratio approaches the predicted leading
  coefficient.
- **G2 Floor.** Lemma Floor witness; no API that clamps `lo` toward `hi`.
- **G3 Residual squeeze.** Helmholtz widths strictly decrease on
  `{1,4,16,64}` and stay above the rounding floor.
- **G4 Six callables.** Each `SqueezeKind` invokes `squeeze_*` and
  returns a `SqueezeReport` with the locked `scope` / honesty keys.
- **G5 Inconclusive / empty ball.** Flat peak and empty Kantorovich ball
  do not raise.
- **G6 Terminology + homes.** Three-row collapse table in canonical
  sources; this spec has a §2 home; no new package; retired penalty
  wordings stay retired.

## 9. Benchmark plan

- `benchmarks/enclosure_collapse.py`: Width Law relative error,
  Helmholtz decrease, six kinds callable.
- Smoke JSON: `docs/benchmarks/enclosure_collapse_smoke.json`.
- Full artifacts under `$OMNIBIAS_SCRATCH/enclosure_collapse/`.
- Package tests are the CI gate; no new CI job.

## 10. Honesty and scope

- Enclosure Collapse is **not** a derivative and **not** a 0/1 step.
  The founding bias collapse (`delta -> 0`) and temperature collapse
  (`beta -> inf`, feasibility) remain the other two limits. Do not
  conflate the three.
- `lo` and `hi` are not two biases. The integral window
  `S(z+b_hi)-S(z+b_lo)` is bias-geometry held finite, not this spec.
  Conformal slabs (04-02) are a different object.
- `continuum_navier_stokes_claim=False`. `interval_verified` only when
  the residual path is actually interval, not FFT evidence.
- Empty Kantorovich ball is a legal reject. `Inconclusive` is
  first-class.
- `theorem_prover_verified` / `mathlib_verified` default `False`. Lean
  only on finite rationals (`f_lower > 0`, `gap <= tol`, `p(r) < 0`,
  `error_bound <= threshold`). Never a continuum PDE.
- Scope fields: `local_box`, `parameter_box`, `finite_horizon`,
  `model_problem`.
- Group 08 does not host this third sense; the pointer lives here.

## 11. Open questions and risks

- **Wrapping vs true range.** A non-constant residual cannot be driven
  below `w_true` by splits alone. The Helmholtz witness is the clean
  case (`w_true = 0`).
- **Critical-point factor.** The 1-D `r^2` factor is locked in code and
  tests; a multivariate Hessian form would need a new factor.
- **Lean.** Finite rational obligations only. A Width Law *limit*
  statement is analytic and stays out of Lean.
- **Falsifier.** If predicted `w(r)` systematically fails to track
  measured mean-value width for Riccati activations as `r -> 0`, the
  Width Law implementation is wrong.

## 12. Implementation checklist

- [x] `packages/omnibias-core/src/omnibias/core/verified/enclosure_collapse.py`
- [x] `packages/omnibias-verify/src/omnibias/verify/enclosure_collapse.py`
- [x] Core + verify tests (grid and random soundness)
- [ ] No torch/jax Width Law twins
- [x] Docs page and mkdocs nav entry
- [x] Package tests as CI (no new job)
- [x] Sorted `__all__` on touched `__init__.py`
- [x] Index row in `theory/README.md`

---

## Repo invariants this spec must respect

- **Pure core**: no torch, jax, tensorflow or keras imports from
  `omnibias.core`.
- **Bit-identical twins**: not applicable to the Width Law (no twins).
  Remainder twins already share `omnibias.core.remainder_train`.
- **Default dtype**: framework default if a twin is ever called.
- **Vendor-neutral language**: artifacts go to `$OMNIBIAS_SCRATCH`.
- **Terminology**: founding bias collapse (`delta -> 0`) vs temperature
  collapse (`beta -> inf`) vs Enclosure Collapse (`width -> 0`).
- **Executable docs**: cookbook fences run; API page is mkdocstrings.
- **Earned flags**: `theorem_prover_verified` only on a genuine kernel
  pass.
- **Typing tier**: new modules `mypy --strict --follow-imports=silent`
  clean.
