# 02-04 Weak-form VPINN with closed-form test functions

## 1. Thesis and status

A Petrov-Galerkin PINN whose **test functions are OMBU bumps with closed-form
antiderivatives**: integration by parts becomes exact instead of
quadrature-approximated, the residual needs one fewer derivative of the
solution, and problems with weak or discontinuous data become well posed.

- **Status**: gated (exact on polynomial boxes; boundary bound on by default)
- **Depends on**: 01-01, 01-05
- **Blocks**: 02-05, 03-06, 07-02

## 2. Where it lands

`packages/omnibias-fields/src/omnibias/fields/weak/` with torch and jax twins,
because the field substrate owns operators and quadrature. `omnibias-pinn`
consumes it; `omnibias-variational` may share the assembly.

## 3. Prior art in omnibias

- `packages/omnibias-fields/src/omnibias/fields/_core/quadrature.py` —
  `QuadratureSpec`, the existing quadrature description used by the `integrate`
  and `inner_product` ops.
- `omnibias.fields` ops — `value`, `derivative`, `gradient`, `laplacian`,
  `divergence`, `curl`, `hessian`, `jacobian`, `integrate`, `inner_product`,
  Sobolev norms.
- `docs/operator-surface.md` — the `integral` role: `S(z + b_hi) - S(z + b_lo)`
  with `S' = sigma`. omnibias already has closed-form antiderivatives.
- `omnibias-variational` — action integrals, Euler-Lagrange residuals, the
  arbitrary-order Euler-Poisson functional derivative, and rigorous action
  enclosures. This is the closest existing thing to a weak formulation, but it
  is a *variational* (energy) formulation, not a Petrov-Galerkin one with
  independent test functions.

**Confirmed gap.** There is no weak-form or test-function surface: no
`TestFunctionSpace`, no assembly of `integral(residual * v)`, no
integration-by-parts machinery. Residuals today are strong-form pointwise.

## 4. Mathematics

### Strong versus weak

For `L u = f` on `Omega`, the strong residual is `R(x) = L u(x) - f(x)` and a
PINN minimizes `||R||^2` at collocation points. The weak residual against a test
function `v` is

```
r(v) = integral_Omega ( L u - f ) v  dx
```

and after integration by parts (for a second-order `L = -div(a grad .)`),

```
r(v) = integral_Omega a grad u . grad v  dx  -  integral_{dOmega} a (grad u . n) v  ds
       - integral_Omega f v  dx
```

The middle term vanishes (or is prescribed) if `v` is chosen appropriately.
Three consequences:

1. **One fewer derivative of `u`.** The weak form needs `grad u`, not
   `laplacian u`. For a network this halves the jet order required and removes
   the worst-conditioned term.
2. **Weak data admitted.** Discontinuous coefficients `a`, point sources, and
   non-smooth solutions are meaningful in the weak form and meaningless in the
   strong one.
3. **Test-function choice becomes a design variable.** Petrov-Galerkin: trial
   space (the network) and test space (the bumps) are independent.

### Why closed-form antiderivatives matter

Assembling `r(v)` requires integrals. Standard VPINNs use Gauss quadrature, so
the weak residual carries a quadrature error that can exceed the discretization
error, especially for oscillatory or sharply peaked `v`.

With an OMBU test function built from the `integral` role, the needed integrals
are **differences of the closed-form antiderivative at the window edges**. For a
1-D window `[p, q]` and the order-`n` bump `v = sigma^(n)(z + mu)`:

```
integral_p^q v  =  [ sigma^(n-1)(z + mu) ]_p^q      exactly
integral_p^q v' =  [ sigma^(n)(z + mu) ]_p^q        exactly
```

and in general every moment integral of a pack against a polynomial reduces to
antiderivative differences plus lower-order terms by repeated integration by
parts, all closed form. So the assembly of `r(v)` for polynomial coefficient
data is **exact** rather than quadrature-approximated.

For non-polynomial coefficients, quadrature returns, but only on the coefficient
factor, and the test function's contribution stays exact. That is still a strict
improvement, and it is the honest general claim.

### Boundary terms

From spec 01-05: analytic bumps are not compactly supported, so the boundary
term is not exactly zero. It is bounded by a certified exponentially small
quantity via `tail_bound`. The assembly must **add that bound to the reported
residual**, not silently drop it. This is the difference between a weak form
that is honest and one that quietly cheats.

### Test-space design

A test space is a bank (spec 01-02) of bumps at offsets `tau_j`, orders `n_j`,
and scales `alpha_j`. Design choices:

- **Number** of test functions sets the number of equations; a square or
  overdetermined system is preferable.
- **Order** sets how many derivatives are moved off `u` and, by spec 01-07,
  which frequency band the test function probes.
- **Overlap** controls conditioning of the discrete system.

Spec 01-07 turns the last two into a calculation: choose test functions so their
bands cover the residual spectrum.

### Multi-dimensional case

Tensor-product bumps `v(x) = prod_d v_d(w_d . x + mu_d)` keep the exact-integral
property on boxes. On general domains, combine with the SDF machinery in
`omnibias.pinn.domain` to restrict the window, at the cost of returning to
quadrature near the boundary. State that cost.

## 5. Worked example

Problem: `-u'' = f` on `[0, 1]`, `u(0) = u(1) = 0`, with
`f(x) = pi^2 sin(pi x)`, exact solution `u(x) = sin(pi x)`.

Test function: an order-2 bump centred at `mu = 0.5` with scale `alpha = 10`,

```
v(x) = sigma''( 10 (x - 0.5) ),     sigma = tanh
```

Weak residual after one integration by parts:

```
r(v) = integral_0^1 u'(x) v'(x) dx  -  integral_0^1 f(x) v(x) dx  -  [ u' v ]_0^1
```

The boundary term: `v(0) = sigma''(-5)` and `v(1) = sigma''(5)`.

```
tanh(5) = 0.9999092,   1 - t^2 = 1.8158e-4
sigma''(5) = -2 * 0.9999092 * 1.8158e-4 = -3.6313e-4
```

so `|v|` at the boundary is `3.63e-4`, not zero. With `|u'| <= pi` the boundary
term is bounded by `2 * pi * 3.6313e-4 = 2.2815e-3`. Increasing the scale to
`alpha = 30` shrinks it: `tanh(15) = 1 - 2.06e-13`, `sigma''(15) = -4.1e-13`, so
the bound falls to `2.6e-12`. **The tail bound is the design constraint on
`alpha`**, and it is computable in advance.

The exact-integral property: since `sigma''` is odd,

```
integral_0^1 v'(x) dx = (1/10) [ sigma''(10(x - 0.5)) ]_0^1
                      = (1/10) ( sigma''(5) - sigma''(-5) )
                      = (1/10) ( -3.6311e-4 - 3.6311e-4 ) = -7.2622e-5
```

two activation evaluations, no quadrature, no error. The competing Gauss rule
must resolve a function whose features live on a width of `1/alpha = 0.1`; its
error is a strong function of node count and of `alpha`, and the benchmark's job
is to measure exactly where that error crosses the discretization error of a
trained network. The argument for this spec is that the crossover exists and is
reached at realistic `alpha`; if the benchmark shows it is not, G1 will say so.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/fields/weak/_core.py
@dataclass(frozen=True)
class TestFunctionSpace:
    bank: BankSpec                 # offsets and scales, spec 01-02
    orders: tuple[int, ...]
    base: str = "tanh"
    window: tuple[float, float] | None = None
    @property
    def size(self) -> int: ...

def exact_moment(space: TestFunctionSpace, j: int, index: int) -> float:
    """integral x^j v_index dx, closed form via antiderivative differences."""
def boundary_bound(space: TestFunctionSpace, *, deriv_bound: float) -> Interval:
    """Certified bound on the dropped boundary term."""
```

```python
# omnibias/fields/weak/torch.py  (and jax twin)
def weak_residual(
    field, space: TestFunctionSpace, *, operator: WeakForm, source,
    quadrature: QuadratureSpec | None = None,
) -> Tensor:
    """Assemble r(v_i) for every test function. Uses exact integrals where the
    coefficient data is polynomial; falls back to `quadrature` otherwise and
    records which path was taken."""

def weak_loss(field, space, *, operator, source, include_boundary_bound=True) -> Tensor:
    """Sum of squared weak residuals plus, if requested, the certified boundary
    bound so the reported number is never optimistic."""
```

The `include_boundary_bound=True` default is deliberate: the honest number
should be the easy one to compute.

## 7. Practical use cases

1. **Discontinuous coefficients.** Layered media, composite materials: the
   strong form is undefined at the interface, the weak form is not.
2. **Point sources and line loads.** Delta-type data is legitimate against a
   test function and meaningless pointwise.
3. **Fourth-order problems.** Two integrations by parts move two derivatives off
   `u`, so a biharmonic problem needs only second derivatives of the network.
4. **Better conditioning.** Strong-form PINNs are notoriously badly conditioned
   because they differentiate the network twice; the weak form differentiates
   once.
5. **Certified residual reporting** for the frontier track (spec 07-02): a weak
   residual with a certified boundary bound is a sounder object than a pointwise
   residual at sampled collocation points.

## 8. Acceptance gates

Baselines: a strong-form PINN on the same field, and a standard VPINN with
Gauss quadrature test functions, both at matched parameter count and matched
wall time.

- **G1 exact-integral verification.** For polynomial coefficient data, assembled
  integrals match high-precision references to `<= 1e-13` relative, while the
  Gauss-quadrature path shows its expected finite error.
- **G2 boundary honesty.** The reported residual always includes the boundary
  bound; a test asserts that disabling it changes the number, so the default is
  not vacuous.
- **G3 accuracy.** On a problem with a discontinuous coefficient and a known
  exact solution, relative `L2` error `<= 1e-6` with skill `> 0`, beating the
  strong-form baseline, over five seeds.
- **G4 conditioning.** The measured condition number of the discrete system is
  at least `10x` lower than the strong-form collocation system on the same
  problem.
- **G5 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/weak_form_vpinn.py` with three arms (strong, Gauss VPINN, exact
  VPINN) on three problems (smooth, discontinuous coefficient, point source).
- Records which integration path was taken per assembly, so the exact-versus-
  quadrature fraction is auditable.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/weakform/`.

## 10. Honesty and scope

- Test functions come from the founding bias collapse (`delta -> 0`). No
  temperature collapse appears.
- **Integrals are exact only for polynomial coefficient data on box windows.**
  Everything else falls back to quadrature on the coefficient factor, and the
  implementation must record which path ran rather than advertising exactness
  generally.
- **Boundary terms are not zero.** They are certified small. Any statement that
  omits the bound is an overclaim, which is why it is on by default.
- The weak residual being small does not prove the solution is accurate; it
  proves the residual is small against *this* finite test space. A larger test
  space can reveal error the smaller one missed. Report the test-space size with
  every result.
- Certificate tier: sound enclosure for the boundary bound; the rest is
  empirical.

## 11. Open questions and risks

- **Test-space conditioning.** Overlapping bumps at similar scales produce a
  nearly singular system. Report the condition number; consider orthogonalizing.
- **Cost.** Each test function is an integral; a large test space is expensive.
  Measure the accuracy-per-cost curve against the strong form honestly.
- **General domains.** The exactness argument is cleanest on boxes. Curved
  domains via SDF windows reintroduce quadrature; quantify how much is lost.
- **Falsifier.** If the Gauss VPINN matches the exact VPINN at equal cost on all
  three benchmark problems, the closed-form-integral advantage is not material
  and the spec reduces to "VPINNs are good", which is not news.

## 12. Implementation checklist

- [ ] `packages/omnibias-fields/src/omnibias/fields/weak/_core.py`
- [ ] torch and jax twins with a parity test
- [ ] Reuse `QuadratureSpec` for the fallback path; do not fork it
- [ ] Exact-integral test versus high-precision references
- [ ] Boundary-bound soundness test and a test that the default is not vacuous
- [ ] Path-recording test: assembly reports exact versus quadrature per term
- [ ] Condition-number comparison test
- [ ] `benchmarks/weak_form_vpinn.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Regenerate `__all__` in `omnibias/fields/__init__.py`
- [ ] Index row in `theory/README.md`
