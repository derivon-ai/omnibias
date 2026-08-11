# 07-02 Navier-Stokes adjacent: tighter finite enclosures

## 1. Thesis and status

omnibias already certifies finite residual enclosures for incompressible flow;
three of the new primitives make those enclosures **tighter and cheaper** —
closed-form test functions remove a quadrature error term, multi-pack bases
represent vortex sheets without smearing, and exact Jacobians remove
differentiation error from the validated integrator. None of this touches global
regularity.

- **Status**: designed
- **Depends on**: 01-01, 01-05, 01-09, 01-12, 02-04, 02-05, 02-12, 02-13, 03-06, 03-08, 03-10, 07-01
- **Blocks**: none

## 2. Where it lands

`omnibias.pinn.certified` — the existing home, extended. Sixteen modules already
live there including `navier_stokes.py`, `fluid_rigorous.py`,
`viscous_perturbation.py`, `euler2d.py`, `sqg.py`, `ipm.py`, `boussinesq.py`.
Adding a package would fragment a coherent certified-fluids surface.

## 3. Prior art in omnibias

Substantial, and it constrains what "new" means here.

- `omnibias.pinn.certified.navier_stokes` — `HonestyLabels`,
  `prove_navier_stokes_periodic_residual`,
  `navier_stokes_periodic_residual_schema_errors`, and validators at three
  certificate sites that each append an error unless
  `continuum_navier_stokes_claim` is `False`.
- `omnibias.pinn.certified.machine` — the proof machine; blocks the verdict if
  `honesty.continuum_navier_stokes_claim` is `True`.
- `omnibias.pinn.certified.fluid_rigorous` — `StreamfunctionField`,
  `certified_streamfunction_residual`,
  `certified_shear_streamfunction_residual`, `prove_streamfunction_residual`,
  `replay_streamfunction_residual`, plus shear and cellular streamfunctions.
- `omnibias.pinn.certified.viscous_perturbation` —
  `viscous_perturbation_enclosure` with `continuum_navier_stokes_claim: False`.
- `omnibias.pinn.certified.{euler2d,sqg,ipm,boussinesq,dissipation_threshold,fluid_rollout}`.
- `omnibias.core.verified.{lohner,kantorovich,sqg,pde_certificate}`.
- `benchmarks/ipm_boussinesq_scaffold_smoke.py` with scaffold floors of `2.0`.

**Confirmed gap**, narrowly: the residual enclosures are **strong-form** and
pointwise, so they need the solution's second derivatives and a quadrature bound
over the domain. There is no weak-form certified residual, no interface-adapted
basis, and the validated flow uses numerically differentiated Jacobians in
places where the tower could supply exact ones.

## 4. Mathematics

### Where the width of a residual enclosure comes from

For the periodic incompressible problem, a strong-form residual enclosure at
tolerance `eps` accumulates width from four sources:

```
W_total  <=  W_repr  +  W_quad  +  W_deriv  +  W_round
```

- `W_repr`: how well the ansatz represents the true field.
- `W_quad`: the bound on the residual *between* sample points — for a strong
  form this requires a Lipschitz or Taylor-model bound on the residual field
  over each cell, which grows with the residual's derivative order.
- `W_deriv`: error in evaluating derivatives of the ansatz. **Zero** for the
  activation tower, non-zero wherever a numerical derivative is used.
- `W_round`: outward rounding, irreducible and small.

The three primitives attack `W_quad`, `W_repr` and `W_deriv` respectively.

### (a) Weak form removes derivatives from the residual

The strong residual for momentum needs `u_xx`. Testing against `v` and
integrating by parts moves one derivative onto `v`:

```
int ( u_t + u . grad u + grad p - nu lap u ) v
  = int ( u_t + u . grad u ) v  -  int p div v  +  nu int grad u : grad v
```

with boundary terms vanishing on the periodic box. The residual functional now
needs only `grad u`, one order lower.

Why that matters for the *enclosure* rather than merely for training: `W_quad`
is bounded using a Taylor model of the residual field, whose width grows with
the order of derivatives it contains. Dropping one order typically reduces the
Taylor-model remainder by a factor of the cell diameter. And because spec 02-04
supplies test functions with **closed-form antiderivatives**, the integrals
above are evaluated exactly rather than by quadrature, removing the quadrature
term from `W_quad` entirely rather than bounding it.

That is a genuine structural improvement to the enclosure, not a training trick.

### (b) Multi-pack bases for shear layers

`fluid_rigorous.shear_streamfunction` already handles a shear profile. A
vortex sheet or a thin shear layer is a near-discontinuity in a derivative, and
a smooth global ansatz represents it with `W_repr` proportional to the layer's
inverse thickness. A heterogeneous multi-pack (spec 01-01) placed at the layer
supplies exactly the jet coordinates the transmission conditions need at exactly
that location (spec 02-05), so `W_repr` becomes independent of the layer
thickness in the direction across it.

### (c) Exact Jacobians in the validated flow

`omnibias.core.verified.lohner`'s QR-Lohner step propagates an enclosure through
the variational equation, which needs the Jacobian of the vector field. Supplied
numerically, the Jacobian carries its own error into `W_deriv` and, worse, into
the wrapping-effect growth over many steps. Supplied exactly by the tower, that
contribution vanishes and only rounding remains.

### What none of this does

It shortens the interval. It does not extend the time horizon to infinity, does
not quantify over initial data, and does not survive the continuum limit. The
sealed scope stays exactly what `HonestyLabels` already pins.

## 5. Worked example

**Counting the width improvement on a 2D Taylor-Green problem.**

Take the periodic box `[0, 2 pi]^2`, viscosity `nu = 0.1`, the Taylor-Green
initial condition, horizon `T = 0.5`, and a certified residual over a
`16 x 16` cell decomposition, so cell diameter `h = 2 pi / 16 = 0.3927`.

*Strong form.* The residual field contains `lap u`, so bounding it between
sample points uses a Taylor model whose remainder scales as
`C_3 h^2 / 8` with `C_3` a bound on third derivatives of `u`. For Taylor-Green
at this viscosity, `|u| <= 1` and derivatives up to third order are bounded by
roughly `1`, giving

```
W_quad(strong) ~ 1 * 0.3927^2 / 8 = 0.0193
```

*Weak form.* The residual functional contains only `grad u`, so the Taylor-model
remainder involves `C_2` and one fewer power is available to hide in — but the
key change is that the test-function integrals are exact, so the quadrature
contribution disappears and only the representation of `u` inside the cell
remains. The remainder scales as `C_2 h^2 / 8` with the *same* `h^2` but a
smaller constant, and the quadrature term that would otherwise sit alongside it
(a Gauss rule's own error, typically comparable in size at this cell count) is
gone.

Estimating conservatively at a factor of `2` from the constant and a factor of
`2` from removing quadrature:

```
W_quad(weak) ~ 0.0048
```

a `4x` narrower quadrature contribution at identical cost, because the exact
antiderivative replaces a quadrature rule rather than adding to it.

*The honest caveat, which is the instructive part.* This estimate assumes
`W_quad` dominates. If `W_repr` dominates — which it will whenever the ansatz is
the weak link, and for a hard flow it usually is — then a `4x` improvement in
`W_quad` moves `W_total` by almost nothing. So the first measurement in the
benchmark must be a **width decomposition**: report `W_repr`, `W_quad`,
`W_deriv` and `W_round` separately before claiming an improvement in any of
them. A method that improves a non-dominant term and reports only the total is
indistinguishable from one that does nothing.

*Exact Jacobians in the flow.* Over `n` QR-Lohner steps the wrapping effect
amplifies per-step error roughly geometrically with the local Lyapunov exponent.
At `T = 0.5` with an exponent near `1` and `n = 50` steps, removing a per-step
Jacobian error of `1e-10` in favour of exact evaluation reduces the accumulated
contribution by that same factor times the amplification — significant only if
the Jacobian error was near the dominant term, which the decomposition will
again reveal.

## 6. Proposed API

Additions to the existing certified surface.

```python
# omnibias/pinn/certified/weak_form.py
def certified_weak_residual(
    field, test_basis, *, cells, honesty: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Weak-form certified residual with exact test-function integrals.
    Emits the same certificate schema as certified_streamfunction_residual and
    is subject to the same continuum_navier_stokes_claim=False validator."""

def width_decomposition(cert: dict[str, Any]) -> WidthReport:
    """Splits the enclosure width into repr / quad / deriv / round.
    Required before any width-improvement claim."""

@dataclass(frozen=True)
class WidthReport:
    w_repr: float
    w_quad: float
    w_deriv: float
    w_round: float
    dominant: str
```

`WidthReport.dominant` exists so that a benchmark cannot report an improvement
in a non-dominant term without the fact being visible in the artifact.

## 7. Practical use cases

1. **Narrower published enclosures** for the flows already certified, at equal
   or lower cost.
2. **Shear and vortex-sheet problems** currently out of reach because a smooth
   ansatz cannot represent the layer within a useful width.
3. **Longer validated horizons** where the Jacobian error was limiting the step
   count.
4. **Honest reporting of where the width comes from**, which is useful
   independently of any improvement and is currently unavailable.

## 8. Acceptance gates

Baselines: the existing `certified_streamfunction_residual` and
`prove_navier_stokes_periodic_residual` outputs, at matched configuration.

- **G1 width decomposition first.** Every certified run emits a `WidthReport`.
  No width claim is accepted without one. This gate precedes the others
  deliberately.
- **G2 weak-form narrower.** On the Taylor-Green configuration, the weak-form
  enclosure is at least `2x` narrower **in its dominant term** than the strong
  form at matched cost, or the result is reported as no improvement.
- **G3 soundness preserved.** `100%` enclosure coverage against high-precision
  references over `1000` synthetic configurations. A single miss is a bug.
- **G4 shear representation.** For a shear layer of thickness `d`, the multi-pack
  ansatz's `W_repr` is independent of `d` over `d in [1e-3, 1e-1]`, while the
  smooth baseline's grows at least as `1/d`.
- **G5 exact-Jacobian horizon.** The validated flow reaches at least `1.5x` the
  time horizon at fixed enclosure width, versus numerically differentiated
  Jacobians.
- **G6 honesty flags intact.** Every new certificate passes the existing
  validators, with `continuum_navier_stokes_claim`, `three_d_claim` and
  `unproven_claim` all `False`. Asserted by test against the existing schema
  checkers.

## 9. Benchmark plan

- `benchmarks/ns_weak_form_enclosure.py`: width decomposition, weak-versus-strong
  comparison, shear-thickness sweep, horizon comparison. Smoke JSON in
  `docs/benchmarks/`, full under `$OMNIBIAS_SCRATCH/ns_adjacent/`.
- Reuse `benchmarks/_gates.py` including the new `require_enclosure_coverage`
  from spec 06-01.

## 10. Honesty and scope

- **Nothing here approaches global regularity.** Every result is one
  discretization, one box, one horizon, finite dimension.
- The existing flags stay pinned: `continuum_navier_stokes_claim = False`,
  enforced by validators that already exist and must not be relaxed to
  accommodate a new certificate shape.
- "Tighter enclosure" is a claim about **width**, not about correctness. A wide
  sound enclosure and a narrow sound enclosure are equally true; the narrow one
  is more useful.
- The founding `delta -> 0` bias collapse supplies the multi-pack basis and the
  exact derivatives. No temperature collapse appears.
- Certificate tier: **sound enclosure**, with the positivity or sign obligations
  eligible for the Lean kernel exactly as today. `theorem_prover_verified` is
  earned by the kernel, never asserted.

## 11. Open questions and risks

- **`W_repr` probably dominates.** This is the main risk and G1 exists to expose
  it. If representation error dominates everywhere that matters, the weak-form
  contribution is real but small, and the spec should say so.
- **Weak form needs a test-space argument.** A residual small against *some*
  test functions is not small; the enclosure must account for the test space's
  completeness, and a naive implementation would produce a narrower interval
  that means less. This is the most likely soundness bug and needs its own test.
- **Multi-pack conditioning.** Placing packs at a thin layer can produce
  ill-conditioned bases; the conditioning must be reported.
- **Lohner wrapping.** Exact Jacobians reduce one error source but do not
  address the wrapping effect, which is usually the binding constraint on
  horizon. G5 may fail for that reason and the failure would be informative.
- **Falsifier.** If G1's decomposition shows `W_quad` is negligible in every
  configuration of interest, the weak-form contribution to *certification* is
  not worth building, though it may still help training (spec 02-04).

## 12. Implementation checklist

- [ ] `packages/omnibias-pinn/src/omnibias/pinn/certified/weak_form.py`
- [ ] `WidthReport` and `width_decomposition`, emitted by every certified run
- [ ] Test-space completeness accounted for in the weak enclosure, with a
      dedicated soundness test
- [ ] Multi-pack shear basis reusing spec 01-01 and 02-05
- [ ] Exact tower Jacobians wired into `omnibias.core.verified.lohner` calls
- [ ] Conditioning reported for interface-placed packs
- [ ] `1000`-configuration coverage test
- [ ] Existing schema validators run against the new certificate shape
- [ ] `benchmarks/ns_weak_form_enclosure.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: existence and smoothness for the 3D incompressible Navier-Stokes
equations (Clay Millennium Problem).**

It stays external for a structural reason that no improvement in this spec
touches. The parent asserts that for **every** smooth divergence-free initial
datum of finite energy, a smooth solution exists for **all** time, in the
**continuum**. Everything produced here is a statement about one
finite-dimensional discretization, one fixed initial condition, and one finite
time horizon. The three quantifiers that make the problem hard — all data, all
time, the continuum limit — are precisely the three this machinery does not
range over, and narrowing an interval does not begin to range over them.

`omnibias.pinn.certified` already encodes this: `continuum_navier_stokes_claim`
is pinned to `False` at three validator sites and blocks the proof machine's
verdict if asserted. This spec does not claim, imply, or provide evidence for
global regularity, and a certificate produced by it must never be described as
doing so.
