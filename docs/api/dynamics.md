# omnibias-dynamics

Computer-assisted dynamics on the omnibias *validated* tower: rigorous
(interval / Taylor-model) tools for nonlinear ODEs, built on the QR-Lohner
validated flow, the Newton-Kantorovich / radii-polynomial existence machinery,
and the closed-form variational tower in `omnibias.core.verified`.

- **Validated variational / monodromy flow** — propagate a state *and* its
  fundamental matrix rigorously (the basis for Floquet multipliers / stability).
- **Poincare-section enclosures** — validated crossing locations on a hyperplane;
  the legacy crossing flag does not prove a first transversal return.
- **Certified Lyapunov-exponent bounds** — two-sided enclosures of the leading
  exponent from the validated variational flow.
- **Periodic-orbit existence** — a radii-polynomial proof that a *true* periodic
  orbit lives in an explicit ball around a numerical guess.
- **Cubic Abelian-integral count** — an exact Picard--Fuchs syzygy, validated
  complex winding upper count, and Krawczyk real-zero lower count can collapse
  to one instance-level integer. See
  [certified Abelian-integral zero count](abelian_zero_count.md).
- **Closed-form-tower jet bridge** — `vector_field_from_sigma_tower` and
  `sigma_oscillator_field` turn an activation's exact derivative tower into the
  `(VectorField, JacobianEnclosure)` pair the validated flow expects (no autodiff),
  and `discrete_periodic_point` proves fixed points / periodic orbits of iterated
  1-D maps via a Krawczyk certificate.
- **Cone-field hyperbolicity** — a finite orbit segment of interval Jacobians
  expands a cone with uniform factor `eta > 1`; `log(eta)` is a topological-entropy
  lower bound on that segment only. See [cone_field.md](cone_field.md).

!!! note "Soundness, not speed"
    Every enclosure provably contains the true object over the whole time
    interval, and an existence claim is a proof. Validated integration trades
    speed for certainty; a too-large step or too-chaotic a system makes the
    enclosure blow up, which is reported honestly rather than silently widened.

## Public API

::: omnibias.dynamics
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source

Status: Alpha (`0.1.0a1`).

## Source-bound stopped events

`omnibias.dynamics.return_maps` generates the flow and its first/mixed second
parameter variational equations from exact polynomial sources. A certified
event has a unique transverse first eligible positive-time hit, with earlier
and competing events excluded on the checked interval. Polynomial guards can
select a section branch. A singular or unresolved event remains unresolved.

```python
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.dynamics.return_maps import (
    PolynomialEvent, PolynomialFlow, StoppedEventRequest,
    certify_stopped_event, verify_stopped_event,
)

# Field variables: physical x, then the explicit clock.
x = SparsePolynomial.variable(2, 0)
request = StoppedEventRequest(
    flow=PolynomialFlow((SparsePolynomial.constant(2, 1),)),
    initial=(SparsePolynomial.constant(0, 0),),
    parameters=(),
    target=PolynomialEvent(x - 1, direction=1),
    step=0.125, max_steps=16, derivative_order=0,
)
event = certify_stopped_event(request)
assert event.certified and verify_stopped_event(event)
assert event.time_bracket is not None and event.time_bracket.contains(1.0)
```

The variable order is `(state, parameters, clock)`; initial embeddings depend
only on the parameters. Event derivatives include the implicit event-time
terms. `excluded` concerns only the checked finite horizon. It does not exclude
a later return or establish a global cycle bound.

## Exact displacement zero counts

`omnibias.dynamics.cyclicity` counts distinct isolated polynomial zeros,
including multiple endpoint roots. Identity fibers have no isolated zeros.
For parameter boxes it derives a uniform Rolle bound or uses the exact height
degree. `omnibias.dynamics.hilbert16` verifies binary covers of those boxes.

```python
from omnibias.dynamics.cyclicity import (
    ConfluentExponentialPolynomial, certify_exponential_cyclicity,
    certify_polynomial_cyclicity, verify_cyclicity_certificate,
)

h = SparsePolynomial.variable(1, 0)
displacement = (h + 1)**2 * h * (h - 1)**3
count = certify_polynomial_cyclicity(displacement, ((-1, 1),))
assert count.exact_count == 3
assert verify_cyclicity_certificate(displacement, count)

# Exact real exponential sum, including the double zero at kappa=0.
exponential = ConfluentExponentialPolynomial([(0, (1,)), (1, (-2,)), (2, (1,))])
assert certify_exponential_cyclicity(exponential).upper_bound == 2
```

The finite exponential class has a terminating weighted-derivative proof.
No membership of a general Dulac map is inferred. The physical consumer
`certify_planar_return_cyclicity` instead takes a replayed event result and
checks the exact autonomous planar section/initial-height identities before
using actual return derivatives. Its bound concerns that first-hit itinerary.

## Confluent scale primitives

```python
from omnibias.core.verified.asymptotic_jet import (
    power_compensator, signed_root_primitive, weighted_scale_derivative,
)

assert power_compensator(0, 0, 1).contains(0)
assert signed_root_primitive(0, 2).contains(2)
omega, u = (SparsePolynomial.variable(2, i) for i in range(2))
assert not weighted_scale_derivative(omega * u, order=2).terms
```

These compute sound confluent primitives and exact derivatives along
`(omega*exp(-s),u*exp(s))`. They do not prove uniform physical passage
remainders across the Hilbert XVI degeneration charts. The coalescing
χ-atlas in `packages/omnibias-dynamics/HILBERT16-COALESCING-CAPTURE.md`
records the frozen-exponent obstruction and a restricted first-derivative
bound. The tracked-product companion
[`HILBERT16-ENTRY-EXIT-LEADING.md`](../../packages/omnibias-dynamics/HILBERT16-ENTRY-EXIT-LEADING.md)
absorbs the super-small W-ratio in the first derivative. The fold I-map
companion
[`HILBERT16-FOLD-LEADING.md`](../../packages/omnibias-dynamics/HILBERT16-FOLD-LEADING.md)
replaces Gronwall `sigma kappa` at `sep = 0` by `dx/dkappa ~ r/kappa^2`.
The canonical-zeta companion
[`HILBERT16-CANONICAL-ZETA.md`](../../packages/omnibias-dynamics/HILBERT16-CANONICAL-ZETA.md)
seals a Cauchy majorant for `Z` on the `lambda=0` slow-line slice. The
fold-compact companion
[`HILBERT16-FOLD-ZETA.md`](../../packages/omnibias-dynamics/HILBERT16-FOLD-ZETA.md)
seals a Picard-plus-Cauchy majorant on a declared real compact of
`(L, lambda1)`. The frozen-Z C2 companion
[`HILBERT16-PHYSICAL-C2.md`](../../packages/omnibias-dynamics/HILBERT16-PHYSICAL-C2.md)
records exact remainder identities versus the lifted fold. The outgoing
x-corridor companion
[`HILBERT16-OUTGOING-CORRIDOR.md`](../../packages/omnibias-dynamics/HILBERT16-OUTGOING-CORRIDOR.md)
bounds slow time from `r1(1+theta)` to a compact `x_*`. The post-corridor
companion
[`HILBERT16-POST-CORRIDOR.md`](../../packages/omnibias-dynamics/HILBERT16-POST-CORRIDOR.md)
restores `T_*=Theta(eps^2)` and `(h/h_e)^{C eps}->1` independently of
`r1`. The height-envelope companion
[`HILBERT16-HEIGHT-ENVELOPE.md`](../../packages/omnibias-dynamics/HILBERT16-HEIGHT-ENVELOPE.md)
seals the `C=0` `T-h` comparison and a uniform `|q|` ratio as `r1->0`.
The C=2 `|q|` companion
[`HILBERT16-Q-RATIO-C2.md`](../../packages/omnibias-dynamics/HILBERT16-Q-RATIO-C2.md)
bounds the leading ratio for every `x` on `lambda1=-2`.
The `k`/cubic companion
[`HILBERT16-K-ZETA-REMAINDER.md`](../../packages/omnibias-dynamics/HILBERT16-K-ZETA-REMAINDER.md)
seals the `C=0` normal `k=1+O(nu)` jet and the cubic prefactor
`eps^4 x^3/3`.
The kill-compact `Z` companion
[`HILBERT16-KILL-ZETA.md`](../../packages/omnibias-dynamics/HILBERT16-KILL-ZETA.md)
seals a rectangular Cauchy majorant on `lambda1=-2`, `L in [0,1]`,
including `L=0`.
The cancelled-N companion
[`HILBERT16-CANCELLED-N.md`](../../packages/omnibias-dynamics/HILBERT16-CANCELLED-N.md)
seals a holomorphic `Z` bound with `2 eps |V| |Z| < 1` on a declared
slow-line compact.
The height-mix companion
[`HILBERT16-HEIGHT-MIX.md`](../../packages/omnibias-dynamics/HILBERT16-HEIGHT-MIX.md)
seals `C!=0` `ell`/`V` mixing with `|g_h|=O(nu^2)`.
The `T_h`-gap companion
[`HILBERT16-ORBIT-TH.md`](../../packages/omnibias-dynamics/HILBERT16-ORBIT-TH.md)
seals the actual-versus-comparison pointwise gap.
The `T-h`-integral companion
[`HILBERT16-TH-INTEGRAL.md`](../../packages/omnibias-dynamics/HILBERT16-TH-INTEGRAL.md)
seals the comparison-bootstrap majorant `< 9 eps`.
The cubic-orbit companion
[`HILBERT16-VH-ORBIT.md`](../../packages/omnibias-dynamics/HILBERT16-VH-ORBIT.md)
seals a QR-Lohner prefix and a certified `V=-1/4` first-hit.
The matching-chart companion
[`HILBERT16-E-OUT-SECTION.md`](../../packages/omnibias-dynamics/HILBERT16-E-OUT-SECTION.md)
seals first-hit of `E_out=V+rho+nu rho h+C nu^2 rho h^2`, the image of
`x=rho/nu` under `V=-eps x`, on `L in {9/25, 1/16, 0}` including the
kill limit; GRAZING `E_sigma` is excluded.
The shrinking-eps companion
[`HILBERT16-E-OUT-EPS.md`](../../packages/omnibias-dynamics/HILBERT16-E-OUT-EPS.md)
seals first-hit on `n in {16,20,25}` at `L=0` inside `T=n^2/8`.
The comparison-speed companion
[`HILBERT16-E-OUT-SPEED.md`](../../packages/omnibias-dynamics/HILBERT16-E-OUT-SPEED.md)
seals an `O(1/eps^3)` hitting-time majorant on the kill line.
The incoming GRAZING comparison-speed companion
[`HILBERT16-E-SIGMA-SPEED.md`](../../packages/omnibias-dynamics/HILBERT16-E-SIGMA-SPEED.md)
seals an `O(1/eps^3)` reverse-time majorant on `V in [0, 1]`; Lohner wrapping
refuses a certified `E_sigma` first-hit.
The incoming-wall companion
[`HILBERT16-E-SIGMA-IN.md`](../../packages/omnibias-dynamics/HILBERT16-E-SIGMA-IN.md)
seals first-hit of `V=1/4` on the reverse cubic from `V=0`, `h=4 eps^3`.
The declared-point companion
[`HILBERT16-E-SIGMA-HIT.md`](../../packages/omnibias-dynamics/HILBERT16-E-SIGMA-HIT.md)
seals first-hit of GRAZING `E_sigma` from `(V,h)=(3/4,1/4)`; the GRAZING
start `V=0` still excludes `E_sigma` on that compact Lohner horizon.
The comparison companion
[`HILBERT16-E-SIGMA-FROM0.md`](../../packages/omnibias-dynamics/HILBERT16-E-SIGMA-FROM0.md)
isolates a unique increasing `E_sigma` zero from `V=0` on a height
majorant tube; Lohner wrapping still refuses `certify_stopped_event`
from `V=0`.
The uniform companion
[`HILBERT16-E-SIGMA-UNIF.md`](../../packages/omnibias-dynamics/HILBERT16-E-SIGMA-UNIF.md)
keeps `E_sigma>0` on eight Interval slabs covering `eps in [0, 1/8]`.
The orbit-aligned companion
[`HILBERT16-E-SIGMA-WALL.md`](../../packages/omnibias-dynamics/HILBERT16-E-SIGMA-WALL.md)
hits GRAZING `E_sigma` from `(1/4,1/40)` inside the certified `V=1/4`
return box.
The wall-box companion
[`HILBERT16-E-SIGMA-BOX.md`](../../packages/omnibias-dynamics/HILBERT16-E-SIGMA-BOX.md)
covers `[1/50, 4/125]` by twelve `h`-slabs at `V=1/4`.
The L=0 whole-wall companion
[`HILBERT16-E-SIGMA-SPAN.md`](../../packages/omnibias-dynamics/HILBERT16-E-SIGMA-SPAN.md)
covers `[19/1000, 1/25]` by twenty-one `h`-slabs containing the
`L=0` wall box.
The L-pack wall-span companion
[`HILBERT16-E-SIGMA-PACK.md`](../../packages/omnibias-dynamics/HILBERT16-E-SIGMA-PACK.md)
covers `[17/1000, 7/200]` by eighteen `h`-slabs on `L in {9/25, 1/16}`.
The shrinking-eps aligned companion
[`HILBERT16-E-SIGMA-EPS.md`](../../packages/omnibias-dynamics/HILBERT16-E-SIGMA-EPS.md)
certifies `(1/4,1/40)` at `n in {16, 20, 25}`.
The one-shot companion
[`HILBERT16-E-SIGMA-ONESHOT.md`](../../packages/omnibias-dynamics/HILBERT16-E-SIGMA-ONESHOT.md)
hits GRAZING `E_sigma` from `V=0` in one run at `eps=1/16`.
The shrinking-eps one-shot companion
[`HILBERT16-E-SIGMA-ONESHOT-EPS.md`](../../packages/omnibias-dynamics/HILBERT16-E-SIGMA-ONESHOT-EPS.md)
repeats that one-shot at `n in {16, 20, 25}`.
G1 and G4 remain failed (`Z_x` / `sep>0` physical C2,
a uniform Lohner first-hit for every `eps` on chart O).
