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
bound. The saddle-node, shrinking-root, two-blow-up, and next-atlas /
scale-dichotomy companions fail on the same named sequences; G1 and G4
remain failed.
