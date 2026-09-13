# Source-bound finite-time return maps

`omnibias.dynamics.return_maps` certifies a **regular first positive event**
for every parameter in a finite box. It derives interval flow, event
transversality, earlier-event exclusion, and first/mixed second derivatives
from exact polynomial sources. The result is a finite-time numerical
certificate with a written analytic justification below. This is not a
singular-passage theorem, a complete graphic cover, or a Hilbert XVI solution.
No novelty claim is made.

## Exact source and event semantics

The source is `omnibias.core.realization.polynomial.SparsePolynomial`, which
stores rational coefficients and derives its polynomial derivatives exactly.
No field, Jacobian, event, or derivative callback is accepted. A
`PolynomialFlow` with physical state dimension `n` and parameter dimension
`m` takes `n` component polynomials in the variables
`(y[0:n], theta[0:m], t)`, including the final clock coordinate even for an
autonomous field. The initial embedding consists of `n` polynomials in the
`m` parameters. Thus an initial-section coordinate and a field parameter
can participate in the same mixed derivative.

A `PolynomialEvent` specifies `g(y, theta, t)=0`, optionally restricted by
the strict polynomial inequality `guard(y, theta, t)>0`. Without a guard,
every section zero is eligible. A target direction of `+1` or `-1` requires
that sign of the actual total time derivative of `g` at the first eligible
event; direction `0` accepts either strict sign. An earlier eligible event
in the opposite direction blocks the request. Competitors stop on any
eligible zero, independently of their direction setting.

The source can lie exactly on the target section. The engine proves this
identity by exact polynomial substitution and then requires a signed
departure slab. It excludes the initial time, not a caller-selected block of
unchecked steps. A section with an uncertain initial incidence or a singular
departure is unresolved. Initial competing events must also be excluded.

The returned `request` retains the exact sources, parameter box, finite
integration budget, and derivative order. `source_fingerprint` binds those
inputs. `verify_stopped_event(result)` replays the full computation and compares
all output bounds. A status string or a modified result is not an independently
checkable proof without that replay. No Lean verification flag is assigned.

## Finite-time theorem used by a successful result

Write `w=(y,theta,t)` and generate the extended polynomial field
`V=(F,0,1)`. On each step the interval Taylor integrator constructs a Picard
enclosure of the complete extended trajectory slab and a Taylor endpoint
enclosure. The clock endpoints use the exact rational value of the binary
floating-point step times the integer step index; rounded accumulation does
not create gaps between slabs. Frozen parameter and clock identities tighten
the generated enclosures.

For each event, the exact polynomial Lie derivative is

\[
v_g(w)=Dg(w)V(w).
\]

The slab record encloses `g` at both endpoints, throughout the slab, and
`v_g` throughout the slab. Integrating the last enclosure gives an additional
sound enclosure of `g`, intersected with direct polynomial evaluation.
Earlier event zeros are excluded by one of the following proved conditions:

1. the event's guard is nonpositive throughout the slab;
2. the event value excludes zero throughout the slab;
3. its total time derivative has one strict sign and its two endpoint values
   have the same strict sign;
4. for the initial target identity only, a strictly signed departure excludes
   all positive-time zeros in that first slab.

When a target event becomes possible, the engine requires opposite strict
endpoint signs, a single strict sign for `v_g` on the whole bracket, and a
strictly positive target guard there. The bracket may span consecutive slabs
if a sampled endpoint intersects zero. The intermediate-value theorem gives
one event for each source parameter; strict monotonicity gives uniqueness.
The earlier exclusions make it the first eligible positive event. Every
competing event must be excluded on the entire bracket as well as all earlier
slabs. This is conservative when a competitor occurs just after the target
inside the same slab; a finer step may resolve that case.

The algorithm never jumps across an undecided slab. A tangency, cubic
nontransverse crossing, hidden pair of earlier zeros, ambiguous guard, or
failed flow enclosure returns `unresolved`. `excluded` means no eligible
positive target zero on the **checked finite horizon**. Neither outcome is
a global absence-of-cycle theorem.

## Actual event derivatives

The generated variational equations propagate, at fixed time,

\[
S_a=\partial_{\theta_a}w,\qquad
Q_{ab}=\partial_{\theta_a\theta_b}w,
\]

with initial derivatives obtained from the exact initial embedding. If
`A=DV` and `B=D²V`, their equations are

\[
\dot S_a=AS_a,\qquad
\dot Q_{ab}=AQ_{ab}+B[S_a,S_b].
\]

These equations are integrated together with the state. They do not come
from `poincare_map_jet`, whose output is an enclosure-width budget.

For `E(t,theta)=g(w(t,theta))` and its unique event time `tau(theta)`, the
strict denominator `E_t=v_g` permits the implicit-function formulas

\[
\tau_a=-E_a/E_t,\qquad
\tau_{ab}=-\frac{E_{ab}+E_{ta}\tau_b+E_{tb}\tau_a
                         +E_{tt}\tau_a\tau_b}{E_t}.
\]

Here `E_a=Dg S_a`, `E_ab=Dg Q_ab+D²g[S_a,S_b]`,
`E_ta=D(v_g) S_a`, and `E_tt=D(v_g) V`. For the returned physical state
`R(theta)=y(tau(theta),theta)`, coordinatewise,

\[
R_a=S_a+V\tau_a,
\]

\[
R_{ab}=Q_{ab}+(AS_a)\tau_b+(AS_b)\tau_a
                   +(AV)\tau_a\tau_b+V\tau_{ab}.
\]

Only physical state components of the last two formulas are returned.
All factors are enclosed on the event bracket. Coordinate interval Newton
can further contract the event state without discarding any section zero.
`return_jacobian` uses `[state][parameter]` indices and `return_hessian` uses
`[state][parameter_a][parameter_b]`. A requested derivative order of zero or
one leaves the unavailable higher outputs as `None`.

## Executable guarded-return example

The guard matters: the oscillator hits the whole line `x=0` after half a
period, but returns to `x=0, y>0` after one full period.

```python
import math
from omnibias.core.realization.polynomial import SparsePolynomial as P
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.return_maps import (
    PolynomialEvent, PolynomialFlow, StoppedEventRequest,
    certify_stopped_event, verify_stopped_event,
)

x, y = P.variable(4, 0), P.variable(4, 1)
h = P.variable(1, 0)
request = StoppedEventRequest(
    flow=PolynomialFlow((y, -x), parameter_count=1),
    initial=(P.constant(1, 0), h),
    parameters=(Interval(1.0, 1.0001),),
    target=PolynomialEvent(x, direction=1, guard=y),
    step=0.0625, max_steps=110, order=8, derivative_order=2,
)
result = certify_stopped_event(request)
assert result.certified and verify_stopped_event(result)
assert result.time_bracket is not None
assert result.time_bracket.contains(2 * math.pi)
assert result.return_jacobian is not None
assert result.return_jacobian[1][0].contains(1.0)
```

This example is a center: the return map is the identity and its periodic
orbits are not isolated limit cycles. Establishing a cycle count requires a
separate displacement zero theorem, identity handling, and a proof that the
chosen return itineraries capture the cycles being counted. Bounds uniform
as a saddle-node or infinite-time passage degenerates require further
analytic theorems; increasing this routine's finite step budget supplies no
such uniformity.

The regression tests compare moving-section first and mixed derivatives
against analytic formulas on a deterministic grid and random samples. They
also exercise guarded full returns, nonlinear initial embeddings, conflicting
stops, unresolved tangencies, concealed earlier crossings, failed Picard
enclosures, and replay rejection after source or result tampering.
