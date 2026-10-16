# Global minimum enclosures on compact boxes

`minimize_box` bounds the global minimum of a scalar function on a declared
compact box. It returns a feasible witness, an interval containing the minimum
value, and boxes covering **every** global minimizer. A finite computation can
end with an unresolved gap; the return status reports that outcome.

This connects [deep derivative propagation](deep-bias-collapse.md) to a global
argument: verified jets enclose derivatives throughout a box, and interval
branch-and-bound uses those bounds to exclude regions or reduce them to faces.
It implements classical interval optimization, with no claim of a new general
optimization theorem or uniformly fast solution of nonconvex problems.

## A deep scalar network

The following network represents
\(F(x)=\sin(\tanh(\sigma(x)))\). Each layer has unit weight and zero bias.
The adapter requires the rigorous transcendental backend; install
`omnibias-core[verified]` for this example.

```python
from omnibias.core.verified.global_min import minimize_box
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.mlp_objective import IntervalMLP

objective = IntervalMLP.from_layers([
    ([[1.0]], [0.0], "sigmoid"),
    ([[1.0]], [0.0], "tanh"),
    ([[1.0]], [0.0], "sin"),
])
assert objective.input_dim == 1

result = minimize_box(
    objective.value,
    (Interval(-2.0, 2.0),),
    gradient=objective.gradient,
    value_tolerance=1e-6,
    max_boxes=127,
)
assert result.status == "value_tolerance"
assert result.gap_upper <= 1e-6
assert result.point == (-2.0,)
print(result.minimum, result.point_value)
```

Here every factor of
\[
F'(x)=\cos(\tanh(\sigma(x)))
       (1-\tanh^2(\sigma(x)))\sigma(x)(1-\sigma(x))
\]
is positive. A positive derivative enclosure allows reduction to the left
endpoint. This proves a minimum on the supplied interval. Over all of
\(\mathbb R\), this function has infimum zero and no finite minimizer.

`IntervalMLP.from_layers` accepts affine layers followed by `None`, `"sigmoid"`,
`"tanh"`, `"sin"`, or `"cos"`, with a scalar final output. It freezes stored
floating-point weights and biases as exact real constants. Its bounds concern
the ideal real-valued network with those constants; they do not bound an
arbitrary hardware forward pass with its own rounding behavior. Mutating the
original input lists does not change the stored objective.

This adapter optimizes **network inputs** with fixed parameters. Training all
weights and biases needs a different objective representation. A custom
`minimize_box` callback can instead treat its box coordinates as parameters and
evaluate the loss with interval-valued weights. For example, a value callback
for \((\sin(w)-1/2)^2\) on \([-3,3]\) can enclose both zero-loss weights
\(\pi/6\) and \(5\pi/6\). Any supplied gradient must then enclose derivatives
with respect to those parameters; `IntervalMLP.gradient` remains an input
gradient. Importing arbitrary compiled Python, Torch, or JAX graphs into this
verified adapter remains future work.

## Multiple minima in two dimensions

Custom interval callbacks can describe objectives independently of the network
adapter. Polynomial arithmetic needs no transcendental backend.

```python
def value(box):
    x, y = box
    return (x**2 - 1.0)**2 + y**2

def gradient(box):
    x, y = box
    return (4.0*x*(x**2 - 1.0), 2.0*y)

result = minimize_box(
    value,
    (Interval(-2.0, 2.0), Interval(-1.0, 1.0)),
    gradient=gradient,
    value_tolerance=1e-6,
    max_boxes=511,
)
assert result.minimum.lo <= 0.0 <= result.minimum.hi
assert result.status == "value_tolerance"
assert result.gap_upper <= 1e-6
for minimizer in ((-1.0, 0.0), (1.0, 0.0)):
    assert any(
        all(interval.lo <= x <= interval.hi
            for interval, x in zip(box, minimizer))
        for box in result.candidates
    )
print(result.point, result.candidates)
```

The exact minima are \((-1,0)\) and \((1,0)\), both with value zero. The returned
candidate boxes cover both. A small **value** gap does not require that these
boxes individually isolate the minimizers, or that the witness be the unique
minimizer. For a constant objective, the entire initial domain can remain a
valid candidate box.

## Callback and result contract

The domain is a nonempty finite-dimensional box with finite ordered endpoints;
zero-width coordinates are allowed. Endpoints use stored binary64 floats or
integers exactly representable in that format. Inexact conversion is refused
because moving an endpoint would change the optimization problem. For every
box the routine evaluates,
`value(box)` must return an `Interval` containing the real range of the same
continuous objective \(f\). If supplied, `gradient(box)` must return one
interval per input coordinate, enclosing each partial derivative throughout
that box. The gradient argument requires \(f\) to be continuously
differentiable on a neighborhood of the domain. A plain sampled gradient is
insufficient. Omitting the gradient gives value-only branch-and-bound.

For arbitrary callbacks, these enclosure properties are assumptions; the
optimizer cannot establish them by sampling. `IntervalMLP` supplies them for its
supported fixed network using outward interval arithmetic and verified jets.
It refuses a conditional transcendental fallback when a rigorous backend is
required. Callback exceptions and invalid enclosures are failures, not evidence
that the corresponding region is empty. Sound unbounded callback enclosures are
allowed, but can keep the minimum-value gap infinite. Unusable centered-form
arithmetic falls back to the other valid bounds.

The returned `BoxMinimum` exposes:

| Field | Meaning |
| --- | --- |
| `minimum` | An interval containing the global minimum value. |
| `point` | A feasible input in the original domain. |
| `point_value` | An interval enclosing the objective at that witness. |
| `candidates` | Retained boxes whose union contains all global minimizers. |
| `gap_upper` | An outward upper bound on the remaining minimum-value gap. |
| `status` | `"value_tolerance"`, `"box_budget"`, or `"resolution_limit"`. |

The counters `evaluated_boxes`, `value_evaluations`, and
`gradient_evaluations` report the work performed. `max_boxes` counts evaluated
boxes, including the root and boxes later discarded; each uses at most two
value calls and one gradient call.

`"value_tolerance"` means the outward gap is at most `value_tolerance`.
`"box_budget"` means the `max_boxes` resource limit stopped refinement.
`"resolution_limit"` means the remaining uncertainty could not be resolved by
further representable subdivision. The latter two statuses still return valid
bounds and a minimizer cover under the callback assumptions. Neither reports
that the requested value accuracy was achieved. Interval uncertainty can remain
even on a singleton input.

## Why the enclosure survives refinement

Let \(\mathcal C\) be the retained boxes, including boxes that can no longer be
split. Give each box a lower bound \(L_B\le\min_{x\in B}f(x)\). A feasible
witness \(p\) and its interval evaluation supply \(f(p)\le U\). The invariant is

\[
\operatorname{argmin}_{x\in D}f(x)\subseteq\bigcup_{B\in\mathcal C} B,
\qquad
\min_{B\in\mathcal C}L_B\le \min_{x\in D}f(x)\le f(p)\le U.
\]

For a center \(c\in B\), the fundamental theorem of calculus along the segment
from \(c\) to \(x\) gives the mean-value inclusion

\[
f(B)\subseteq F(\{c\})+\sum_i G_i(B)(B_i-c_i).
\]

All arithmetic in this expression rounds outward. Its lower endpoint can be
combined with the natural value enclosure and inherited parent bound by taking
their maximum. Unknown or unusable derivative bounds must not manufacture a
tighter lower bound.

If \(G_i(B)\) is strictly positive, every point away from the lower face has a
strictly better point on that face with the other coordinates fixed. Therefore
all minimizers in \(B\) lie on the face. A strictly negative enclosure gives the
upper face. A nonstrict sign is insufficient for preserving **all** minimizers:
a flat function can minimize everywhere. Merely excluding zero from a gradient
component does not justify discarding boundary minima.

A box can be discarded when \(L_B>U\). Using \(L_B\ge U\) could discard another
minimizer tied with the witness. Otherwise, a split replaces its parent by two
closed children covering it. Replacement must be atomic with respect to the
budget: the algorithm keeps the parent until the children are represented.
Resolution-limited boxes remain in both the candidate union and the global
lower-bound calculation. These operations preserve the invariant by induction.

## Scope and prior work

The [positive exponential mean-map inverse](inverse-pde-map.md) has additional
convex structure: a certified stationary point and exact full affine rank
establish its unique global minimum over all of real space. That specialized
argument does not require a compact search box; it relies on properties that
an arbitrary neural-network objective need not have.

Lowest-lower-bound selection, interval value bounds, centered forms and
monotonicity tests are established interval global optimization techniques.
[Krawczyk and Neumaier, *Interval Slopes for Rational Functions and Associated
Centered Forms*](https://doi.org/10.1137/0722037) develops interval slope enclosures
and centered forms.
[Csendes, *Generalized subinterval selection criteria for interval global
optimization*](https://www.inf.u-szeged.hu/~csendes/numalg.pdf) identifies the
lowest-lower-bound rule with Moore–Skelboe and treats convergence assumptions
separately from enclosure preservation. [*Polytopal Spatial Branch and Bound
Global Optimization Algorithm*](https://informatica.vu.lt/journal/INFORMATICA/article/1389/read)
explicitly studies all-minimizer enclosures and monotonic reduction to faces.

The finite-run guarantee does not imply efficient convergence. Interval
overestimation, rounding floors, high dimension and many competing minima can
exhaust the budget. No guarantee extends beyond the declared box, and extra
nonlinear constraints need their own verified feasibility treatment. This
routine provides an interval computation under stated assumptions; it does not
run a Lean proof checker or issue a formal theorem-prover certificate.
