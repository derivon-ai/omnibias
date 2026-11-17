# Global minima of a nonconvex two-center Gaussian potential

For two Gaussian centers with \(q>1\), the velocity equation can have
several stationary solutions. An exact scalar reduction identifies all
global minima, including a two-way tie, without comparing nearly equal
numerical objective values. A rigorous root bracket then encloses the
selected minimizers.

This complements the [regular inverse](gaussian-heat-inverse.md) and
[critical chart](gaussian-critical-inverse.md). It certifies global
minima of a target-tilted potential; it does not claim that all solutions
of the nonconvex velocity equation are global minima.

## Reduction of the original objective

Write the two original centers in their stored order as

\[
a_-=c-a,\qquad a_+=c+a,\qquad
\alpha=\|a\|_2^2>0,\qquad
q=\frac{\alpha}{2\nu s}>1.
\]

Let \(\delta=b_+-b_-\), \(\lambda=1/q\in(0,1)\), and
\(P=I-aa^T/\alpha\). For target \(v\), define

\[
y=q\frac{a\cdot(x-c)}{\alpha}+\frac\delta2,
\qquad
w=\frac{s\,a\cdot v}{\alpha}+\frac\delta{2q}.
\]

The Gaussian potential \(\Phi_s=-2\nu\log\tau_s\), including
the heat normalization, satisfies

\[
\Phi_s(x)-v\cdot x
=\frac{\|P(x-c)-sPv\|_2^2}{2s}
 +2\nu V_w(y)+\text{constant},
\]

where the omitted constant is independent of \(x\), and

\[
V_w(y)=\frac{\lambda y^2}{2}-\log\cosh y-wy,
\qquad
V'_w(y)=F(y)-w,\quad F(y)=\lambda y-\tanh y.
\]

The transverse minimizer is exactly \(P(x-c)=sPv\). Each scalar
global minimizer lifts to

\[
\boxed{\quad
X=c+sPv+\frac aq\left(y-\frac\delta2\right).
\quad}
\]

Thus the scalar analysis preserves the original model and objective,
including arbitrary bias contrast and transverse targets.

## Selecting every global minimum

The scalar potential is coercive because its quadratic term dominates
\(\log\cosh y\le|y|\) and the linear target term. It therefore
attains a global minimum.

For \(w>0\), reflection gives

\[
V_w(y)-V_w(-y)=-2wy<0\qquad(y>0).
\]

No negative point can be a global minimizer, and zero is excluded by
\(V'_w(0)=-w<0\). On the positive axis,

\[
F'(y)=\lambda-\operatorname{sech}^2y
\]

increases strictly from \(\lambda-1<0\) to \(\lambda>0\).
Consequently \(F\) first decreases below zero and then increases to
infinity. The equation \(F(y)=w\) has exactly one positive solution,
which must be the unique global minimizer. For \(w<0\), symmetry
selects the unique negative global minimizer.

At \(w=0\), zero is a strict local maximum since
\(V''_0(0)=\lambda-1<0\). Exactly one positive root \(m\) of
\(F(m)=0\) exists. Its reflection \(-m\) is the other global
minimizer, with equal objective value. These are the only two global
minima; the zero stationary root is excluded.

The complete classification is therefore:

| Exact effective field | All scalar global minimizers |
| --- | --- |
| \(w>0\) | The unique positive root of \(F(y)=w\). |
| \(w<0\) | The negative of the unique positive root of \(F(y)=|w|\). |
| \(w=0\) | Both signs of the unique positive root of \(F(y)=0\). |

The distinction uses exact rational arithmetic. A numerical interval
containing zero is not evidence of a tie.

## An exact hyperplane chooses the branch

Using the relation between \(q\), viscosity and age,

\[
w=\frac{s}{\alpha}\bigl(a\cdot v+\nu\delta\bigr).
\]

Since \(s/\alpha>0\), the exact tie and branch-selection hyperplane
in target space is

\[
\boxed{\qquad a\cdot v+\nu\delta=0.\qquad}
\]

Its location is independent of the midpoint and heat age while the
model remains in the \(q>1\) regime. Changing the original bias
contrast translates this hyperplane. The minimizer values on either side
remain nonlinear, and the global-minimizer map becomes set-valued on the
plane. In the \(q=1\) model the same plane locates the critical
singularity of the unique inverse instead.

This is a precise hyperplane decision mechanism within this Gaussian
family. It does not establish activation independence or turn arbitrary
neural decisions into linear constraints.

## A rational bracket for the winning positive root

Let \(W=|w|\). The unique positive root is contained strictly between

\[
\ell=1-\lambda,\qquad h=\frac{W+1}{\lambda}.
\]

To prove the lower sign, observe that for \(y>0\),

\[
\tanh y>y-y^3/3.
\]

The difference vanishes at zero and has derivative
\(y^2-\tanh^2y>0\). Since \(0<\ell<1\),

\[
F(\ell)-W< -\ell^2(1-\ell/3)-W<0.
\]

At the upper endpoint,

\[
F(h)-W=1-\tanh h>0.
\]

These endpoint signs are analytic facts about exact rational endpoints.
They remain valid when a finite-precision evaluation of \(\tanh h\)
cannot distinguish it from one. The bracket excludes the zero root at
coexistence, so it always isolates the winning positive root.

During refinement, a rigorously negative midpoint residual moves the
lower endpoint up, and a rigorously positive residual moves the upper
endpoint down. An inconclusive residual leaves the current certified
bracket intact. The initial bracket need not lie wholly in a region of
positive derivative; uniqueness of the positive root and the endpoint
signs suffice for this bisection invariant.

The implementation avoids directly subtracting nearly equal values of
\(\lambda y\) and \(\tanh y\). With \(L=\lambda y-W\),
the sign is immediate for \(L\le0\) or \(L\ge1\). Otherwise it
compares a rigorous enclosure of \(e^{-2y}\) with the exact rational
threshold \((1-L)/(1+L)\); smaller exponential values mean a negative
residual. An elementary rational upper bound for the exponential can
sometimes decide this comparison without a transcendental evaluation.

## API and completion status

`certify_gaussian_two_center_minimum(model, target, *,
tolerance=Fraction(1, 10**12), max_steps=256)` in
`omnibias.core.verified.gaussian_two_center` constructs a
`GaussianTwoCenterMinimum` certificate or returns `None`. The model
must have exactly two distinct centers and \(q>1\); target entries
are exact integers or `Fraction` values.

The result retains `model`, `target`, `midpoint`, `half_difference`,
`squared_half_distance`, `orthogonal_projector`, `bias_contrast`,
`inverse_q`, `selection_value`, `field`,
`branch_signs`, `positive_root_bounds`, `minimizer_boxes`, `tolerance`,
`max_steps`, `iterations`, `termination`, and `certificate`.
`selection_value` is \(a\cdot v+\nu\delta\), `field` is the exact
\(w\), and `positive_root_bounds` is an exact
rational pair enclosing the unique winning positive root. At an exact
tie, `branch_signs` is `(-1, 1)` and the two boxes follow that order.
Otherwise there is one sign and one global-minimizer box; the `unique`
property distinguishes these cases.

A returned certificate encloses **all global minimizers**, even when
refinement stops with a coarse bracket. `tolerance` concerns the width of
the scalar root interval, not the width of every lifted coordinate.
The termination labels distinguish completion from refinement limits:

| Termination | Interpretation |
| --- | --- |
| `tolerance` | The exact scalar bracket width meets the requested tolerance. |
| `precision` | The numerical sign enclosure cannot justify further refinement. |
| `steps` | The permitted refinement steps were exhausted. |
| `budget` | Refinement exhausted its available resource allowance. |

Near the critical parameter, resolving even the exponential comparison
can require more precision than the binary64 interval endpoints provide.
A coarse enclosure remains valid; no unresolved sign is promoted to a
decision. A certified tie does not select a single-valued optimizer or
define a derivative across the branch-selection plane.

`max_steps=0` retains the analytic initial bracket and needs no
transcendental evaluation. Exponential refinement requires the rigorous
mpmath backend and refuses prior conditional libm fallback history. The
factory raises for malformed inputs or an unavailable required backend;
unsupported models and exhausted construction or output budgets return
`None`. A refinement budget limit can still leave a valid coarse result.

```python
from fractions import Fraction
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_two_center import (
    certify_gaussian_two_center_minimum,
)

model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=Fraction(1, 4))
tie = certify_gaussian_two_center_minimum(model, (0,))
assert tie is not None and not tie.unique
assert tie.branch_signs == (-1, 1)
assert tie.minimizer_boxes[0][0].hi < -0.9
assert tie.minimizer_boxes[1][0].lo > 0.9

selected = certify_gaussian_two_center_minimum(model, (Fraction(1, 100),))
assert selected is not None and selected.unique
assert selected.branch_signs == (1,)
assert selected.selection_value == Fraction(1, 100)
assert selected.minimizer_boxes[0][0].lo > 0.9

# Exact classification survives a contrast that binary64 rounds to zero.
epsilon = Fraction(1, 10**1000)
biased = GaussianHeatMap(((-1,), (1,)), (0, epsilon), viscosity=Fraction(1, 4))
tiny = certify_gaussian_two_center_minimum(biased, (0,), max_steps=0)
assert tiny is not None and tiny.unique and tiny.branch_signs == (1,)
assert tiny.field == epsilon/4 and tiny.termination == "steps"
```

## Composing an exact affine restriction

The existing [affine restriction](gaussian-affine-constraints.md) rewrites
the original Gaussian objective on \(Cx=d\) as a projected Gaussian
objective, up to a constant independent of the feasible point. If the
projected family still has two distinct centers and \(q_T>1\), this
global-minimum certificate applies to that family at the projected
target. The projected family's unconstrained minimizers lie on the
original affine space, so they are exactly the original constrained
global minimizers. The enclosing boxes need not consist entirely of
feasible points for a general oblique constraint.

For centers \((-2,-1),(2,1)\), equal original biases, \(\nu=s=1\),
and constraint \(x_2=1\), the transformed centers are
\((-2,1),(2,1)\) and the transformed biases are \((0,1)\).
The normal-distance penalty changes the bias contrast. Consequently the
constrained tie occurs at target \(v_1=-1/2\), rather than zero.
Both the original and projected potentials are nonconvex in this example.

```python
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat

source = GaussianHeatMap(((-2, -1), (2, 1)), (0, 0))
restriction = restrict_gaussian_heat(source, ((0, 1),), (1,))
assert restriction is not None
assert restriction.model.biases == (0, 1)
assert restriction.model.contraction_bound == 2
target = (Fraction(-1, 2), Fraction(7, 3))
constrained = certify_gaussian_two_center_minimum(
    restriction.model, restriction.project_target(target),
)
assert constrained is not None and constrained.branch_signs == (-1, 1)
assert constrained.selection_value == 0
assert all(box[1].lo == box[1].hi == 1 for box in constrained.minimizer_boxes)
```

This composition is a fixed-age result. The normal-distance contribution
to transformed biases depends on age. Holding those transformed biases
fixed while changing age defines a different family and does not compute
the original constrained time evolution. In particular, the
age-independent hyperplane statement above applies to a fixed pair of
original biases; an age-dependent affine transformation must retain its
bias dependence.

This composition does not extend to inequality constraints by discarding
an infeasible unconstrained winner and checking only the boundary. For
centers \((-2),(2)\), equal biases, \(\nu=s=1\), target
\(v=1/5\), and constraint \(x\le0\), the unconstrained winner
is positive. The boundary has reduced potential \(V(0)=0\), but
\(V(-3/2)=57/80-\log\cosh(3/2)<-3/80\): use
\(\log\cosh(3/2)>3/2-\log2>3/4\). Thus a negative local
minimum omitted by the unconstrained selection wins the constrained
problem. Inequality-constrained extensions must account for these other
local minima as well as boundary candidates.

The [complete polyhedral scalar reduction](gaussian-polyhedral-reduction.md)
handles these cases by compiling every transverse quadratic piece and
bounding its global minimum, including constraint breakpoints.

## Prior work and scope

The scalar potential is the classical Curie–Weiss mean-field form.
Its coexistence and positive-field global-minimum selection are discussed
by [Kochmański, Paszkiewicz and Wolski, *Curie–Weiss magnet: a simple
model of phase transition*](https://arxiv.org/abs/1301.2141), including
their discussion around Figure 2. The specialization here binds that
classification to exact Gaussian metadata and rigorous minimizer
enclosures; no new general theorem is claimed.

Only the two-center family is covered. The certificate does not classify
all local minima or saddle points, solve an arbitrary nonconvex mixture,
or certify parameter derivatives. Its source heat field still solves its
forward PDE, but multiple stationary preimages prevent treating the
global-minimum selection as a globally smooth inverse hodograph field.
Numerical proof data and certificate sealing are not verification by an
external formal theorem prover.
