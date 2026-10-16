# Joint bias and target derivatives of constrained Gaussian decisions

The constrained Gaussian optimizer has a certified one-sided response to
simultaneous changes in its original biases and optimization target.
Centers, heat age, viscosity and polyhedral constraints remain fixed.
The response can exist where an ordinary Jacobian does not.

This interface uses a compiled critical face from a
[grouped equilibrium](gaussian-grouped-directional.md) or
[projected equilibrium](gaussian-projected-equilibrium.md). The groups
describe the base point only: a bias direction may split an equal-logit
group immediately after the perturbation.

## The joint forcing

Let \(X(b,v)\) minimize the Gaussian potential minus \(v\cdot x\)
over the fixed polyhedron, and consider

\[
b(t)=b+t\beta,\qquad v(t)=v+t\xi.
\]

At the certified base point, write \(p_j\) for the original posterior
probabilities, \(\mu=\sum_jp_ja_j\), and
\(\bar\beta=\sum_jp_j\beta_j\). Differentiating the normalized
weights at fixed \(x\) gives

\[
D_bp_j[\beta]=p_j(\beta_j-\bar\beta),\qquad
D_bU[\beta]=-
 \frac{\operatorname{Cov}_p(a,\beta)}s.
\]

Thus the effective target forcing is

\[
\boxed{\quad
\zeta=\xi+\frac{\operatorname{Cov}_p(a,\beta)}s.
\quad}
\]

Let \(H=D_xU(X)\) and let \(K\) be the original critical cone,
formed from the exact tight/strong/weak classification. The joint
one-sided Hadamard derivative is

\[
\boxed{\quad
D X(b,v;\beta,\xi)
=\lim_{t\downarrow0}
 \frac{X(b+t\beta,v+t\xi)-X(b,v)}t
=\underset{w\in K}{\arg\min}
 \left(\frac12w^THw-\zeta^Tw\right).
\quad}
\]

The Hadamard statement also permits the directions to converge as
\(t\downarrow0\). Bias changes do not alter the sufficient global
curvature bound \(m=(1-q_{\mathrm{base}})/s>0\), so the perturbed
problems remain strongly convex on the base equality space. In the
second-order variational expansion, the mixed parameter/spatial term is
\(w^T(D_bU[\beta]-\xi)=-\zeta^Tw\). The same critical-cone
argument then yields the unique quadratic-program response.

A common bias direction \(\beta_j=c\) changes the Gaussian potential
only by a spatially constant term. Its covariance and optimizer response
are exactly zero when \(\xi=0\); exact arithmetic preserves this
cancellation before interval conversion.

## Within-group and between-group contributions

For each base logit group, define the rational quantities

\[
\bar\beta_g=\frac1{n_g}\sum_{j\in g}\beta_j,
\qquad
k_g=\frac1{n_g}\sum_{j\in g}(a_j-c_g)\beta_j.
\]

With group masses \(\pi_g\), the covariance decomposes as

\[
\operatorname{Cov}_p(a,\beta)
=\sum_g\pi_gk_g+
  \operatorname{Cov}_{\pi}(c_g,\bar\beta_g).
\]

For every proposed critical face, the exact projected-centroid identities
give the same \(Pc_g\). Therefore

\[
P\zeta=P\xi+\frac1s\sum_g\pi_gPk_g.
\]

The between-group term is normal to the critical tangent. It must still
be retained in \(L\zeta\), where \(L=(EE^T)^{-1}E\), to obtain
correct equality and strong-row quadratic-program multipliers. Its
selected-weak components are structurally zero because centroid
differences lie in the span of the base-equality and strong-active
normals.

Subtracting a common exact bias direction before computing these moments
preserves gauge invariance. Likewise, centroid and group-direction
differences can remove common offsets from the between-group covariance.
The compiled probability enclosures can be reused; new exponential
evaluations are unnecessary for this directional query.

## Residual certification with interval forcing

The exact input directions \(\xi,\beta\) generally produce a
nonrational \(\zeta\). Replacing it by a rounded vector in an
exact-direction API would certify a different problem. Instead enclose
\(P\zeta\) and \(L\zeta\) rigorously and use

\[
r=PHP\widehat w-P\zeta,\qquad
\|w_E-\widehat w\|_2\le
\frac{\|r\|_{2,\mathrm{upper}}}{m},
\]

where \(\widehat w\) is the exactly projected binary64 proposal and
\(w_E\) is the true equality-QP solution on the proposed critical face.
Its multipliers satisfy

\[
\eta=L\zeta-LHPw_E.
\]

Selected weak multipliers and unselected weak slacks must be nonnegative.
Base-equality and strong-row multipliers have no sign restriction. Once
these signs are certified, the enclosure contains the true joint
directional response. The error concerns that response, not a finite
parameter step.

A zero-response shortcut requires exact zero-forcing evidence. An
interval containing zero is not enough. For example, the groupwise
identities \(P\xi+Pk_g/s=0\) establish \(P\zeta=0\) exactly.
Weak signs remain necessary even then.

The returned dual quantities describe the critical QP's linearized
stationarity. They are not a claim of a unique ordinary derivative for
an arbitrary multiplier selection when tight constraints are redundant.

## API and numerical boundary

`certify_gaussian_joint_directional(face, target_direction,
bias_direction, proposal)` returns a `GaussianJointDirectional`
certificate or `None`; both are in
`omnibias.core.verified.gaussian_joint_directional`. The `face` is produced by
`compile_gaussian_grouped_critical_face`. Target and original mode-bias
directions contain exact integers or `Fraction` values. The response
proposal contains finite binary64 values, including exactly representable
integer entries.

The result retains `target_direction`, `bias_direction` and the exact
`relative_bias_direction`, which subtracts the first mode's bias
direction. `group_bias_means` averages those relative directions within
each base group; `group_bias_covariances` contains the exact within-group
vectors \(k_g\), unchanged by the common shift.

`forcing_enclosure`, `projected_forcing_enclosure` and
`normal_forcing_enclosure` enclose \(\zeta\), \(P\zeta\) and
\(L\zeta\). `zero_projected_forcing` records the exact groupwise
zero-forcing proof. The artifact also retains the input `proposal`,
exact `projected_proposal`, exact center `value`, `enclosure`, Euclidean
`error_bound`, `coordinate_error_bounds`, and residual evidence.
An exact zero-forcing shortcut uses zero as the response center and
still checks the weak signs.

`equality_multipliers` follow the compiled constraint order, including
selected weak rows. `weak_multipliers` and `weak_slacks` follow original
inequality indices; nonweak entries are structural zeros with no primal
multiplier/slack meaning. The `face` and `certificate` retain the
dependency and proof context.

The query uses cached rigorous probability/curvature enclosures, bounded
exact arithmetic and interval matrix actions. It performs no new solve,
factorization, root iteration or exponential evaluation. Malformed
inputs raise; an unsuitable proposed face, inconclusive signs, damaged
dependencies, nonfinite bounds or exhausted resources return `None`.
A refusal does not show that the joint directional derivative is
undefined.

## An exact scalar bias-collapse response

Take centers \(-1,1\), biases \((1/4,1/4)\), \(\nu=s=1\),
target zero and the constraint \(x\le0\). At the optimizer \(X=0\),
the multiplier is zero, \(H=1/2\), and \(K=\{w\le0\}\).
For original bias direction \((\beta_1,\beta_2)\),

\[
\operatorname{Cov}(a,\beta)=\frac{\beta_2-\beta_1}{2},
\qquad
\boxed{D X(b,0;\beta,\xi)=
       \min(2\xi+\beta_2-\beta_1,0).}
\]

This response is positively homogeneous and piecewise linear in the
joint direction. It is not an ordinary Jacobian-vector product at the
switch.

```python
from fractions import Fraction
import math

from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_projected_equilibrium import (
    certify_gaussian_projected_equilibrium,
)
from omnibias.core.verified.gaussian_grouped_directional import (
    compile_gaussian_grouped_critical_face,
)
from omnibias.core.verified.gaussian_joint_directional import (
    certify_gaussian_joint_directional,
)

source = GaussianHeatMap(
    ((-1,), (1,)), (Fraction(1, 4), Fraction(1, 4)), viscosity=1, time=1,
)
base = restrict_gaussian_heat(source, (), ())
assert base is not None
primal_face = compile_gaussian_polyhedral_face(base, ((1,),), (0,), (0,))
assert primal_face is not None
equilibrium = certify_gaussian_projected_equilibrium(primal_face, (0,), (0,))
assert equilibrium is not None and equilibrium.weak_indices == (0,)
free = compile_gaussian_grouped_critical_face(equilibrium)
wall = compile_gaussian_grouped_critical_face(equilibrium, (0,))
assert free is not None and wall is not None

opening = certify_gaussian_joint_directional(free, (0,), (0, -1), (-1.0,))
closing = certify_gaussian_joint_directional(wall, (0,), (0, 1), (0.0,))
contact = certify_gaussian_joint_directional(free, (-1,), (0, 2), (0.0,))
assert opening is not None and closing is not None and contact is not None
assert opening.enclosure[0].lo <= -1 <= opening.enclosure[0].hi
assert opening.error_bound < Fraction(1, 10**12)
assert closing.value == (0,) and closing.error_bound == 0
assert contact.zero_projected_forcing
assert contact.value == (0,) and contact.error_bound == 0
```

### Zero derivative does not mean finite-step inactivity

Choose \(\xi=-1\) and \(\beta=(0,2)\). The displayed derivative
is zero. Nevertheless the perturbed optimizer is negative for every
\(t>0\). Its source velocity is

\[
U_t(x)=x-\tanh(x/2+t),\qquad v(t)=-t.
\]

At the boundary, \(U_t(0)-v(t)=t-\tanh t>0\), so the optimum lies
inside \(x<0\). The implicit equation

\[
x-\tanh(x/2+t)=-t
\]

and the Taylor expansion of \(\tanh\) give

\[
X(b+t\beta,-t)=-\frac23t^3+O(t^5).
\]

The first derivative is therefore zero while motion begins at cubic
order. A directional certificate does not establish persistence of the
active set or an exact constant decision on a finite neighborhood.

## A nonuniform projected example

Use the three-dimensional source and polyhedron from
[the projected-equilibrium example](gaussian-projected-equilibrium.md#a-switch-with-a-nonalgebraic-normal-mean):
centers \((-1,0,0),(1,0,0),(0,-1,4),(0,1,4)\), biases
\((1/4,1/4,21/4,21/4)\), equality \(z=0\), inequalities
\(x+2y\le0\), \(y\le0\), and target \((1,2,0)\).
At \(X=0\), the first inequality is strong and the second weak.

Take \(\xi=0\), \(\beta=(0,2,0,0)\), and write
\(A=1/(1+e)\), \(B=e/(1+e)\). Then

\[
\operatorname{Cov}(a,\beta)=(A,0,-4AB).
\]

With
\(h_x=(1+2e)/(2(1+e))\) and
\(h_y=(2+e)/(2(1+e))\), the weak wall releases and

\[
w=\frac{(4A,-2A,0)}{4h_x+h_y}.
\]

The equality-row quadratic-program multiplier is \(-4AB\). It comes
entirely from the between-group forcing: dropping that term would leave
the primal tangent response unchanged here but produce the wrong normal
dual response.

```python
source3 = GaussianHeatMap(
    ((-1, 0, 0), (1, 0, 0), (0, -1, 4), (0, 1, 4)),
    (Fraction(1, 4), Fraction(1, 4), Fraction(21, 4), Fraction(21, 4)),
    viscosity=1, time=1,
)
base3 = restrict_gaussian_heat(source3, ((0, 0, 1),), (0,))
assert base3 is not None
primal_face3 = compile_gaussian_polyhedral_face(
    base3, ((1, 2, 0), (0, 1, 0)), (0, 0), (0,),
)
assert primal_face3 is not None
equilibrium3 = certify_gaussian_projected_equilibrium(
    primal_face3, (1, 2, 0), (0, 0, 0),
)
assert equilibrium3 is not None
critical3 = compile_gaussian_grouped_critical_face(equilibrium3)
assert critical3 is not None

A = 1 / (1 + math.e)
hx = (1 + 2 * math.e) / (2 * (1 + math.e))
hy = (2 + math.e) / (2 * (1 + math.e))
denominator = 4 * hx + hy
joint = certify_gaussian_joint_directional(
    critical3, (0, 0, 0), (0, 2, 0, 0),
    (4 * A / denominator, -2 * A / denominator, 0.0),
)
assert joint is not None
assert joint.error_bound < Fraction(1, 10**12)
assert joint.enclosure[0].lo > 0 and joint.enclosure[1].hi < 0
# The normal term affects the equality-row QP multiplier.
assert joint.forcing_enclosure[2].hi < 0
assert joint.normal_forcing_enclosure[0].hi < 0
assert joint.equality_multipliers[0].hi < 0
```

## Scope and prior work

This is classical parametric critical-cone sensitivity specialized to
Gaussian moments and exact constraint geometry. See
[Jittorntrum's directional differentiability result without strict complementarity](https://link.springer.com/chapter/10.1007/BFb0121215)
and [Levy and Rockafellar on solution mappings](https://sites.math.washington.edu/~rtr/papers/rtr163-GeometrySolutionMappings.pdf).
No general novelty or performance claim is made.

Only original biases and targets vary. Centers, viscosity, age and
constraint data do not. The certificate gives one first directional
response, not an ordinary VJP at a switch, a finite-step error bound,
higher derivatives or a complete active-set search. Dense matrices and
exact arithmetic remain bounded by resource limits; sealed evidence is
not formal theorem-prover verification.
