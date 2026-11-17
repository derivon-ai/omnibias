# Exact constrained equilibria with nonalgebraic Gaussian means

A Gaussian decision can have an exact rational constrained optimizer even
when its full posterior mean and equality multipliers are nonalgebraic.
The constraints remove the normal component of stationarity. This
interface checks the remaining tangent equation exactly, proves which
tight inequalities have zero or positive multipliers, and reuses the
[grouped directional certificate](gaussian-grouped-directional.md).

The Gaussian source, heat age, viscosity and polyhedron remain fixed.
The directional result concerns changes to the target only.

[Joint bias and target derivatives](gaussian-joint-directional.md) extend
this calculus to simultaneous changes in the original Gaussian log weights
and target, while keeping centers, heat age, viscosity and geometry fixed.

## Projected group centroids establish stationarity

At a rational point \(X\), partition the original rational Gaussian
scores

\[
\ell_j=b_j-\frac{\|X-a_j\|^2}{4\nu s}
\]

by exact equality. Let \(c_g\) be each group's exact centroid,
\(n_g\) its size, and \(\alpha_g\) its score relative to the first
mode. Its positive posterior mass is

\[
\pi_g=\frac{n_ge^{\alpha_g}}{\sum_h n_he^{\alpha_h}},
\qquad \mu=\sum_g\pi_gc_g.
\]

Unlike the shared-full-centroid interface, the \(c_g\) may differ.
For the proposed primal face, let \(E\) stack the original equality
rows and selected tight inequality rows. The rows must be independent.
Let \(P_E\) project orthogonally onto \(\ker E\), and set
\(L_E=(EE^T)^{-1}E\). Require the exact identities

\[
P_Ec_g=P_E(X-sv)\qquad\text{for every group }g.
\]

Then the tangent stationarity equation holds regardless of the group
weights. More explicitly, define rational group multiplier coefficients

\[
f_g=v-\frac Xs+\frac{c_g}s,\qquad
\lambda_g=L_Ef_g.
\]

The checker replays \(E^T\lambda_g=f_g\) exactly for every group.
Thus the true multiplier vector \(\lambda=\sum_g\pi_g\lambda_g\)
satisfies

\[
U(X)-v+E^T\lambda=0.
\]

All original inequalities must be feasible, and selected inequality
multipliers must be nonnegative. With \(q_{\mathrm{base}}<1\) on
the base equality space, these KKT conditions certify the unique global
minimum over the declared polyhedron. The ambient Hessian may be
indefinite.

## Exact weak rows, rigorously positive strong rows

For a selected inequality row \(i\), its multiplier is a positive
weighted average of the rational coefficients \(\lambda_{g,i}\).
The checker uses the following evidence:

- If every coefficient is zero, the multiplier is exactly zero.
- If all coefficients are nonnegative and at least one is positive,
  the multiplier is strictly positive because every true group mass is
  positive. This remains valid when a stored interval lower endpoint is
  zero from underflow.
- An exact Abel-summation test can prove strict positivity even with
  mixed-sign coefficients and arbitrarily small exact score gaps.
- Otherwise, a rigorously positive interval lower bound proves strict
  positivity. An inconclusive sign causes refusal.

Omitted inequality multipliers are exactly zero in this representation.
Every exactly tight inequality is retained, including omitted or
redundant rows. Tight zero-multiplier rows are weak-active; positive
ones are strong-active. Equality multipliers have no sign restriction.
The recorded strong indices carry their proof of positivity; they need
not be recoverable from the rounded interval endpoints alone.

For the declared rational data, the groupwise zero test is also
necessary. If a weighted multiplier is zero, then

\[
\sum_g n_g\lambda_{g,i}e^{\alpha_g}=0
\]

forces every coefficient to vanish by Lindemann--Weierstrass.
Likewise, rational projected stationarity forces the projected group
centroids to agree with \(P_E(X-sv)\). These are corollaries of the
classical independence theorem; see
[Popescu's primary proof](https://arxiv.org/abs/2306.14352).
They do not apply to arbitrary real biases or irrational constraint data,
and do not make every proposed face or numerical sign enclosure succeed.

### Exact signs from ordered logits

Sort the distinct exact group scores so
\(\alpha_0<\cdots<\alpha_{G-1}\). For one multiplier, put
\(a_g=n_g\lambda_{g,i}\), including the group cardinality, and form
the exact suffix sums \(S_k=\sum_{g=k}^{G-1}a_g\). Finite summation
by parts gives

\[
\sum_g a_ge^{\alpha_g}
=S_0e^{\alpha_0}
 +\sum_{k=1}^{G-1}S_k
     \bigl(e^{\alpha_k}-e^{\alpha_{k-1}}\bigr).
\]

This follows directly by substituting \(a_k=S_k-S_{k+1}\), with
\(S_G=0\), and telescoping. Every displayed exponential or difference
is strictly positive. Therefore nonnegative suffix sums with at least
one positive entry prove a strictly positive multiplier; its denominator
is positive as well. No numerical exponential subtraction or positive
binary64 lower margin is needed for this sign proof.

This is classical
[Abel summation by parts](https://fractal.math.unr.edu/~ejolson/311-06/handout/sumpart.pdf),
applied to the exact group coefficients. Exact ordering, merging equal
scores and including cardinalities are essential. It is a sufficient
criterion, not a complete sign test: positive expressions such as
\(2-e^{1/10}\) fail the suffix test and can still be handled by
rigorous interval bounds. Probability enclosures used elsewhere in the
artifact retain their usual backend requirements.

## Why within-group moments suffice for directional curvature

For a valid multiplier representation, the critical cone imposes the
base equalities and strong-active rows as equalities, while weak-active
rows remain inequalities. Since every weak multiplier coefficient
vanishes groupwise, each \(f_g\) already lies in the span of the
base-equality and strong-active normals.

Consequently, for every later critical face with projector \(P\),
the projected group centroids \(Pc_g\) are identical. Selecting a
subset of weak rows as additional equalities preserves this identity;
the compiler replays it exactly.

Define the exact within-group covariance

\[
\Sigma_g=\frac1{n_g}\sum_{j\in g}(a_j-c_g)(a_j-c_g)^T.
\]

The full covariance contains an additional term:

\[
\Sigma=\sum_g\pi_g\Sigma_g+
       \sum_g\pi_g(c_g-\mu)(c_g-\mu)^T.
\]

That between-group term must be retained in the full Hessian
\(H=I/s-\Sigma/(2\nu s^2)\). It disappears after right
multiplication by \(P\), since \(P(c_g-\mu)=0\). For the later
face's exact left inverse \(L\), \(LP=0\), giving

\[
PHP=\frac Ps-
       \frac{\sum_g\pi_gP\Sigma_gP}{2\nu s^2},
\qquad
LHP=-\frac{\sum_g\pi_gL\Sigma_gP}{2\nu s^2}.
\]

Thus the derivative query can use exact within-group centered moments
before interval conversion, even though the full mean is unknown in
rational arithmetic. Substituting within-group covariance for the full
covariance everywhere would be incorrect.

The existing residual proof applies unchanged. For the exactly projected
binary64 proposal \(\widehat w\), an outward enclosure of
\(P(H\widehat w-\xi)\) gives an error bound divided by
\(m=(1-q_{\mathrm{base}})/s\). Tangent coordinate radii and projected
weak-row norms preserve structural zero slacks. Selected weak multiplier
signs and unselected weak slacks still have to be certified before the
enclosure represents the true critical-cone derivative.

## API and numerical boundary

`certify_gaussian_projected_equilibrium(face, target, point)` in
`omnibias.core.verified.gaussian_projected_equilibrium` takes exact
integer or `Fraction` coordinates and returns a
`GaussianProjectedEquilibrium` or `None`.

The artifact retains `groups` and relative `group_logits` in first-appearance
order, exact `group_centroids`, `group_centroid_offsets` relative to the
first group, and `centered_centers` centered within each mode's **own**
group. `projected_centroid` is the common projection onto the proposed
primal face's tangent. `group_multipliers` contains one exact coefficient
row per group, in the original selected-face constraint order.

Its `posterior`, `group_probabilities`, `equality_multipliers` and
`inequality_multipliers` are interval enclosures. `slacks` and the
`active_indices`, `strong_indices`, `weak_indices` carry exact
feasibility/classification evidence. The source `face`, `target`,
`point`, `strong_convexity_bound` and `certificate` bind these quantities
to the declared problem. This artifact does not supply an exact full
mean or a Hessian.

The existing `compile_gaussian_grouped_critical_face` accepts this
artifact and checks the projected-centroid identity on each proposed
critical face. It reuses the sealed probability enclosures, adds
between-group covariance to its full `hessian`, and retains the
within-group formulas for `projected_hessian` and `multiplier_hessian`.
`certify_gaussian_grouped_directional` then accepts exact directions
and finite binary64 response proposals as before. No new response
solver or active-set search is introduced.

The equilibrium check performs no new solve, factorization or root
iteration after the original face has been compiled. Multiple logit
groups require a rigorous exponential backend; a single group has
exact uniform weights. Invalid exact inputs raise. Failed identities,
inconclusive multiplier signs, failed base contraction, damaged evidence,
nonfinite enclosures or exhausted budgets return `None`; a required
missing rigorous backend raises `RuntimeError`. Prior fallback history
is preserved.

## A switch with a nonalgebraic normal mean

Take \(\nu=s=1\), centers

\[
(-1,0,0),\ (1,0,0),\ (0,-1,4),\ (0,1,4),
\]

and biases \((1/4,1/4,21/4,21/4)\). Impose the equality \(z=0\)
and inequalities \(x+2y\le0\), \(y\le0\), with target
\(v=(1,2,0)\). At \(X=0\), the two score groups have relative
scores \(0,1\) and centroids \((0,0,0),(0,0,4)\). Hence

\[
\mu=\left(0,0,\frac{4e}{1+e}\right),\qquad
\lambda_{z=0}=\frac{4e}{1+e},\quad
\lambda_{x+2y\le0}=1,\quad \lambda_{y\le0}=0.
\]

The full mean and equality multiplier are nonalgebraic, but feasibility,
tangent stationarity and the strong/weak split have exact evidence.
The base equality gives \(q_{\mathrm{base}}=1/2\), whereas the
ambient value is \(9/4\). In fact

\[
H=\operatorname{diag}(h_x,h_y,h_z),\quad
h_x=\frac{1+2e}{2(1+e)},\quad
h_y=\frac{2+e}{2(1+e)},\quad
h_z=1-\frac{8e}{(1+e)^2}<0.
\]

For target direction \((1,0,0)\), the weak wall releases and

\[
DX(v;(1,0,0))=\frac{(4,-2,0)}{4h_x+h_y}.
\]

For the opposite direction the weak wall binds and the derivative is
zero. The response is directional, not an ordinary Jacobian.

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
    certify_gaussian_grouped_directional,
)

source = GaussianHeatMap(
    ((-1, 0, 0), (1, 0, 0), (0, -1, 4), (0, 1, 4)),
    (Fraction(1, 4), Fraction(1, 4), Fraction(21, 4), Fraction(21, 4)),
    viscosity=1, time=1,
)
base = restrict_gaussian_heat(source, ((0, 0, 1),), (0,))
assert base is not None
assert source.contraction_bound == Fraction(9, 4)
assert base.model.contraction_bound == Fraction(1, 2)
primal_face = compile_gaussian_polyhedral_face(
    base, ((1, 2, 0), (0, 1, 0)), (0, 0), active_indices=(0,),
)
assert primal_face is not None
equilibrium = certify_gaussian_projected_equilibrium(
    primal_face, (1, 2, 0), (0, 0, 0),
)
assert equilibrium is not None
assert equilibrium.group_centroids == ((0, 0, 0), (0, 0, 4))
assert equilibrium.group_multipliers == ((0, 1), (4, 1))
assert equilibrium.strong_indices == (0,)
assert equilibrium.weak_indices == (1,)

releasing = compile_gaussian_grouped_critical_face(equilibrium)
binding = compile_gaussian_grouped_critical_face(equilibrium, (1,))
assert releasing is not None and binding is not None
# Between-group covariance is essential for this full-Hessian entry.
assert releasing.hessian[2][2].hi < 0

# Floating-point formulas propose the response; intervals certify it.
hx = (1 + 2 * math.e) / (2 * (1 + math.e))
hy = (2 + math.e) / (2 * (1 + math.e))
denominator = 4 * hx + hy
positive = certify_gaussian_grouped_directional(
    releasing, (1, 0, 0), (4 / denominator, -2 / denominator, 0.0),
)
negative = certify_gaussian_grouped_directional(
    binding, (-1, 0, 0), (0.0, 0.0, 0.0),
)
assert positive is not None and negative is not None
assert positive.error_bound < Fraction(1, 10**12)
assert positive.enclosure[0].lo > 0
assert positive.enclosure[1].hi < 0
assert negative.value == (0, 0, 0)
assert negative.error_bound == 0
```

## A bias limit need not commute with decision differentiation

Consider the scalar source with centers \(-1,1\), \(\nu=s=1\),
biases \((1/4,1/4+\varepsilon)\), and constraint \(x\le0\).
Its velocity is

\[
U_\varepsilon(x)=x-\tanh\!\left(\frac{x+\varepsilon}{2}\right),
\qquad U_\varepsilon'(x)\ge\frac12.
\]

At target zero, the optimizer is \(X_\varepsilon(0)=0\) for every
\(\varepsilon\ge0\). For \(\varepsilon>0\), the active
multiplier is \(\tanh(\varepsilon/2)>0\). Its group coefficients
are \((-1,1)\), its ordered scores are \((0,\varepsilon)\), and
its suffix sums are \((0,1)\). The Abel proof establishes strict
activity even for an exact gap below binary64 resolution.

For a perturbed target \(-t\), \(t\ge0\), the optimizer stays
at zero exactly while \(t\le\tanh(\varepsilon/2)\). Therefore

\[
DX_\varepsilon(0;-1)=0\quad(\varepsilon>0).
\]

At \(\varepsilon=0\), the multiplier is exactly zero and the
critical cone is \(w\le0\), with Hessian \(1/2\). Its response
in direction \(-1\) is \(-2\). Thus

\[
\lim_{\varepsilon\downarrow0}\lim_{t\downarrow0}
 \frac{X_\varepsilon(-t)-X_\varepsilon(0)}t=0,
\qquad
\lim_{t\downarrow0}\lim_{\varepsilon\downarrow0}
 \frac{X_\varepsilon(-t)-X_\varepsilon(0)}t=-2.
\]

This is a concrete noncommutation of a bias limit and target
differentiation for constrained Gaussian decisions. It does not
contradict normalized finite-difference identities for smooth
activation derivatives. The strict-active neighborhood shrinks to
zero as the two biases meet.

```python
def scalar_equilibrium(epsilon):
    scalar_source = GaussianHeatMap(
        ((-1,), (1,)),
        (Fraction(1, 4), Fraction(1, 4) + epsilon),
        viscosity=1, time=1,
    )
    scalar_base = restrict_gaussian_heat(scalar_source, (), ())
    assert scalar_base is not None
    scalar_face = compile_gaussian_polyhedral_face(
        scalar_base, ((1,),), (0,), active_indices=(0,),
    )
    assert scalar_face is not None
    result = certify_gaussian_projected_equilibrium(scalar_face, (0,), (0,))
    assert result is not None
    return result

# Keep the bias gap exact: converting these biases to float changes the model.
tiny = scalar_equilibrium(Fraction(1, 10**1000))
collapsed = scalar_equilibrium(Fraction(0))
assert tiny.strong_indices == (0,) and tiny.weak_indices == ()
assert tiny.inequality_multipliers[0].lo == 0
assert collapsed.strong_indices == () and collapsed.weak_indices == (0,)

tiny_face = compile_gaussian_grouped_critical_face(tiny)
collapsed_face = compile_gaussian_grouped_critical_face(collapsed)
assert tiny_face is not None and collapsed_face is not None
tiny_response = certify_gaussian_grouped_directional(tiny_face, (-1,), (0.0,))
collapsed_response = certify_gaussian_grouped_directional(
    collapsed_face, (-1,), (-2.0,),
)
assert tiny_response is not None and collapsed_response is not None
assert tiny_response.value == (0,) and tiny_response.error_bound == 0
assert collapsed_response.enclosure[0].lo <= -2 <= collapsed_response.enclosure[0].hi
assert collapsed_response.error_bound < Fraction(1, 10**12)
```

## Scope and prior work

This specializes classical KKT and critical-cone sensitivity to exact
Gaussian logit-group identities; no general mathematical novelty is
claimed. The directional theorem is described in
[Jittorntrum's work without strict complementarity](https://link.springer.com/chapter/10.1007/BFb0121215)
and [Levy and Rockafellar's analysis of solution mappings](https://sites.math.washington.edu/~rtr/papers/rtr163-GeometrySolutionMappings.pdf).

The interface certifies one exact equilibrium and proposed directional
faces. It performs no active-set search and makes no claim about arbitrary
nonconvex optimization, finite-step remainders, higher derivatives or
derivatives under changing constraints. Dense geometry and interval
curvature remain resource-bounded; certificate sealing is not a formal
theorem-prover check.
