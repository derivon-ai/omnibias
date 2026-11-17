# Certified projected-mean solves for Gaussian decisions

The [posterior-domain uniqueness bound](gaussian-entropy-uniqueness.md) also
gives a contraction on a box containing every possible projected posterior
mean. It can support a certified solve even when the Gaussian potential is
nonconvex elsewhere and its global diameter contraction test fails.

The numerical proposal is kept separate from its proof. An exact rational
probability vector produces an exactly feasible projected point. A rigorous
normal-cone residual then bounds that point's distance and objective gap from
the unique global optimizer. The proposal need not equal the true posterior.

## From the existing box to a contraction

Let \(K\) be the exact convex box recorded as `physical_bounds` in the
posterior-domain certificate. That source computes affine logit-contrast
bounds throughout \(K\), not just at known minimizers. Therefore every
original posterior \(\pi(x)\), for \(x\in K\), lies in the
certified probability polytope \(P\).

For a successful uniqueness bound \(B<2\nu s\), define

\[
\rho=\frac{B}{2\nu s}<1,\qquad
\mu_{\rm sc}=\frac{1-\rho}{s}
=\frac{2\nu s-B}{2\nu s^2}>0.
\]

The Gaussian mean derivative is
\(D\mu(x)=\operatorname{Cov}_{\pi(x)}(a)/(2\nu s)\).
The covariance bound throughout \(K\), convexity of \(K\), and
nonexpansiveness of projection show that

\[
T(x)=\operatorname{Proj}_C\bigl(sv+\mu(x)\bigr)
\]

is \(\rho\)-Lipschitz on \(K\). Moreover,

\[
T(\mathbb R^d)\subset
\operatorname{Proj}_C(sv+\operatorname{conv}\{a_j\})\subset K.
\]

The closed box is complete, so Banach's theorem gives a unique fixed point.
Every constrained stationary point satisfies this fixed-point equation, and
a global minimum exists by coercivity. Thus that point is the unique
constrained stationary point and global minimum. Exact real iteration enters
\(K\) after one step from any real initial point and converges thereafter.

The same bound gives
\(\nabla^2F(x)\succeq\mu_{\rm sc}I\) throughout \(K\).
It does not assert convexity outside \(K\). The spatial factor
\(\mu_{\rm sc}\) differs from the entropy monotonicity factor
\(2\nu-B/s\) stored in the uniqueness certificate.

## A feasible candidate and a rigorous residual

Choose any exact rational simplex vector \(p\):
\(p_j\ge0\), \(\sum_jp_j=1\). It may lie on the simplex boundary
or outside the certified posterior polytope. Compute

\[
x=\operatorname{Proj}_C\left(sv+\sum_jp_ja_j\right).
\]

The exact rational projection is feasible and belongs to \(K\). For the
true original posterior at this point, set

\[
r=\frac1s\sum_j\bigl(p_j-\pi_j(x)\bigr)a_j.
\]

Projection supplies an exact normal vector \(n\in N_C(x)\) satisfying
\(\nabla F(x)+n=r\). Strong convexity on \(K\) therefore yields

\[
\boxed{\|x-x_*\|\le\frac{\|r\|}{\mu_{\rm sc}}},\qquad
\boxed{0\le F(x)-F(x_*)\le
\frac{\|r\|^2}{2\mu_{\rm sc}}}.
\]

For example, the gap inequality follows by applying strong convexity to
\(x_*\), dropping the nonnegative normal-cone contribution and completing
the square. Strong monotonicity of the gradient plus the normal cone gives
the distance inequality. Both points and their connecting segment lie in
\(K\), which is essential: the same local bound cannot be applied to an
arbitrary point outside that box.

Interval posterior enclosures produce a rigorous residual bound. Exact center
differences can be used because \(\sum_j(p_j-\pi_j)=0\), avoiding
large common-translation cancellation. No logarithm of a proposed weight is
needed, so zero proposal weights are allowed. An enclosure box may contain
infeasible points; the recorded rational candidate itself is exactly feasible.

## Residual and solver APIs

`certify_gaussian_entropy_residual(unique, probabilities)` in
`omnibias.core.verified.gaussian_entropy_residual` takes a
`GaussianEntropyUnique` and an exact integer/`Fraction` simplex vector.
It returns `GaussianEntropyResidual` or refuses.

| Field | Meaning |
| --- | --- |
| `unique`, `probabilities`, `point` | Bound source proof, exact proposal and exact feasible projection. |
| `projection_source`, `normal` | Exact \(sv+\sum_jp_ja_j\) and its projection normal divided by \(s\). |
| `posterior`, `residual` | Rigorous original posterior and normal-cone residual enclosures. |
| `residual_squared_bound` | Exact rational upper bound on \(\|r\|^2\). |
| `contraction_bound`, `strong_convexity_bound` | Exact \(\rho\) and \(\mu_{\rm sc}\). |
| `error_bound`, `enclosure` | Outward Euclidean distance bound and a box containing the unique optimizer. |
| `value`, `minimum` | Original normalized objective at the candidate and global minimum enclosures. |
| `suboptimality_bound` | Exact rational upper bound on the candidate's objective gap. |

Objective values use the same centered normalization as the
[entropy global optimizer](gaussian-entropy-global.md); omitted common heat,
bias and target-translation constants do not change minimizers.

`solve_gaussian_entropy(unique, *, initial_probabilities=None,
tolerance=1e-8, max_iterations=32)` in
`omnibias.core.verified.gaussian_entropy_solver` repeatedly proposes rational
probabilities and certifies their projected points. The default initial vector
is uniform. Updates normalize exact rational midpoints of posterior intervals.
These are numerical proposals, not assertions that the posterior is rational.
Each accepted residual certificate includes their actual error.

The returned `GaussianEntropySolve` stores the selected `residual`,
`iterations`, `evaluations`, `selected_iteration`, `error_bounds`, `status`,
`tolerance`, `max_iterations` and the source `unique`. It retains the best
certified residual-squared bound among evaluated iterates. The history need
not improve monotonically.

`max_iterations` counts updates after the initial evaluation; zero is allowed.
`status="tolerance"` confirms the requested **spatial distance** bound.
`"iteration_budget"` and `"stagnation"` return the best certified result
without asserting that accuracy. Objective-gap accuracy is reported separately.
Refusal, resource limits or inadequate transcendental enclosures cannot be
interpreted as convergence. Source integrity, resource and provenance rules
remain in force.

## A nonconvex example

This family has global diameter bound \(q=4\), yet its certified posterior
domain gives a contraction around the entire possible optimizer image. Its
potential has a negative Hessian entry near \((6,0)\), outside that image.

```python
from omnibias.core.verified.gaussian_entropy_solver import solve_gaussian_entropy
from omnibias.core.verified.gaussian_entropy_unique import (
    certify_gaussian_entropy_unique,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import (
    certify_gaussian_posterior_domain,
)
from omnibias.core.verified.polyhedral_projection import (
    compile_polyhedral_projector,
)

model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -8, -8))
projector = compile_polyhedral_projector(2, (), ())
assert projector is not None
domain = certify_gaussian_posterior_domain(model, projector, (0, 0))
assert domain is not None
unique = certify_gaussian_entropy_unique(domain)
assert unique is not None and model.contraction_bound == 4
solution = solve_gaussian_entropy(unique, tolerance=1e-10, max_iterations=8)
assert solution is not None and solution.status == "tolerance"
assert solution.residual.error_bound <= 1e-10
assert solution.residual.contraction_bound < 1
```

A single boundary proposal also obtains a valid, possibly coarse certificate:

```python
from omnibias.core.verified.gaussian_entropy_residual import (
    certify_gaussian_entropy_residual,
)

coarse = certify_gaussian_entropy_residual(unique, (0, 1, 0))
assert coarse is not None and coarse.point == (4, 0)
assert coarse.error_bound > solution.residual.error_bound
```

## Numerical and mathematical limits

For inexact updates using proposals \(p_n\), let
\(\epsilon_n=\|\sum_j(p_{n,j}-\pi_j(x_n))a_j\|\).
Once \(x_n\in K\), nonexpansiveness gives

\[
\|x_{n+1}-x_*\|\le\rho\|x_n-x_*\|+\epsilon_n.
\]

Vanishing errors imply convergence, while persistent numerical errors can
leave an error floor. The exact-real theorem does not guarantee that a fixed
precision run meets every requested tolerance. The implementation checks its
actual residuals and termination state instead.

Gaussian posterior-mean iteration is classical; see
[Carreira-Perpiñán, Equation 7](https://faculty.ucmerced.edu/mcarreira-perpinan/papers/pami07.pdf).
Here the declared posterior-domain bound, convex projection and residual
inequalities supply the stated global guarantees. No new general convergence
theorem, universal speed claim or selected-optimizer derivative is asserted.
Uniqueness and convergence do not make an optimizer differentiable at every
constraint switch.

The [stable-face sensitivity certificate](gaussian-entropy-sensitivity.md)
can use a sufficiently tight residual enclosure to prove the optimizer's
active face and bound its target Jacobian, including with redundant or
opposing constraint rows.
