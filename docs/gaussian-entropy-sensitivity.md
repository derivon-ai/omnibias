# Certified target derivatives on stable Gaussian constraint faces

A [certified projected-mean solve](gaussian-entropy-solver.md) locates the
unique global minimizer even when the Gaussian potential is nonconvex away
from the certified posterior domain. Additional face and curvature proofs
can certify the derivative of that exact minimizer with respect to its target.
This is a derivative of the mathematical decision, not of a finite solver run.

The construction supports redundant inequalities and equalities represented
by opposing inequalities. At a genuine active-set switch it refuses an
ordinary derivative unless the required stable-face conditions hold.

## Establishing the face without guessing zero slacks

Write the polyhedron as \(C=\{x:Gx\le h\}\), the target as \(v\), and
\(\mu(x)=\sum_j\pi_j(x)a_j\) for the original Gaussian posterior mean.
The exact optimizer satisfies

\[
x_* = \operatorname{Proj}_C(q_*),\qquad q_*=sv+\mu(x_*).
\]

A residual certificate supplies an exactly feasible sample
\(x_0=\operatorname{Proj}_C(q_0)\) and an enclosure of \(x_*\).
Let \(J\) contain **all** original inequalities tight at \(x_0\).
Choose independent rows \(E\) spanning these normals and form the exact
orthogonal tangent projector

\[
R=I-E^\top(EE^\top)^{-1}E.
\]

A nonnegative projection multiplier alone is insufficient to establish a
stable face. The compiler instead seeks \(\epsilon>0\) such that the exact
projection of \(q_0-\epsilon\sum_{j\in J}G_j^\top\) is still \(x_0\).
Its nonnegative KKT multipliers, with \(\epsilon\) added to every tight
row, give a strictly positive vector \(\lambda_0\) satisfying

\[
q_0-x_0=G_J^\top\lambda_0.
\]

For an enclosed projection source \(q\), correct the coefficients of the
independent rows by \((EE^\top)^{-1}E(q-q_0)\), leaving the other
coefficients unchanged. The resulting candidate

\[
x(q)=x_0+R(q-q_0)
\]

obeys every tight equation exactly. If all corrected tight multipliers stay
strictly positive and all other slacks stay strictly positive, then
\(q-x(q)\in N_C(x(q))\). Thus \(x(q)=\operatorname{Proj}_C(q)\).
Applying this to an enclosure of the true \(q_*\) proves the optimizer's
face and local affine projection formula. A slack interval merely containing
zero is never used as evidence of equality.

Only the independent basis is inverted. The positive witness uses every
tight original row, so opposing rows and redundant rows are allowed. For
example, both signs of an equality normal can have positive coefficients;
these coefficients are a nonunique normal-cone representation, not unique
equality multipliers.

## A convergent derivative series

At the exact optimizer, let \(\Sigma_*\) be the original center covariance.
The posterior-domain certificate supplies
\(\|\Sigma_*\|\le B<2\nu s\). Differentiating the fixed-point equation
on the stable face gives

\[
J=\frac{\partial x_*}{\partial v}
=sR+\frac{R\Sigma_*J}{2\nu s},\qquad RJ=J.
\]

Define

\[
T=\frac{R\Sigma_*R}{2\nu s},\qquad
q\ge\|T\|,\quad q<1.
\]

The implementation may tighten the source bound \(B/(2\nu s)\) using a
rigorous upper bound on \(\operatorname{tr}T\), since \(T\) is positive
semidefinite. Then

\[
J=s\sum_{k=0}^{\infty}T^kR,
\qquad
\left\|J-s\sum_{k=0}^{n-1}T^kR\right\|
\le\tau_n:=\frac{s q^n}{1-q}.
\]

Original-posterior interval moments enclose the finite matrix polynomial.
No rational approximation is substituted for the true posterior. The tail
acts only on the tangent space, giving the sharper coordinate bounds

\[
|\mathrm{tail}_{ij}|\le\tau_n\sqrt{R_{ii}R_{jj}},\qquad
|(\mathrm{tail}\,g)_i|\le\tau_n\sqrt{R_{ii}}\,\|Rg\|.
\]

Exact rational projection of a cotangent precedes numerical conversion.
Every exact constraint-normal cotangent therefore has an exactly zero VJP.
Increasing the number of terms reduces the analytic tail, but interval
rounding and dependence can still limit the final enclosure's width.

## Why this is a derivative of the global decision

The original Hessian at the optimizer is

\[
H_* = I/s-\Sigma_*/(2\nu s^2)\succeq(1-q_{\rm source})I/s>0.
\]

Together with the stable face, this gives a nonsingular tangent KKT system
and a smooth local branch. The conclusion does not require the optimizer to
be interior to the recorded physical box: positivity at the exact root and
Gaussian smoothness imply positivity in some ambient neighborhood.

For bounded nearby targets the Gaussian objective remains uniformly
coercive. Uniqueness at the original target therefore forces all nearby
global minimizers into this local branch neighborhood. Consequently the
branch remains the unique global minimizer there. The certificate proves
that some such neighborhood exists; it does not give its radius or a
finite-step Taylor remainder.

For any basis \(Z\) of the face tangent space, the equivalent formula is
\(J=Z(Z^\top H_*Z)^{-1}Z^\top\). The implementation uses the convergent
series above instead of a new inverse factorization. Smooth KKT sensitivity
is classical; see the local differentiable-solution theorem and
inverse-Hessian formula in [Zavala, Theorem 2](https://www.mcs.anl.gov/uploads/cels/papers/P1644.pdf).
The interval and exact-geometry checks bind that argument to the declared
Gaussian model and polyhedron; no new general sensitivity theorem is claimed.

## APIs and a constrained nonconvex example

`certify_gaussian_entropy_face(residual, *, max_trials=32)` in
`omnibias.core.verified.gaussian_entropy_face` returns a
`GaussianEntropyFace` when its exact witness and interval margins succeed.
It binds the source residual, all tight `active_indices`, independent
`basis_indices`, exact `normal_rows`, `gram_inverse` and `tangent_projector`.
It also records `optimizer_bounds`, original `posterior` and `covariance`,
`projection_input_bounds`, `projection_multipliers`, `slacks`,
`witness_epsilon`, `witness_multipliers` and `strong_convexity_bound`.
Projection multipliers have the scaling of \(q-x\); objective KKT
multipliers are divided by \(s\).

`certify_gaussian_entropy_sensitivity(face, *, terms=16)` in
`omnibias.core.verified.gaussian_entropy_sensitivity` returns a
`GaussianEntropySensitivity` containing:

| Field | Meaning |
| --- | --- |
| `face`, `terms` | Bound face proof and finite series length. |
| `projected_covariance`, `contraction_bound` | Enclosure of \(R\Sigma_*R\) and exact upper bound \(q<1\). |
| `polynomial` | Interval enclosure of \(\sum_{k=0}^{n-1}T^k\). |
| `tail_norm_bound` | Outward rational upper bound on \(s q^n/(1-q)\). |
| `target_jacobian` | Full interval enclosure of the true local target Jacobian. |

`.vjp(cotangent)` accepts exact integer/`Fraction` entries and encloses the
true target VJP. The Jacobian is symmetric, so the same contraction also
represents a target directional derivative. The cotangent is fixed; uncertainty
or dependence in an upstream loss gradient requires its own treatment.

```python
from omnibias.core.verified.gaussian_entropy_face import (
    certify_gaussian_entropy_face,
)
from omnibias.core.verified.gaussian_entropy_sensitivity import (
    certify_gaussian_entropy_sensitivity,
)
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

model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -20, -20))
# Two representations of the same inequality x[0] <= 0.
projector = compile_polyhedral_projector(2, ((1, 0), (2, 0)), (0, 0))
assert projector is not None
domain = certify_gaussian_posterior_domain(model, projector, (1, 0))
assert domain is not None
unique = certify_gaussian_entropy_unique(domain)
assert unique is not None and model.contraction_bound == 4
solution = solve_gaussian_entropy(unique, tolerance=1e-12, max_iterations=12)
assert solution is not None and solution.status == "tolerance"
face = certify_gaussian_entropy_face(solution.residual)
assert face is not None and face.active_indices == (0, 1)
sensitivity = certify_gaussian_entropy_sensitivity(face, terms=16)
assert sensitivity is not None
normal_vjp = sensitivity.vjp((2**80, 0))
assert all(axis.lo == axis.hi == 0 for axis in normal_vjp)
```

The actual root lies on \(x_1=0\), but the current posterior-domain compiler
uses a conservative box with width in both coordinates. The stronger biases
in this example make its covariance test succeed on that actual box; no
lower-dimensional source-box assumption is needed.

These factories may refuse when the optimizer enclosure is too broad,
the face witness is unresolved, or resource or arithmetic checks fail.
Refusal does not prove nondifferentiability. In particular, uniqueness alone
is insufficient: a one-mode Gaussian with \(x\ge0\) has the unique decision
\(x_*(v)=\max(sv+a,0)\), which has no ordinary target derivative at its
switch. This API provides neither an active-set search, derivatives with
respect to geometry or model parameters, nor a certified perturbation radius.
No speed or general activation-independence claim follows from the series.

The [bias sensitivity extension](gaussian-entropy-bias.md) reuses this
certificate to enclose joint target and original log-weight derivatives.
It preserves exact common-bias invariance and relates the bias response
to the linearized Burgers equation for the underlying Gaussian field.
