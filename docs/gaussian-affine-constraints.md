# Gaussian inverse decisions under affine constraints

An intersection of hyperplanes can impose exact constraints on a
[Gaussian inverse decision](gaussian-heat-inverse.md). For this family,
restricting the objective to an affine space produces another Gaussian
mixture with explicitly transformed centers and biases. Existing inverse
certificates then prove a unique constrained global minimum under a
condition involving only the center separation along the feasible space.
The unconstrained objective need not be convex.

This construction uses the specified Gaussian potential. Hyperplane
geometry alone does not make an arbitrary activation interchangeable or
turn every constrained problem into a linear system.

## Exact restriction and uniqueness

Let the ambient dimension be \(n\), and let the consistent constraints be
\(Cx=d\). Redundant rows are allowed. Write \(P\) for the orthogonal
projector onto \(\ker C\), \(N=I-P\), and \(x_0\) for the minimum-norm
feasible point. Thus

\[
P^T=P=P^2,\qquad Px_0=0,\qquad
\{x:Cx=d\}=x_0+\operatorname{range}P.
\]

Independent constraint rows give
\(P=I-C^T(CC^T)^{-1}C\) and
\(x_0=C^T(CC^T)^{-1}d\). Exact row reduction first removes consistent
redundancy. These quantities remain rational for rational input; no rounded
orthonormal basis is needed.

For fixed heat age \(s>0\) and viscosity \(\nu>0\), define

\[
c_j=\|Na_j-x_0\|_2^2,\qquad a'_j=x_0+Pa_j,
\qquad b'_j=(b_j-b_0)-\frac{c_j-c_0}{4\nu s}.
\]

The two displacement components are orthogonal, so every feasible \(x\)
satisfies

\[
\|x-a_j\|_2^2=\|x-a'_j\|_2^2+c_j.
\]

Consequently the transformed density equals the original density times
\(\exp(-b_0+c_0/(4\nu s))\) on the feasible space. Both use the same
ambient Gaussian normalization. Their potentials
\(\Phi=-2\nu\log\tau\) differ there by the constant
\(2\nu b_0-c_0/(2s)\). This common bias gauge changes neither the
minimizer nor its target derivatives.

Let

\[
D_T=\max_{i,j}\|P(a_i-a_j)\|_2,\qquad
q_T=\frac{D_T^2}{8\nu s}.
\]

If \(q_T<1\), the transformed mixture has a unique global inverse at
every target. Its inverse at \(w=Pv\) satisfies

\[
X=sw+\sum_jp_ja'_j=x_0+sw+P\sum_jp_ja_j,
\]

so it is feasible. On the feasible space, its stationarity equation is
exactly \(P(\nabla\Phi(X)-v)=0\). Every tangent Hessian eigenvalue
lies in \([(1-q_T)/s,1/s]\); hence \(X\) is the unique global solution
of

\[
\underset{Cx=d}{\operatorname{minimize}}\quad \Phi(x,s)-v\cdot x.
\]

The criterion depends on tangent center distances, not on constraint
offsets or biases. Adding constraints contracts these distances further.
When \(q_T\ge1\), this sufficient test is inconclusive. Full-rank
constraints leave the single feasible point \(x_0\), and \(P=0\).

## Exact reduction and a certified feasible decision

```python
from fractions import Fraction

from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.inverse_gaussian_affine import inverse_gaussian_affine

source = GaussianHeatMap(
    ((-1, -2), (1, 2)), (Fraction(9, 4), Fraction(1, 4)),
    viscosity=1, time=1,
)
# Both rows impose the same hyperplane y = 1.
restriction = restrict_gaussian_heat(source, ((0, 1), (0, 2)), (1, 2))
assert restriction is not None
assert restriction.constraint_rank == 1
assert restriction.anchor == (0, 1)
assert source.contraction_bound == Fraction(5, 2)
assert restriction.model.contraction_bound == Fraction(1, 2)
assert restriction.model.centers == ((-1, 1), (1, 1))
assert restriction.model.biases == (0, 0)

decision = inverse_gaussian_affine(restriction, (0, 0))
assert decision.certificate is not None
proof = decision.certificate
assert proof.feasible_point[1] == 1
assert proof.enclosure[0].lo <= 0 <= proof.enclosure[0].hi
assert proof.enclosure[1].lo <= 1 <= proof.enclosure[1].hi
assert proof.objective_gap_bound >= 0
```

The exact optimum is \((0,1)\): the transformed posterior is uniform
there. The original ambient Hessian at this point is
\(\begin{pmatrix}1/2&-1\\-1&-1\end{pmatrix}\), which has a negative
eigenvalue. Its tangent restriction is the positive scalar \(1/2\).
Thus this example checks more than a conservative failure of the ambient
diameter test: the objective actually has negative ambient curvature.

`restrict_gaussian_heat` accepts exact integers and `Fraction` values for
the constraint matrix and right-hand side. Its frozen
`GaussianAffineRestriction` retains `source_model`, the transformed
`model`, `constraints`, `rhs`, `constraint_rank`, `anchor`, `projector`,
`normal_cost_differences`, and its `certificate`.
`project_target` computes \(Pv\); `project_point` computes \(x_0+Px\)
exactly. Malformed input raises, inconsistent constraints raise
`InconsistentAffineConstraints`, and resource exhaustion returns `None`.
Normal cost differences are computed directly, avoiding squares of a
potentially huge common normal offset. A rank-zero restriction can retain
the original model unchanged instead of applying the optional common bias
gauge above.

`inverse_gaussian_affine` returns `status`, `iterations`, `tolerance_met`,
and an optional `certificate`. Status is `certified`, `not_certified`, or
`outside_contraction_regime`. A certified result can have
`tolerance_met=False`; its bounds remain valid but may be wide.
The certificate retains the underlying
`root_certificate`, a physical root `enclosure`, its `error_bound`, an
exact rational `feasible_point`, and an exact `objective_gap_bound`.
For an exactly affine restricted map, `root_certificate` is `None` and
the optimum is computed exactly with zero error and gap.
A supplied numerical scaled point \(y=x/s\) can instead be checked by
`certify_gaussian_affine_inverse(restriction, target, scaled_point)`.

The feasible point is an exact point on all declared hyperplanes. An
axis-aligned interval enclosure generally also contains infeasible points;
its guarantee concerns the unique feasible minimizer contained inside it.
Since the objective restricted to the affine space is \(1/s\)-smooth,
a certified feasible-point distance \(\delta\) gives

\[
0\le F_v(\widehat x)-F_v(X)\le\frac{\delta^2}{2s},
\qquad F_v(x)=\Phi(x,s)-v\cdot x.
\]

The gap concerns the original constrained objective. Its proof does not
require evaluating a logarithm of the density or retaining the common
bias gauge.

## Differentiated decisions and constraint parameters

At a feasible inverse point let \(\mu=\sum_jp_ja_j\),
\(r_j=P(a_j-\mu)\), and

\[
C_T=\sum_jp_jr_jr_j^T,\qquad
H_T=I-\frac{C_T}{2\nu s}.
\]

The ambient Jacobian of the transformed inverse is \(J'=sH_T^{-1}\).
The constrained map first projects its input target, so its true target
Jacobian is

\[
S=D_vX=PJ'P=sPH_T^{-1}P.
\]

It vanishes in normal directions and is positive definite on the tangent
space. In general it is not \(P(D_vX_{\rm original})P\): the original
unconstrained inverse may not even be uniquely defined. The
[target derivative certificate below](#certified-target-derivatives-from-a-forward-approximation)
uses this identity to check numerical forward and derivative proposals
against the true constrained map. It accounts for projection rounding,
including when a large normal component nearly cancels.

Constraint parameters require a different derivative. For locally
independent rows \(C\), write \(B=\nabla_x^2\Phi(X,s)\) and choose
the multiplier convention
\(\nabla\Phi(X,s)-v+C^T\lambda=0\). The differential satisfies

\[
\begin{pmatrix}B&C^T\\C&0\end{pmatrix}
\begin{pmatrix}\delta X\\\delta\lambda\end{pmatrix}
=
\begin{pmatrix}
\delta v-(\delta C)^T\lambda-\delta_{\rm model}\nabla\Phi\\
\delta d-(\delta C)X
\end{pmatrix}.
\]

Tangent positive definiteness makes this KKT matrix nonsingular even
when \(B\) is indefinite. With fixed \(C\) and fixed model, put
\(R=C^T(CC^T)^{-1}\); then

\[
\delta X=S\,\delta v+(R-SBR)\,\delta d.
\]

In the preceding example, moving the offset in \(y=d\) gives
\(\partial_dX=(2,1)\) at \(d=1\). Merely projecting a target gradient
does not include this motion of the feasible space. Redundant
representations need an independent-row or multiplier-gauge convention;
these differentiability statements do not cross changes of constraint
rank.

### Numerical trainable layers

Torch and JAX expose `gaussian_heat_inverse_constrained` in their
`gaussian_constrained` modules. They project through a reduced QR
factorization and call the existing numerical implicit inverse. Numerical
constraints must have full row rank; the exact core reduction, in contrast,
also accepts redundant consistent rows.

```python
import torch

from omnibias.torch.gaussian_constrained import gaussian_heat_inverse_constrained

target_tensor = torch.zeros(2, dtype=torch.float64, requires_grad=True)
offset_tensor = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
age_tensor = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
numerical = gaussian_heat_inverse_constrained(
    target_tensor,
    torch.tensor([[-1.0, -2.0], [1.0, 2.0]], dtype=torch.float64),
    torch.tensor([2.25, 0.25], dtype=torch.float64),
    torch.tensor([[0.0, 1.0]], dtype=torch.float64),
    offset_tensor, time=age_tensor,
)
assert bool(numerical.converged)
target_grad, offset_grad, age_grad = torch.autograd.grad(
    numerical.value[0], (target_tensor, offset_tensor, age_tensor),
)
assert torch.allclose(target_grad, torch.tensor([2.0, 0.0], dtype=torch.float64))
assert torch.allclose(offset_grad, torch.tensor([2.0], dtype=torch.float64))
assert torch.allclose(age_grad, torch.tensor(-2.0, dtype=torch.float64))
```

The age gradient includes the time dependence of the normal-energy bias
shift. It is the derivative of the original constrained problem. The
returned `value`, `converged`, `residual_bound`, `contraction_bound`,
`feasibility_residual`, `constraint_reciprocal_condition`, and
`constraints_valid` are numerical results and diagnostics, not interval
certificates. In particular, QR and projection rounding are not included
in a rigorous feasibility bound. Nearly dependent or redundant numerical
rows are refused; this does not prove that the exact constraints are
inconsistent. The exact core checker can independently certify numerical
proposals.

## The forced inverse PDE

Hold the original centers, biases, viscosity, target, and constraints
fixed while varying heat age. The transformed biases are then
time-dependent:
\(\partial_s b'_j=(c_j-c_0)/(4\nu s^2)\). Applying the fixed-bias
[inverse evolution](gaussian-inverse-evolution.md) to the transformed
model alone therefore misses a term.

Define the tangent third-moment trace and normal-cost covariance

\[
T_T=\sum_jp_jr_j\|r_j\|_2^2,\qquad
K=\sum_jp_jr_j(c_j-c_0)=\operatorname{Cov}_p(Pa,c).
\]

The constant cost gauge cancels because \(\sum_jp_jr_j=0\).
The original constrained inverse obeys

\[
\boxed{\quad
X_s=Pv+\frac{1}{4\nu s^2}H_T^{-1}(T_T+K).
\quad}
\]

To verify it directly, differentiate the projected posterior mean at
fixed \(x\). At an inverse point,

\[
\partial_s\mu'\big|_x
=\frac{T_T+K-2sC_TPv}{4\nu s^2}.
\]

Differentiating \(X=sPv+\mu'(X,s)\) gives the displayed formula.
The vector \(K\) uses one scalar cost per center and costs \(O(mn)\)
arithmetic; no full third-order tensor is needed.

For the hodograph form, let \(G=H_TP/s\), the inverse of \(S\) on
the tangent space and its Moore–Penrose inverse in ambient coordinates.
Then

\[
X_s=Pv+\nu\sum_{\alpha,\beta}(GG^T)_{\alpha\beta}
            \partial_{v_\alpha}\partial_{v_\beta}X
       +\frac{H_T^{-1}K}{4\nu s^2}.
\]

Unlike the unconstrained case, the ambient target Jacobian is singular;
an ordinary ambient matrix inverse would be incorrect. The diffusion
contraction equals \(H_T^{-1}T_T/(4\nu s^2)\).

Writing \(\Delta c=\max_jc_j-\min_jc_j\), directional covariance
bounds yield

\[
\|T_T\|_2\le\frac{D_T^3}{4},\qquad
\|K\|_2\le\frac{D_T\Delta c}{4},\qquad
\|X_s-Pv\|_2\le
\frac{D_T^3+D_T\Delta c}{16\nu s^2(1-q_T)}.
\]

For the example, \(c=(9,1)\), \(T_T=0\), \(K=(-4,0)\), and
\(H_T=\operatorname{diag}(1/2,1)\). Thus the true time derivative is
\((-2,0)\); incorrectly freezing the projected biases gives zero.
When all normal costs coincide, the bias change is a common gauge and
the forcing vanishes.

```python
from omnibias.core.verified.gaussian_affine_evolution import certify_gaussian_affine_evolution
from omnibias.core.verified.interval import Interval

evolution = certify_gaussian_affine_evolution(
    restriction, (Interval.point(0.0), Interval.point(0.0)),
)
assert evolution is not None
assert evolution.normal_cost_covariance[0].lo <= -4 <= evolution.normal_cost_covariance[0].hi
assert evolution.time_derivative[0].lo <= -2 <= evolution.time_derivative[0].hi
assert evolution.time_derivative[1].lo <= 0 <= evolution.time_derivative[1].hi
```

`GaussianAffineEvolution` retains `restriction`, `target_box`,
`projected_target_box`, `spatial_enclosure`, the constrained
`target_jacobian`, `normal_cost_covariance`, `diffusion`, `normal_forcing`,
`time_derivative`, `correction_norm_bound`, `base_evolution`, and its
`certificate`. It combines projected-model enclosures with the additional
posterior covariance. The `diffusion` field is the homogeneous tangent
diffusion; `normal_forcing` encloses the missing \(K\) term.

The certificate holds at every target in the specified box at one fixed
heat age. It differentiates the original fixed-bias family and fixed
constraints; it does not enclose an interval of ages. Dense core projectors
and Jacobians remain subject to resource budgets. In particular, fixed-bias
aging-series derivatives of `restriction.model` describe a different
evolution unless the normal costs coincide. They must not be substituted
for the original constrained time derivative.

### The forward trace PDE

There is an independent forward check of the forcing sign. Let
\(r=\operatorname{rank}C\), and let \(\tau_A\) be the restriction of
the original ambient heat density to the affine set, retaining its
\((4\pi\nu s)^{-n/2}\) prefactor. This is a trace, without renormalizing
its integral over the feasible space. Differentiating each Gaussian gives

\[
\partial_s\tau_A=\nu\Delta_T\tau_A+
 \left(\frac{\mathbb E_p c}{4\nu s^2}-\frac{r}{2s}\right)\tau_A.
\]

Indeed, its time derivative contains the full squared distance, while its
tangent Laplacian contains only the tangent distance and dimension
\(n-r\). For \(\Phi_A=-2\nu\log\tau_A\) and
\(U=P\nabla\Phi_A\), this implies

\[
\partial_s\Phi_A+\tfrac12\|U\|^2
=\nu\Delta_T\Phi_A-\frac{\mathbb E_p c}{2s^2}+\frac{\nu r}{s},
\]

\[
\boxed{\quad
U_s+(D_TU)U=\nu\Delta_TU-\frac{K}{4\nu s^3}.
\quad}
\]

Here \(D_TU\) denotes the tangent derivative, not the center diameter.
The posterior identity \(\nabla_T\mathbb E_p c=K/(2\nu s)\)
fixes the coefficient. The rank term disappears after the spatial
gradient. Inverting this forced Burgers equation on the tangent space
recovers the positive \(H_T^{-1}K/(4\nu s^2)\) term above.

## Certified constrained forward approximation

`omnibias.core.verified.gaussian_affine_aging` evaluates constrained
predictors and higher-order aging series for the original fixed-bias
family. Once the exact restriction is compiled, these evaluations use
forward coefficient recurrences without inverse iteration or a matrix
solve. The normal-cost contribution is essential: applying the
unconstrained fixed-bias aging API to `restriction.model` would describe
a different age dependence.

Put \(u_j=Pa_j\), \(w=Pv\), and define the limiting posterior using
the **original** biases:

\[
p_j^\infty\propto\exp\left(b_j+\frac{u_j\cdot w}{2\nu}\right),
\qquad \mu_\infty=\sum_jp_j^\infty u_j,\qquad A_j=u_j-\mu_\infty.
\]

Write \(\Delta c=\max c_j-\min c_j\),
\(V=D_T^2+\Delta c\), \(Q=V/(8\nu s)\), and

\[
T_\infty=\mathbb E_\infty[A\|A\|^2],\qquad
K_\infty=\mathbb E_\infty[A(c-c_0)].
\]

Two explicit approximations are

\[
P_0=sw+x_0+\mu_\infty,\qquad
P_1=P_0-\frac{T_\infty+K_\infty}{4\nu s}.
\]

For \(q_T<1\), their errors satisfy, uniformly over every real target,

\[
\|X-P_0\|_2\le\frac{D_TQ}{2},\qquad
\|X-P_1\|_2\le\frac{D_TQ(Q+q_T)}2\le D_TQ^2.
\]

These real bounds do not require \(Q<1\), although they can be loose
when the normal-cost range is large. To prove them, freeze the true
tangent mean \(m_*=X-sw-x_0\). The scalar costs
\(g_j=\|u_j-m_*\|^2+c_j\) have range at most \(V\). Interpolate
positive weights proportional to
\(p_j^\infty\exp[-\theta g_j/(4\nu s)]\), \(0\le\theta\le1\).
The cross-covariance norm is at most \(D_TV/4\), proving the first
bound. The second mean derivative is
\((4\nu s)^{-2}\mathbb E[(u-\mathbb E u)(g-\mathbb E g)^2]\);
its norm is at most \((4\nu s)^{-2}D_TV^2/4\), so the Taylor
remainder is at most \(D_TQ^2/2\). Replacing \(m_*\) by
\(\mu_\infty\) in the linear term contributes at most
\(q_T\|m_*-\mu_\infty\|\), giving the stated second bound.

For all orders, write \(z=1/s\) and
\(h=X-sw-x_0-\mu_\infty\). Completing the Gaussian square gives

\[
h=\frac{\sum_jp_j^\infty A_j
 \exp\{z[A_j\cdot h-(\|A_j\|^2+c_j-c_0)/2]/(2\nu)\}}
 {\sum_jp_j^\infty
 \exp\{z[A_j\cdot h-(\|A_j\|^2+c_j-c_0)/2]/(2\nu)\}}.
\]

For a formal parameter \(t\), set
\(h(t/s)=\sum_{n\ge1}\gamma_nt^n\). The logit coefficients are

\[
\ell_{j,1}=-\frac{\|A_j\|^2+c_j-c_0}{4\nu s},\qquad
\ell_{j,n}=\frac{A_j\cdot\gamma_{n-1}}{2\nu s}\quad(n\ge2).
\]

Let \(e_{j,0}=1\), and propagate
\(e_{j,n}=n^{-1}\sum_{k=1}^n k\ell_{j,k}e_{j,n-k}\).
With \(d_n=\sum_jp_j^\infty e_{j,n}\) and
\(b_n=\sum_jp_j^\infty A_je_{j,n}\), quotient division gives
\(\gamma_0=0\) and
\(\gamma_n=b_n-\sum_{k=1}^n d_k\gamma_{n-k}\).
The explicit factor \(z\) makes each coefficient depend only on previous
ones; no implicit matrix solve is needed in this formal recurrence.

A conservative convergence proof is uniform in the target. For
\(D_T>0\), use the complex ball radius \(R=D_T/2\) and inverse-age
disk radius \(Z=8\nu/(3V)\). At \(|z|\le Z\), put
\(Q_z=|z|V/(8\nu)\le1/3\). On \(\|h\|\le R\), the logit
phase half-width is at most
\(Q_z(1+D_T^2/V)\le2Q_z\). Rotating the common phase bounds the
partition denominator away from zero. The positive-weight pairwise
covariance bound then gives contraction at most

\[
\frac{|z|D_T^2}{8\nu}\sec^2(2Q_z)\le\frac35.
\]

The image of zero has norm at most
\((D_T/2)\tan Q_z\le3D_T/16\). Therefore the map takes the closed
ball into radius \(39D_T/80<R\). The strict margins persist in a
neighborhood of the closed disk, giving a holomorphic fixed point there.
Cauchy's coefficient bound and a geometric sum imply

\[
\|\gamma_n\|_2\le\frac{D_T}{2}(3Q)^n,\qquad
\left\|X-sw-x_0-\mu_\infty-\sum_{n=1}^{N}\gamma_n\right\|_2
\le\frac{D_T}{2}\frac{(3Q)^{N+1}}{1-3Q},\quad Q<\tfrac13.
\]

Only the correction \(h\) is analytic at \(z=0\); the full inverse
contains \(w/z\). If \(D_T=0\), every tangent center coincides and
\(h\equiv0\): the inverse is exactly affine at every age, regardless
of the normal-cost range. That case must be handled before any \(Q\)
test. For nonzero diameter, this disk proof establishes neither convergence
throughout \(q_T<1\) nor an error bound for a floating-point recurrence.

### Enclosing the formula and an exact feasible point

`certify_gaussian_affine_predictor(restriction, target_box, order=1)`
supports orders zero and one under \(q_T<1\), using the real bounds.
`certify_gaussian_affine_aging(restriction, target_box, order=8)` uses the
series bound and requires \(Q<1/3\) when \(D_T>0\). Both handle
\(D_T=0\) exactly before testing \(Q\).

```python
from omnibias.core.verified.gaussian_affine_aging import (
    certify_gaussian_affine_aging,
    certify_gaussian_affine_predictor,
)

aging_source = GaussianHeatMap(
    source.centers, source.biases, viscosity=source.viscosity, time=16,
)
aging_restriction = restrict_gaussian_heat(aging_source, ((0, 1), (0, 2)), (1, 2))
assert aging_restriction is not None
point_target_box = (Interval.point(0.0), Interval.point(0.0))
first_order = certify_gaussian_affine_predictor(aging_restriction, point_target_box)
forward_series = certify_gaussian_affine_aging(aging_restriction, point_target_box, order=8)
assert first_order is not None and forward_series is not None
assert forward_series.aging_parameter == Fraction(3, 32)
assert forward_series.approximation_error_bound < Fraction(1, 10000)
assert forward_series.feasible_point[1] == 1
assert forward_series.feasible_point_error_bound >= forward_series.approximation_error_bound
assert forward_series.feasible_point_objective_gap_bound >= 0
```

Both factories return a frozen `GaussianAffineAging` record or `None`.
It retains `mode` (`"predictor"` or `"series"`), `restriction`,
`target_box`, `projected_target_box`, `order`, `aging_parameter` \(Q\),
`relative_mean`, `correction_coefficients`, and its `certificate`.
`relative_mean` encloses \(\mathbb E_\infty[P(a-a_0)]\), before
adding the transformed reference center.
The coefficient rows enclose the scaled ordinary coefficients
\(\gamma_n\), including a zero row at order zero.

`predictor_enclosure` contains the exact finite formula at every target
in the box, including variation over that box and interval evaluation
error. `approximation_error_bound` is the Euclidean analytic truncation
bound \(\delta\) for the exact formula; `enclosure` adds that error
to cover the true constrained inverse. Since the exact formula is feasible,
`objective_gap_bound` is \(\delta^2/(2s)\). It does not by itself
certify a rounded midpoint or a tensor evaluation of the formula.

For an actual candidate, the record provides `feasible_point`: the exact
rational orthogonal projection of the midpoint of `predictor_enclosure`
onto the declared affine space. If \(r_i\) are the coordinate half-widths
of that enclosure, nonexpansiveness of projection proves

\[
\|\texttt{feasible\_point}-X(v,s)\|_2
\le\delta+\sqrt{\sum_i r_i^2}
\quad\text{for every target }v\text{ in the box}.
\]

`feasible_point_error_bound` is an outward rational bound on this sum;
`feasible_point_objective_gap_bound` is its square divided by \(2s\).
These fields include interval evaluation error and target-box variation.
The candidate exactly satisfies the constraints, while the enclosing
coordinate box can contain infeasible points.

Invalid input raises; failed contraction or series conditions, exhausted
budgets, damaged dependencies, and unavailable finite enclosures return
`None`. A missing required rigorous backend raises. Higher order reduces
the analytic series tail, but accumulated interval widths can limit the
quality of the final enclosure. No age-derivative API for this constrained
series is provided here.

### Trainable forward evaluations

The paired Torch and JAX `gaussian_constrained_aging` modules expose
`gaussian_heat_inverse_constrained_predictor` and
`gaussian_heat_inverse_constrained_aging`. Each takes the original centers,
biases, constraints, viscosity, and age. QR preprocessing supplies the
tangent geometry; the forward formula includes the original normal-cost
term. Numerical constraints require full row rank, as for the implicit
constrained layer above.

```python
from omnibias.torch.gaussian_constrained_aging import gaussian_heat_inverse_constrained_aging

forward_target = torch.zeros(2, dtype=torch.float64, requires_grad=True)
numerical_series = gaussian_heat_inverse_constrained_aging(
    forward_target,
    torch.tensor([[-1.0, -2.0], [1.0, 2.0]], dtype=torch.float64),
    torch.tensor([2.25, 0.25], dtype=torch.float64),
    torch.tensor([[0.0, 1.0]], dtype=torch.float64),
    torch.tensor([1.0], dtype=torch.float64),
    time=16.0, order=8,
)
assert bool(numerical_series.valid)
assert numerical_series.correction_coefficients.shape == (9, 2)
forward_target_gradient, = torch.autograd.grad(numerical_series.value[0], forward_target)
assert bool(torch.isfinite(forward_target_gradient).all())
```

`GaussianConstrainedAgingResult` contains `value`,
`correction_coefficients`, `approximation_error_bound`, `contraction_bound`
\(q_T\), `aging_parameter` \(Q\), `feasibility_residual`,
`constraint_reciprocal_condition`, and `valid`. Coefficients have shape
`(order + 1, ..., dimension)`. The predictor accepts orders zero and one
under \(q_T<1\); the series requires \(Q<1/3\) for nonzero tangent
diameter. Both bypass the \(Q\) test for an exactly affine tangent map.

These are finite floating-point formulas. Autodiff differentiates the
approximation, including its parameter-dependent geometry; it does not
certify derivatives of the true constrained inverse. The numerical error
and feasibility diagnostics omit QR, coefficient, and evaluation rounding.
Use the independent exact core model and its interval checks when a
rigorous feasible-point or objective-gap statement is required.

## Fixed-age constrained approximation

The condition \(Q<1/3\) belongs to the original-family inverse-age
expansion above; it is not necessary for every forward value approximation.
At a fixed queried age, all normal costs can first be included in
\(b'_j(s)\). Applying the existing Gaussian predictor or aging formula
to that transformed family at \(Pv\) then approximates the same
constrained optimum. Its bounds depend only on \(q_T\):

\[
\epsilon_0\le\frac{D_Tq_T}{2},\qquad
\epsilon_1\le D_Tq_T^2,\qquad q_T<1,
\]

\[
\epsilon_N\le\frac{D_T}{2}
 \frac{(3q_T)^{N+1}}{1-3q_T},\qquad N\ge2,\quad q_T<\tfrac13.
\]

These analytic bounds remain valid when \(Q\) is large. They are
bounds for a different finite formula, rather than a proof that its actual
error is always smaller. Outward projection of a target box can also lose
correlations and widen its interval evaluation.

The core factory
`certify_gaussian_affine_approximation(restriction, target_box, order=8)`
in `omnibias.core.verified.gaussian_affine_approximation` composes these
existing value certificates with the exact affine restriction. Orders
zero and one use the real predictor bounds; higher orders use the geometric
bound. Zero tangent diameter is handled exactly.

For a concrete family, take centers \(\pm(1,1)\), zero original biases,
\(\nu=1\), \(s=4\), and the constraint \(y=L\ge0\). Then
\(q_T=1/8\), while \(Q=(1+L)/8\) is arbitrarily large. The
transformed centers are \((-1,L),(1,L)\), and their biases are
\((0,L/4)\). At target zero the tangent inverse solves
\(x=\tanh(x/8+L/8)\). The fixed-age approximation applies throughout
this family.

```python
from omnibias.core.verified.gaussian_affine_approximation import (
    certify_gaussian_affine_approximation,
)

large_cost_source = GaussianHeatMap(((-1, -1), (1, 1)), (0, 0), time=4)
large_cost_restriction = restrict_gaussian_heat(large_cost_source, ((0, 1),), (10,))
assert large_cost_restriction is not None
assert large_cost_restriction.model.contraction_bound == Fraction(1, 8)
assert certify_gaussian_affine_aging(large_cost_restriction, point_target_box, order=8) is None

fixed_age = certify_gaussian_affine_approximation(
    large_cost_restriction, point_target_box, order=8,
)
assert fixed_age is not None
assert fixed_age.approximation_error_bound < Fraction(1, 1000)
assert fixed_age.feasible_point[1] == 10
assert fixed_age.feasible_point_error_bound >= fixed_age.approximation_error_bound
```

`GaussianAffineApproximation` retains `restriction`, `target_box`,
`projected_target_box`, `order`, `correction_coefficients`,
`predictor_enclosure`, `approximation_error_bound`, `objective_gap_bound`,
`enclosure`, `feasible_point`, `feasible_point_error_bound`,
`feasible_point_objective_gap_bound`, `base_approximation`, and its
`certificate`. The optional base artifact records the unconstrained
Gaussian approximation used in the composition; an exactly affine case
needs no such artifact.

The error and gap meanings match those above: the exact feasible formula
has gap at most \(\epsilon_N^2/(2s)\), while the returned rational
feasible point uses the additional interval half-diagonal in its separate
distance and gap bounds. These are uniform statements for every target
in the original box. Invalid data raise; failed contraction, exhausted
budgets, damaged dependencies, or unavailable finite enclosures return
`None`. A missing required rigorous backend raises.

### Coefficients and differentiated formulas

The stored coefficients hold \(b'(s)\) fixed while varying an auxiliary
age. They are not the original-family inverse-age coefficients documented
earlier. For an original-age-dependent finite formula
\(R_N(s)=F_N(s,b'(s))\), the chain rule is

\[
\frac{dR_N}{ds}=\partial_sF_N+
 \sum_j\partial_{b'_j}F_N\frac{c_j-c_0}{4\nu s^2}.
\]

Freezing the transformed biases omits the second term. The Torch and JAX
`gaussian_constrained_approximation` modules implement
`gaussian_heat_inverse_constrained_approximation` from the original
parameters, keeping that transformation in the autodiff graph.

```python
from omnibias.torch.gaussian_constrained_approximation import (
    gaussian_heat_inverse_constrained_approximation,
)

fixed_age_tensor = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
numerical_fixed_age = gaussian_heat_inverse_constrained_approximation(
    torch.zeros(2, dtype=torch.float64),
    torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float64),
    torch.zeros(2, dtype=torch.float64),
    torch.tensor([[0.0, 1.0]], dtype=torch.float64),
    torch.tensor([10.0], dtype=torch.float64),
    time=fixed_age_tensor, order=8,
)
assert bool(numerical_fixed_age.valid)
assert numerical_fixed_age.correction_coefficients.shape == (9, 2)
approximate_age_derivative, = torch.autograd.grad(numerical_fixed_age.value[0], fixed_age_tensor)
assert bool(torch.isfinite(approximate_age_derivative))
```

The numerical result contains `value`, `correction_coefficients`,
`approximation_error_bound`, `contraction_bound`, `feasibility_residual`,
`constraint_reciprocal_condition`, and `valid`. Numerical full-row-rank
requirements and QR feasibility limitations are the same as for the other
constrained tensor functions. No inverse iteration occurs after its
geometry preprocessing.

Runtime autodiff differentiates this complete finite formula. Neither its
value bound nor a frozen-bias derivative certificate bounds the true
original inverse derivative. That requires a separate derivative or residual
certificate. Large transformed biases, cancellation, interval widening,
and finite arithmetic budgets can still limit evaluation; the \(q_T\)
condition alone does not guarantee an implementation will return a result.

## Certified target derivatives from a forward approximation

The finite formula's autodiff output can be used as a proposal for a
certificate of the **true** constrained target derivative. The factory
`certify_gaussian_affine_target_vjp(restriction, target, scaled_point,
upstream, proposal)` in
`omnibias.core.verified.gaussian_affine_sensitivity` checks both a root
residual and an adjoint residual. After exact restriction setup, it requires
neither inverse iteration nor an adjoint factorization. Dense projector
actions remain \(O(n^2)\), and the Gaussian Jacobian action is \(O(mn)\)
for \(m\) centers.

The target consists of exact integers or `Fraction` values. The other
three vectors contain finite binary64 values, with exactly representable
integers also accepted. The root proposal is the scaled physical point
\(x/s\); it need not be feasible. Proposals need not be accurate, or even
come from the same numerical model: their residuals determine the bound.
The declared model, exact target and declared upstream fix what is certified.

This example obtains both proposals from the finite forward formula. Its
large upstream has exact tangent component \((-8/5,4/5)\); forming that
projection with ordinary floating matrix multiplication could lose it.

```python
from omnibias.core.verified.gaussian_affine_sensitivity import (
    certify_gaussian_affine_target_vjp,
)

vjp_source = GaussianHeatMap(
    ((-1, -2), (1, 2), (2, -1)), (0, Fraction(1, 3), Fraction(-1, 4)),
    time=4,
)
vjp_restriction = restrict_gaussian_heat(vjp_source, ((1, 2),), (1,))
assert vjp_restriction is not None
exact_target = (Fraction(1, 8), Fraction(-1, 4))
upstream = (2**53, 2**54 + 4)
tangent_upstream = vjp_restriction.project_target(upstream)
assert tangent_upstream == (Fraction(-8, 5), Fraction(4, 5))

vjp_target = torch.tensor(
    [float(v) for v in exact_target], dtype=torch.float64, requires_grad=True,
)
vjp_forward = gaussian_heat_inverse_constrained_approximation(
    vjp_target,
    torch.tensor(vjp_source.centers, dtype=torch.float64),
    torch.tensor([float(b) for b in vjp_source.biases], dtype=torch.float64),
    torch.tensor([[1.0, 2.0]], dtype=torch.float64),
    torch.tensor([1.0], dtype=torch.float64),
    time=4.0, order=12,
)
assert bool(vjp_forward.valid)
proposal, = torch.autograd.grad(
    vjp_forward.value @ torch.tensor(
        [float(g) for g in tangent_upstream], dtype=torch.float64,
    ),
    vjp_target,
)
vjp = certify_gaussian_affine_target_vjp(
    vjp_restriction, exact_target,
    tuple((vjp_forward.value.detach() / 4).tolist()),
    upstream, tuple(proposal.tolist()),
)
assert vjp is not None
assert vjp.error_bound < Fraction(1, 10**8)
assert vjp.value[0] + 2 * vjp.value[1] == 0
assert all(box.lo <= value <= box.hi for box, value in zip(vjp.enclosure, vjp.value))

normal_vjp = certify_gaussian_affine_target_vjp(
    vjp_restriction, exact_target, (0.0, 0.0),
    (2**53, 2**54), (0.0, 0.0),
)
assert normal_vjp is not None
assert normal_vjp.value == (0, 0)
assert normal_vjp.error_bound == 0
assert normal_vjp.base_certificate is None
```

Here `enclosure` contains \(S^Tg=Sg\), while the exact rational
`value` is the center of its Euclidean error bound. The equality
\(S^T=S\) also makes this a target JVP certificate. Tensor rounding of
the original biases, root and derivative proposal is harmless to the
claim because the certificate reevaluates residuals for the exact core
model. The certificate does not declare those rounded proposals exact.

To justify its rounding terms, let \(a\) be the supplied derivative
proposal, and form exact \(g_T=Pg\), \(a_T=Pa\). Put

\[
\widehat g=\operatorname{round}(g_T),\qquad
z=\operatorname{round}\bigl(a_T+s(I-P)\widehat g\bigr).
\]

The added normal component agrees with \(J'\widehat g\), since \(J'\)
acts as \(sI\) on normal directions. The existing
[Gaussian target VJP certificate](gaussian-heat-inverse.md)
gives \(\|z-J'\widehat g\|_2\le\beta\).
Using \(PJ'=PJ'P=S\) and \(\|S\|\le s/(1-q_T)\) gives

\[
\boxed{\ \|a_T-Sg\|_2\le E
 =\beta+\frac{s}{1-q_T}\|P(\widehat g-Pg)\|_2
       +\|P(z-a)\|_2.\ }
\]

All projections and squared norms are computed with bounded exact rational
arithmetic; square-root bounds and interval endpoints round outward.
The error is tangent, so coordinate \(i\) has radius
\(\sqrt{P_{ii}}\,E\), preserving exact zero rows of the projector.
An enclosing coordinate box may still contain nontangent vectors.

`GaussianAffineTargetVJPCertificate` retains the original and projected
target, upstream and proposal, the scaled point, exact `value`,
`enclosure`, and `error_bound`. Its
`base_error_bound`, `upstream_rounding_error_bound` and
`proposal_rounding_error_bound` record the three terms above;
`coordinate_error_bounds` records the coordinate radii.
It also retains `restriction`, an optional `base_certificate`, and
its own `certificate`. If \(Pg=0\), the result is exactly zero.
If \(q_T=0\), it is exactly \(sPg\). These cases require no root,
posterior or adjoint evaluation, and the returned `value` replaces the
untrusted `projected_proposal`.

The condition \(q_T<1\) is required even for these exact cases.
Invalid input raises; failed contraction, damaged dependencies, exhausted
budgets or unavailable finite bounds return `None`. Nontrivial
transcendental evaluation requires the rigorous backend. This certifies a
derivative at one exact target for a fixed declared upstream. If that
upstream approximates a loss gradient, its uncertainty needs a separate
bound. Derivatives of constraints, offsets, centers, biases, viscosity and
age are outside this target-only claim.

## Certified derivatives of the original parameters

The target adjoint also determines derivatives with respect to the original
centers, biases, viscosity, heat age, constraint matrix and offsets.
Here the constraints must have full row rank. The exact restriction may
accept redundant rows, but arbitrary independent changes to their entries
can change rank or consistency; they do not define the parameter map
certified in this section.

Use `compile_gaussian_affine_adjoint(restriction)` from
`omnibias.core.verified.gaussian_affine_adjoint` to compile reusable exact
normal geometry. Then
`certify_gaussian_affine_parameter_vjp(geometry, target, scaled_point,
upstream, proposal)` from
`omnibias.core.verified.gaussian_affine_parameters` accepts the same exact
target and untrusted binary64 root/target-adjoint proposals as the
target-only checker. The supplied adjoint is for the target derivative;
the remaining parameter blocks are derived and enclosed by the checker.

Write \(x=X(v)\), \(A=D_xU(x)\), and \(z=(D_vX)^Tg\), with the
declared upstream \(g\) held fixed. The primal and adjoint KKT systems are

\[
U(x)-v+C^T\lambda=0,\qquad Cx=d,
\qquad Az+C^T\eta=g,\qquad Cz=0.
\]

Their matrix is nonsingular because \(A\) is positive definite on the
tangent space and \(C\) has full row rank. Positive definiteness of the
ambient \(A\) is unnecessary. Compile the exact matrix
\(L_C=(CC^T)^{-1}C\) once. Then the multipliers are

\[
\lambda=L_C\left[v+\frac{\mu-x_0}{s}\right],
\qquad
\eta=L_Cg+\frac{L_C\operatorname{Cov}(a)z}{2\nu s^2}.
\]

These identities use \(L_Cx=L_Cx_0\) and \(L_Cz=0\). Thus the
already certified root and target adjoint suffice; no further KKT or
adjoint factorization is needed at each query.

For the scalar output \(\mathcal L=g^TX\), let
\(r_j=a_j-\mu\), \(\rho_j=\|x-a_j\|^2\), and
\(B_\rho=\operatorname{Cov}(a,\rho)\). Differentiating the original
KKT equations gives

\[
\nabla_v\mathcal L=z,\qquad
\nabla_d\mathcal L=\eta,\qquad
\nabla_C\mathcal L=-\lambda z^T-\eta x^T,
\]

\[
\partial_{b_j}\mathcal L=\frac{p_j}{s}\,z\cdot r_j,
\qquad
\nabla_{a_j}\mathcal L=
 \frac{p_jz}{s}+
 \frac{p_j(z\cdot r_j)(x-a_j)}{2\nu s^2},
\]

\[
\partial_\nu\mathcal L=\frac{z\cdot B_\rho}{4\nu^2s^2},
\qquad
\partial_s\mathcal L=
 \frac{z\cdot v}{s}+\frac{z\cdot B_\rho}{4\nu s^3}.
\]

Indeed, the Gaussian score is
\(\ell_j=b_j-\rho_j/(4\nu s)\), and
\(\delta\mu=\sum_jp_j\delta a_j+
\sum_jp_j(a_j-\mu)\delta\ell_j\).
Together with \(U=(x-\mu)/s\), this yields all model derivatives.
In the time formula, \(z\cdot U=z\cdot v\) follows from \(Cz=0\).
These are derivatives of the original model, including its normal-cost
dependence; differentiating a projected model with its biases frozen
would give different parameter derivatives.

The earlier example with negative ambient curvature has exact root
\(x=(0,1)\). For \(g=(1,3)\), its multipliers and target adjoint are
\(\lambda=-1\), \(\eta=5\), and \(z=(2,0)\).
Here we compile the independent representation \(y=1\); the redundant
two-row representation used earlier is deliberately not accepted for
arbitrary constraint-parameter differentiation.

```python
from omnibias.core.verified.gaussian_affine_adjoint import (
    compile_gaussian_affine_adjoint,
)
from omnibias.core.verified.gaussian_affine_parameters import (
    certify_gaussian_affine_parameter_vjp,
)

parameter_restriction = restrict_gaussian_heat(source, ((0, 1),), (1,))
assert parameter_restriction is not None
adjoint_geometry = compile_gaussian_affine_adjoint(parameter_restriction)
assert adjoint_geometry is not None
assert adjoint_geometry.left_inverse == ((0, 1),)

parameter_vjp = certify_gaussian_affine_parameter_vjp(
    adjoint_geometry, (0, 0), (0.0, 1.0), (1.0, 3.0), (2.0, 0.0),
)
assert parameter_vjp is not None
assert parameter_vjp.rhs_gradient[0].lo <= 5 <= parameter_vjp.rhs_gradient[0].hi
for box, exact in zip(parameter_vjp.constraint_gradient[0], (2, -5)):
    assert box.lo <= exact <= box.hi
for box, exact in zip(parameter_vjp.bias_gradient, (-1, 1)):
    assert box.lo <= exact <= box.hi
for row, exact_row in zip(
    parameter_vjp.center_gradient,
    ((Fraction(1, 2), Fraction(-3, 2)), (Fraction(1, 2), Fraction(-1, 2))),
):
    assert all(box.lo <= exact <= box.hi for box, exact in zip(row, exact_row))
assert parameter_vjp.viscosity_gradient.lo <= -2 <= parameter_vjp.viscosity_gradient.hi
assert parameter_vjp.time_gradient.lo <= -2 <= parameter_vjp.time_gradient.hi

# Reuse the same exact geometry with a purely normal upstream.
normal_parameters = certify_gaussian_affine_parameter_vjp(
    adjoint_geometry, (0, 0), (0.0, 1.0), (0.0, 1.0), (0.0, 0.0),
)
assert normal_parameters is not None
assert normal_parameters.primal_multiplier is None
assert normal_parameters.posterior == ()
assert normal_parameters.rhs_gradient[0].lo == normal_parameters.rhs_gradient[0].hi == 1
assert normal_parameters.constraint_gradient[0][1].lo <= -1 <= normal_parameters.constraint_gradient[0][1].hi
```

`GaussianAffineAdjointGeometry` retains the restriction, exact
`left_inverse` and its certificate. Its compiler replays both
\(L_CC^T=I\) and \(C^TL_C=I-P\) exactly. Empty constraints are valid;
redundant rows raise `ValueError`. Geometry compilation alone does not
assert contraction or certify a decision.

The returned `GaussianAffineParameterVJP` retains `geometry`,
`target_certificate`, an optional `root_certificate`, and
`root_enclosure`. Its interval outputs are `target_gradient`,
`center_gradient`, `bias_gradient`, `constraint_gradient`,
`rhs_gradient`, `viscosity_gradient` and `time_gradient`.
It also retains `posterior`, `primal_multiplier`,
`adjoint_multiplier` and its own `certificate`.
An exactly affine restricted map has an exact root and no root certificate.
An omitted primal multiplier means it was unnecessary, not that it is zero.

The posterior can be evaluated through the transformed logits on the
root enclosure: at the true feasible root they equal the original
posterior. Original centers must still be used in the formulas above.
For stable distance differences, put \(k_j=P(a_j-a_0)\). At a feasible
point,

\[
\rho_j-\rho_0=
\|k_j\|^2-2k_j\cdot(x-a_0)+(c_j-c_0).
\]

The common squared distance cancels from \(B_\rho\). Relative centers,
exact normal-cost differences, and exact \(L_Cg\) avoid introducing
rounding loss from large common offsets or normal upstream components.
Outward evaluation of these formulas over certified root and adjoint
boxes encloses the true parameter derivatives, even though the boxes
also contain points that do not satisfy the KKT equations.

An exactly normal upstream has \(z=0\). Its model and target derivatives
vanish, but \(\eta=L_Cg\) and
\(\nabla_C\mathcal L=-\eta x^T\) can be nonzero. This branch needs no
primal multiplier or extra posterior evaluation; obtaining the root may
still require a root residual check. Zero tangent diameter with
\(Pg\ne0\) is different: individual-center derivatives still depend on
the posterior. Full-rank square constraints fix \(x=C^{-1}d\) exactly
and permit a wholly algebraic parameter certificate.

As consistency checks, the formulas satisfy the bias-gauge, translation
and scale identities

\[
\sum_j\partial_{b_j}\mathcal L=0,\qquad
\sum_j\nabla_{a_j}\mathcal L+C^T\nabla_d\mathcal L=g,\qquad
s\partial_s\mathcal L-\nu\partial_\nu\mathcal L=z\cdot v.
\]

All claims are first derivatives at one declared model and target. They
hold the supplied upstream fixed. A loss that explicitly depends on the
parameters requires its additional direct derivative; uncertainty in a
numerically computed loss upstream also needs a separate bound. This
does not certify higher derivatives or differentiation through a change
of constraint rank. Interval outputs can lose correlations and need not
themselves satisfy the exact gradient identities above. No component
accuracy tolerance is promised. The query requires \(q_T<1\); invalid
input raises, while failed contraction, damaged dependencies, exhausted
budgets and nonfinite enclosures return `None`. Required rigorous
transcendental evaluation must be available.

## Scope and established methods

Nullspace elimination and equality KKT systems are classical; see
[Boyd and Vandenberghe, Section 4.2.4 and Chapter 10](https://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
Gaussian conditioning under linear transformations is likewise established;
[Majumdar and Majumdar](https://arxiv.org/abs/1710.09285) give an
orthogonal-projection treatment. Differentiating constrained optimization
layers through KKT conditions predates this implementation, including
[Amos and Kolter's OptNet](https://proceedings.mlr.press/v70/amos17a.html).
The supported capability combines exact affine restriction with the
Gaussian family's existing inverse and enclosure machinery. These facts
do not establish novelty of the general method.

This page concerns affine equalities and a specified positive Gaussian
mixture. It does not solve arbitrary neural objectives or inequalities.
The contraction conditions are sufficient, finite resource budgets still
apply, and certificates require the documented rigorous numerical backend.
No formal Lean verification is claimed.
