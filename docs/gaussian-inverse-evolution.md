# Certified Gaussian inverse evolution on target boxes

The [Gaussian heat inverse](gaussian-heat-inverse.md) satisfies a nonlinear
PDE in target coordinates. Its time derivative can be evaluated using a
contracted posterior moment and one linear solve. This avoids constructing
the full third-order moment or inverse-Hessian tensor. Enclosing these
quantities over an entire target box gives a continuum statement at a fixed
heat age, rather than checks at selected target points.

The family has fixed centers \(a_j\), fixed log weights \(b_j\), and fixed
viscosity \(\nu>0\). Its heat age is \(s>0\), and

\[
p_j(x,s)=\frac{\exp(b_j-\lVert x-a_j\rVert^2/(4\nu s))}
 {\sum_\ell\exp(b_\ell-\lVert x-a_\ell\rVert^2/(4\nu s))},
\qquad U(x,s)=\frac{x-\mu(x,s)}s,
\qquad \mu=\sum_jp_ja_j.
\]

Write \(D=\max_{j,\ell}\lVert a_j-a_\ell\rVert_2\) and
\(q=D^2/(8\nu s)<1\). This sufficient condition gives a unique global
inverse \(x=X(v,s)\) for every real target \(v\). All time derivatives
below hold the target, centers, log weights, and viscosity fixed. Changing
these parameters together with \(s\) describes a different derivative.

## The contracted inverse PDE

At an inverse point let \(r_j=a_j-\mu\), and define

\[
C=\sum_jp_jr_jr_j^T,\qquad
M_3[h,k]=\sum_jp_jr_j(r_j\cdot h)(r_j\cdot k),
\]

\[
T=\sum_jp_jr_j\lVert r_j\rVert_2^2,
\qquad H=I-\frac{C}{2\nu s},\qquad
J=D_vX=sH^{-1},\qquad G=J^{-1}=D_xU.
\]

The matrix \(H\) obeys \((1-q)I\preceq H\preceq I\). Differentiating
the fixed-point equation \(X=sv+\mu(X,s)\) twice in target directions
gives

\[
D_v^2X[h,k]=\frac{1}{4\nu^2s^2}H^{-1}M_3[Jh,Jk].
\]

The inverse Burgers, or hodograph, equation is

\[
X_s=v+\nu\sum_{\alpha,\beta}(GG^T)_{\alpha\beta}
       \partial_{v_\alpha}\partial_{v_\beta}X.
\]

Since \(JG=I\), its entire diffusion term simplifies to

\[
\boxed{\quad X_s=v+\frac{1}{4\nu s^2}H^{-1}T.\quad}
\]

Here \(T_a=\sum_b(M_3)_{abb}\). Computing it directly costs
\(O(md)\) arithmetic for \(m\) centers in dimension \(d\). The formula
does not require storing \(d^3\) entries. A full target Jacobian still has
\(d^2\) entries when explicitly requested.

There is a separate check on the sign and scale. At fixed physical \(x\),
differentiating the posterior weights gives

\[
\mu_s\big|_x=\frac{\mathbb E[r\lVert x-a\rVert^2]}{4\nu s^2}.
\]

At an inverse point \(x-\mu=sv\), so this becomes

\[
\mu_s\big|_x=\frac{T}{4\nu s^2}-\frac{Cv}{2\nu s}.
\]

Thus \(HX_s=v+\mu_s|_x=Hv+T/(4\nu s^2)\), confirming the same
contracted PDE without computing an inverse Hessian. The common Gaussian
normalization cancels from the posterior; it contributes no additional term
to this vector equation.

## Enclosing an entire target box

Let \(V\) be a finite real target box and \(v_c\) its exact midpoint.
A certified root enclosure at \(v_c\), together with

\[
\lVert D_vX\rVert_2\le\frac{s}{1-q},
\]

covers every inverse point for \(v\in V\): enlarge the midpoint root
enclosure by \(sR/(1-q)\) in each coordinate, where
\(R\ge\sup_{v\in V}\lVert v-v_c\rVert_2\). Another valid enclosure
comes from \(X=sv+\mu\) and \(\mu\in\operatorname{conv}\{a_j\}\).
Intersecting these enclosures can improve the spatial cover.

Posterior moments enclosed over this spatial cover contain their values at
every inverse point. Linear-solve proposals need not themselves be accurate
to make a valid certificate. For a fixed vector proposal \(z\), an interval
bound \(\eta\) on

\[
\left\lVert H(x)z-\frac{T(x)}{4\nu s^2}\right\rVert_2
\]

throughout the cover proves

\[
\left\lVert z-\frac{H(x)^{-1}T(x)}{4\nu s^2}\right\rVert_2
\le\frac{\eta}{1-q}.
\]

This certifies the diffusion contribution with one vector solve. The action
\(Hz=z-\mathbb E[r(r\cdot z)]/(2\nu s)\) can also be computed without
forming a dense matrix. Applying the same residual argument to proposed
solutions of \(Hz_i=se_i\) encloses the columns of \(J\).

Interval dependence can make broad boxes give wide bounds. Subdivision can
improve those bounds; a pointwise numerical comparison alone cannot replace
the box proof. The certificate concerns one fixed heat age and the stated
Gaussian family, not arbitrary initial data or an integrated time-step.

```python
from fractions import Fraction

from omnibias.core.verified.gaussian_evolution import certify_gaussian_inverse_evolution
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval

model = GaussianHeatMap(
    ((-1, 0), (1, 1), (0, -1), (1, -1)),
    (0, Fraction(1, 3), Fraction(-1, 4), Fraction(1, 5)),
    viscosity=1, time=2,
)
targets = (Interval(0.125, 0.15625), Interval(-0.125, -0.09375))
evolution = certify_gaussian_inverse_evolution(model, targets)
assert evolution is not None

inverse_positions = evolution.spatial_enclosure
target_jacobian = evolution.target_jacobian  # Entry [i][j] encloses dX_i/dv_j.
heat_age_derivative = evolution.time_derivative  # Every target in the box.
diffusion_correction = evolution.diffusion
```

The factory returns a frozen `GaussianInverseEvolution` or `None` when the
contraction regime, numerical range, or resource checks do not permit the
proof. Invalid arguments raise. Non-affine families require the rigorous
transcendental backend. The optional `max_iterations=128` and
`absolute_tolerance=1e-12` control the midpoint root proposal; they do not
specify the final widths over the box. A coarse certified root can still
produce valid, wider enclosures.

The result also retains `third_trace`, the global Euclidean
`diffusion_norm_bound`, the midpoint `root_certificate`, and the serialized
`certificate`. Identical centers use the exact affine formula and need no
midpoint root certificate. The complete returned Jacobian is a dense matrix;
the contracted evolution formula avoids the third-order tensor.

## A dimension-free bound on the evolution correction

The contracted third moment has the conservative bound
\(\lVert T\rVert_2\le D^3/4\). To prove it, take any unit vector \(u\).
Then \(u\cdot T=\operatorname{Cov}(u\cdot a,\lVert a-\mu\rVert^2)\).
These two scalar variables have ranges at most \(D\) and \(D^2\),
respectively. The latter follows because \(\mu\) belongs to the centers'
convex hull. Cauchy–Schwarz and the bound of range squared divided by four
on a scalar variance prove the claim. Consequently,

\[
\lVert X_s-v\rVert_2
\le\frac{D^3}{16\nu s^2(1-q)}
=\frac{Dq}{2s(1-q)}.
\]

This independent global bound can tighten a moment-based enclosure. It is
not asserted to be sharp. When the centers coincide, \(D=q=0\) and
\(X(v,s)=sv+a_0\), so \(X_s=v\) exactly.

## One-pass inverse predictors

The same moment argument supplies explicit approximations that require one
posterior evaluation, without inverse iteration or a linear solve. Define
the limiting posterior and its centered moments by

\[
p_j^\infty(v)=\operatorname{softmax}_j
       \left(b_j+\frac{a_j\cdot v}{2\nu}\right),\qquad
\mu_\infty=\mathbb E_\infty[a],\qquad
T_\infty=\mathbb E_\infty[(a-\mu_\infty)\lVert a-\mu_\infty\rVert^2].
\]

For every real target in the unique-inverse regime,

\[
\left\lVert X-(sv+\mu_\infty)\right\rVert_2
\le\frac{D^3}{16\nu s}=\frac{Dq}{2},
\]

\[
\left\lVert X-\left(sv+\mu_\infty-\frac{T_\infty}{4\nu s}\right)
\right\rVert_2\le Dq^2.
\]

To see the first bound, set \(m_*=X-sv\in\operatorname{conv}\{a_j\}\).
The actual posterior is proportional to
\(p_j^\infty\exp(-\lambda c_j)\), where
\(\lambda=1/(4\nu s)\) and \(c_j=\lVert a_j-m_*\rVert^2\).
Freeze \(m_*\) and interpolate the tilt from zero to one. The derivative
of its mean is \(-\lambda\operatorname{Cov}(a,c)\); the ranges of
\(u\cdot a\) and \(c\) are at most \(D\) and \(D^2\). Integrating
the resulting bound \(\lambda D^3/4\) proves the first estimate.

For the second, the mean's second tilt derivative is
\(\lambda^2\mathbb E[(a-\mathbb E a)(c-\mathbb E c)^2]\), whose norm
is at most \(\lambda^2D^5/4\). Its first-order Taylor remainder is
therefore at most \(Dq^2/2\). At zero tilt,
\(\operatorname{Cov}_\infty(a,c)=T_\infty-2C_\infty(m_*-\mu_\infty)\).
The first estimate and \(\lVert C_\infty\rVert\le D^2/4\) bound the
remaining correction by another \(Dq^2/2\).

These estimates are uniform in the target and preserve translations of the
centers and common shifts of the biases. The second bound improves the first
when \(q<1/2\); it need not be the better approximation or bound at every
target. No novelty or measured performance claim follows from this derivation.

`certify_gaussian_inverse_predictor(model, target_box, order=1)` in
`omnibias.core.verified.gaussian_predictor` evaluates either predictor with
outward interval arithmetic. Its `predictor_enclosure` encloses the explicit
formula over the box, while `enclosure` also contains the true inverse.
The latter includes the analytic approximation error and an intersection
with the inverse's center-hull bound.

```python
from omnibias.core.verified.gaussian_predictor import certify_gaussian_inverse_predictor

aged_model = GaussianHeatMap(
    model.centers, model.biases, viscosity=model.viscosity, time=8,
)
zeroth = certify_gaussian_inverse_predictor(aged_model, targets, order=0)
first = certify_gaussian_inverse_predictor(aged_model, targets, order=1)
assert zeroth is not None and first is not None
assert first.approximation_error_bound < zeroth.approximation_error_bound
inverse_box = first.enclosure
exact_predictor_objective_gap = first.approximation_error_bound**2 / (2 * aged_model.time)
```

`approximation_error_bound` is a Euclidean bound between the exact predictor
and true inverse **at the same target**. It is not the radius of the returned
box: that box also includes variation across `target_box` and interval
evaluation error. The frozen result retains the model, target box, order,
limiting `relative_mean`, `third_trace`, and serialized `certificate`.
Only orders zero and one are supported. Invalid inputs raise; unsupported
contraction, exhausted resources, or nonfinite bounds return `None`.
Non-affine evaluation requires the rigorous transcendental backend.

The stored core model already contains its diameter. A predictor query uses
\(O(md)\) arithmetic and storage, without a covariance matrix. Model
construction has separate costs, including the center-diameter calculation.
These are value bounds; they do not certify derivatives of the predictor or
the true inverse.

The Torch and JAX modules expose `gaussian_heat_inverse_predictor(target,
centers, biases, viscosity=1.0, time=1.0, order=1)`. They accept a batch of
targets with shape `(..., d)` and shared centers `(m, d)` and biases `(m,)`.
Their values retain gradients into targets, centers, biases, viscosity, and
time. Autodiff differentiates the explicit predictor, not the exact inverse.

```python
import torch
from omnibias.torch.gaussian_predictor import gaussian_heat_inverse_predictor as torch_predictor

query = torch.tensor([[0.125, -0.125], [0.15625, -0.09375]],
                     dtype=torch.float64, requires_grad=True)
centers = torch.tensor([[-1.0, 0.0], [1.0, 1.0], [0.0, -1.0], [1.0, -1.0]],
                       dtype=torch.float64)
biases = torch.tensor([0.0, 0.25, -0.25, 0.125],
                      dtype=torch.float64, requires_grad=True)
prediction = torch_predictor(query, centers, biases, time=8.0, order=1)
assert bool(prediction.valid.all())
prediction.value.square().sum().backward()
assert query.grad is not None and biases.grad is not None
```

```python
import jax
import jax.numpy as jnp
from omnibias.jax.gaussian_predictor import gaussian_heat_inverse_predictor as jax_predictor

jax.config.update("jax_enable_x64", True)
predict = jax.jit(lambda v, a, b: jax_predictor(v, a, b, time=8.0, order=1))
jax_prediction = predict(jnp.asarray(query.detach().numpy()),
                         jnp.asarray(centers.numpy()), jnp.asarray(biases.detach().numpy()))
assert bool(jax_prediction.valid.all())
```

Tensor `approximation_error_bound` and `contraction_bound` are numerical
evaluations of analytic formulas, with stopped gradients. They are **not
outward-rounded certificates** and omit floating-point evaluation error.
`valid` records the numerical checks, not a proof of the error estimate.
Invalid arithmetic or \(q\ge1\) yields false validity and NaN values.
Each tensor call computes the shared diameter in \(O(m^2d)\) work, followed
by \(O(md)\) work per target. The supported bounds and approximation order
remain distinct from the all-order aging series below.

## Measured certified queries

The [recorded CPU experiment](benchmarks/gaussian_inverse_predictor.json)
compares fresh certificates for 17 scalar targets \(v=k/32\),
\(k=-8,\ldots,8\), with centers \((-1,1)\), biases \((0,1/3)\),
\(\nu=1\), and \(s=1024\). Here \(q=1/2048\), so the corrected
predictor's analytic error is \(1/2097152\), small enough for the
requested physical coordinate half-width of \(10^{-6}\). Both methods
must meet that same threshold and enclose independent 80-digit inverse roots.

| Fresh certificate method | Median time per query | Largest coordinate half-width |
| --- | ---: | ---: |
| One-pass corrected predictor | 0.284 ms | \(4.77\times10^{-7}\) |
| Iterative inverse from zero | 1.152 ms | \(3.73\times10^{-8}\) |

The measured median ratio is **4.05×**. The iterative method reaches a
narrower enclosure than requested as a consequence of its final iteration;
this comparison uses a common acceptance threshold, not identical attained
errors. Every timed query computes its posterior moments or inverse
iterations and seals a new certificate. The same exact model is reused by
both methods; its 0.140 ms construction is recorded separately and excluded
equally. No predictor artifact or posterior is reused across queries.

The run used two warmups and nine recorded samples after untimed validation,
in a pure-core process with a 60-second limit and a 512 MiB RSS limit. Tests
and compilation had finished before timing. The artifact retains operands,
software versions, raw samples, source hashes and validation results.
Reproduce it from the repository root with:

```bash
uv run python benchmarks/gaussian_inverse_predictor.py
```

Use `--check-only` to validate without collecting timings. An untimed
\(s=16\) diagnostic encloses its reference but fails the same \(10^{-6}\)
accuracy requirement; it receives no speed comparison. These measurements
cover one scalar family in the large-age regime, with no tensor, derivative,
training or general-PDE timing claim.

## A global optimization consequence

Let \(\Phi(x,s)=-2\nu\log\tau(x,s)\), using the positive Gaussian heat
potential, and define \(F_v(x)=\Phi(x,s)-v\cdot x\). Its Hessian is
\(H/s\), so it is globally \(1/s\)-smooth and
\((1-q)/s\)-strongly convex. Its unique global minimizer is \(X(v,s)\).
Any exact predictor \(P\) with \(\lVert P-X\rVert_2\le\delta\)
therefore satisfies

\[
0\le F_v(P)-\min_xF_v(x)\le\frac{\delta^2}{2s}.
\]

The two predictors give global objective gaps at most
\(D^2q^2/(8s)\) and \(D^2q^4/(2s)\), respectively. The exact rational
`approximation_error_bound` supplies \(\delta\) in the example above.
This applies to the exact predictor formula. A numerical point selected
from an interval or produced by a tensor call needs its additional evaluation
error included in \(\delta\). The result is for this strongly convex
Gaussian potential, not arbitrary neural-network objectives.

## Certified all-order aging series

A separate aging API implements the forward recurrence for higher
approximation orders. It requires the stronger sufficient bound
\(q<1/3\), including at orders zero and one. The preceding one-pass
predictor API retains its broader \(q<1\) scope.
Fix a real target and its limiting posterior \(p=p^\infty(v)\), put
\(A_j=a_j-\mu_\infty\), and set \(z=1/s\). The centered inverse
correction \(h=X-sv-\mu_\infty\) satisfies

\[
h=\frac{\sum_jp_jA_j\exp\!\left(
       \frac{z}{2\nu}(A_j\cdot h-\lVert A_j\rVert^2/2)\right)}
 {\sum_jp_j\exp\!\left(
       \frac{z}{2\nu}(A_j\cdot h-\lVert A_j\rVert^2/2)\right)}.
\]

Write \(h(z)=\sum_{n\ge1}h_nz^n\) and each exponential as
\(e_j(z)=\sum_{n\ge0}e_{j,n}z^n\). For its logarithm coefficients,

\[
\ell_{j,1}=-\frac{\lVert A_j\rVert^2}{4\nu},\qquad
\ell_{j,n}=\frac{A_j\cdot h_{n-1}}{2\nu}\quad(n\ge2),\qquad
e_{j,0}=1,
\]

\[
e_{j,n}=\frac1n\sum_{k=1}^n k\ell_{j,k}e_{j,n-k},\qquad
d_n=\sum_jp_je_{j,n},\qquad b_n=\sum_jp_jA_je_{j,n},
\]

\[
h_0=0,\quad d_0=1,\qquad
h_n=b_n-\sum_{k=1}^n d_kh_{n-k}.
\]

The factor \(z\) in the exponent makes every right-hand side depend only
on previously computed \(h\) coefficients. No implicit matrix solve is
needed. In particular, \(h_1=-T_\infty/(4\nu)\). These are ordinary
Taylor coefficients, not raw derivatives. Direct coefficient convolution
through order \(N\) uses \(O((m+d)N^2+mdN)\) arithmetic and
\(O(md+(m+d)N)\) forward working storage; an autodiff tape can require more.

The implementation avoids forming large unscaled coefficients. At the
declared heat age \(s\), it stores \(c_n=h_n/s^n\), so

\[
X(v,s/t)-(s/t)v-\mu_\infty=\sum_{n\ge0}c_nt^n,\qquad c_0=0.
\]

The logarithm coefficients in the scaled recurrence are
\(-\lVert A_j\rVert^2/(4\nu s)\) at first order and
\(A_j\cdot c_{n-1}/(2\nu s)\) thereafter. The physical inverse is
approximated by \(sv+\mu_\infty+\sum_{n=1}^Nc_n\), evaluating the
formal parameter at \(t=1\). That parameter scales **inverse heat age**;
it is not the physical time or a target coordinate.

A conservative convergence guarantee is uniform over all real targets and
log weights. For \(D>0\), define

\[
R=\frac D2,\qquad Z=\frac{8\nu}{3D^2}.
\]

For complex \(|z|\le Z\), the right-hand side maps the closed complex
ball \(\lVert h\rVert_2\le R\) strictly into itself and is a contraction
with constant at most \(3/5\). Here is an explicit bound. Put
\(Q=|z|D^2/(8\nu)\le1/3\). The logit phases lie in an arc of
half-width at most \(Q(1+2R/D)=2Q\). Rotating the arc bounds the partition
modulus. More precisely, reweighting by the real parts first leaves positive
weights; the normalized partition modulus is at least \(\cos\theta\)
for phase half-width \(\theta\). The pairwise covariance identity and
Cauchy–Schwarz then give
\(\lVert\operatorname{Cov}_{\mathbb C}(A)\rVert\le D^2/(4\cos^2\theta)\)
and \(\lVert\operatorname{Cov}_{\mathbb C}(A,c)\rVert\le D^3/(4\cos^2\theta)\)
when the real scalar cost \(c\) has range at most \(D^2\). These use
the positive-weight variance bounds before division by the complex partition.
The derivative in \(h\) is consequently bounded by

\[
Q\sec^2(2Q)\le\frac{Q}{1-4Q^2}\le\frac35.
\]

At \(h=0\), interpolate the cost tilt as in the predictor proof. The
complex covariance bound gains at most the factor \(\sec^2(tQ)\).
Integrating gives a forcing norm at most
\((D/2)\tan Q\le3D/16\), using
\(\tan Q\le Q/(1-Q^2)\le3/8\). Thus forcing plus contraction displacement
is at most \(3D/16+(3/5)(D/2)=39D/80<R\). The denominator never
vanishes on this domain. Uniform holomorphic contraction gives the analytic
fixed point. The strict ball margin and nonzero partition allow the closed
parameter circle in Cauchy's coefficient estimate, yielding

\[
\lVert h_n\rVert_2\le\frac D2Z^{-n},\qquad
\left\lVert h(z)-\sum_{n=1}^Nh_nz^n\right\rVert_2
\le\frac D2\frac{(3q)^{N+1}}{1-3q},\quad 0\le q<\frac13.
\]

Only the correction is analytic at \(z=0\): the physical inverse has the
term \(v/z\), a pole when \(v\ne0\). If \(D=0\), the correction is
identically zero. The disk is a sufficient bound; it does not assert
convergence throughout \(q<1\). The real order-zero and order-one bounds
above can be substantially tighter than this general Cauchy tail; the
implementation uses the smaller applicable bound at these orders.

`certify_gaussian_inverse_aging(model, target_box, order=8)` in
`omnibias.core.verified.gaussian_aging` returns a frozen
`GaussianInverseAging` or an inconclusive `None`. It computes outward
interval coefficients and intersects each \(c_n\) enclosure with the
coordinate bounds implied by \(\lVert c_n\rVert_2\le(D/2)(3q)^n\).
This remains valid throughout the supplied target box.

```python
from omnibias.core.verified.gaussian_aging import certify_gaussian_inverse_aging

aging = certify_gaussian_inverse_aging(aged_model, targets, order=8)
assert aging is not None
assert len(aging.correction_coefficients) == 9  # Includes structural c0 = 0.
assert aging.approximation_error_bound < first.approximation_error_bound
aging_inverse_box = aging.enclosure
```

`correction_coefficients` has shape `(order + 1, dimension)` and encloses
the ordinary scaled coefficients at every target in the box.
`predictor_enclosure` encloses the exact truncated formula, including its
variation over the box and interval evaluation error.
`approximation_error_bound` is the uniform Euclidean analytic tail between
that exact formula and the true inverse at the same target. `enclosure`
adds that tail and intersects the global center hull. The record also retains
`relative_mean`, exact model metadata, and its serialized `certificate`.

Orders are bounded by the model's resource budget. Invalid inputs raise;
\(q\ge1/3\), exhausted resources, and nonfinite enclosures return `None`.
A missing rigorous transcendental backend raises, except for identical
centers. Increasing order decreases the analytic tail, but interval
dependency and rounding can enlarge the computed coefficient enclosures.
The full returned enclosure therefore need not improve with every order.

The numerical Torch and JAX modules expose
`gaussian_heat_inverse_aging(target, centers, biases, viscosity=1.0,
time=1.0, order=8)`. They use the same scaled recurrence, with no inverse
iteration, dense covariance, or matrix solve. Returned
`correction_coefficients` have shape `(order + 1, ..., dimension)`.

```python
from omnibias.torch.gaussian_aging import gaussian_heat_inverse_aging as torch_aging
from omnibias.jax.gaussian_aging import gaussian_heat_inverse_aging as jax_aging

aging_prediction = torch_aging(query, centers, biases, time=8.0, order=8)
assert bool(aging_prediction.valid.all())
assert aging_prediction.correction_coefficients.shape == (9, *query.shape)
aging_gradient, = torch.autograd.grad(aging_prediction.value.sum(), query)
assert bool(torch.isfinite(aging_gradient).all())

jit_aging = jax.jit(lambda v, a, b: jax_aging(v, a, b, time=8.0, order=8))
jax_aging_prediction = jit_aging(
    jnp.asarray(query.detach().numpy()), jnp.asarray(centers.numpy()),
    jnp.asarray(biases.detach().numpy()),
)
assert bool(jax_aging_prediction.valid.all())
```

Autodiff here differentiates the finite approximation and its coefficients;
it does not supply derivative error bounds for the true inverse. Tensor
`approximation_error_bound` and `contraction_bound` are stopped-gradient
numerical diagnostics, not outward certificates. Invalid arithmetic or
\(q\ge1/3\) gives false validity and NaN values, coefficients, and error
estimates. The analytic guarantee alone does not account for tensor rounding.

### Certifying a target-gradient proposal

The aging value and its autodiff gradient can instead serve as proposals to
the existing [target VJP certifier](gaussian-heat-inverse.md#certifying-a-proposed-target-gradient).
It checks root and adjoint residuals against an exact declared model, so this
composition encloses the true inverse derivative without inverse iteration
or matrix factorization.

```python
from omnibias.core.verified.gaussian_sensitivity import certify_gaussian_target_vjp

exact_scalar_model = GaussianHeatMap(
    ((-1,), (1,)), (0, Fraction(1, 3)), viscosity=1, time=16,
)
scalar_target = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
scalar_prediction = torch_aging(
    scalar_target,
    torch.tensor([[-1.0], [1.0]], dtype=torch.float64),
    torch.tensor([0.0, 1.0 / 3.0], dtype=torch.float64),
    time=16.0, order=8,
)
assert bool(scalar_prediction.valid)
gradient_proposal, = torch.autograd.grad(scalar_prediction.value.sum(), scalar_target)
gradient_certificate = certify_gaussian_target_vjp(
    exact_scalar_model,
    (Fraction(1, 4),),
    scaled_point=tuple((scalar_prediction.value.detach() / 16.0).tolist()),
    upstream=(1.0,),
    proposal=tuple(gradient_proposal.detach().tolist()),
)
assert gradient_certificate is not None
true_target_gradient_box = gradient_certificate.enclosure
```

The tensor bias is the stored binary approximation to \(1/3\); the core
model uses exact \(1/3\). Proposals need not be exact or come from precisely
the same model: the checker independently bounds their errors for the
declared model. The guarantee comes from those residual checks, not from
autodiff alone. An inaccurate proposal can yield a wide enclosure.

## Certified physical-age derivatives

The same coefficient enclosure also supports raw derivatives with respect
to physical heat age. Holding targets, centers, biases, and viscosity fixed,

\[
\partial_s^kX
=\begin{cases}sv+\mu_\infty,&k=0,\\v,&k=1,\\0,&k\ge2,\end{cases}
\quad+\quad\frac{(-1)^k}{s^k}\sum_{n\ge1}n^{\overline{k}}c_n,
\]

where \(n^{\overline{k}}=n(n+1)\cdots(n+k-1)\) and
\(n^{\overline0}=1\). These are raw derivatives, with no division by
\(k!\). They differ from derivatives in the formal inverse-age parameter
\(t\) or in the target coordinates. Unlike a polynomial in physical age,
a finite sum of inverse-age powers can have nonzero derivatives of every
order, including \(k>N\).

For \(k\ge1\), put \(r=3q\) and \(K=N+1\). Cauchy's coefficient
bound gives the Euclidean derivative tail

\[
E_k\le\frac{D}{2s^k}\sum_{n=K}^{\infty}n^{\overline{k}}r^n,
\]

\[
\sum_{n=K}^{\infty}n^{\overline{k}}r^n
=k!r^K\sum_{j=0}^k\binom{K+k-1}{k-j}
                   \frac{r^j}{(1-r)^{j+1}}.
\]

Vandermonde's identity and the geometric binomial moments prove this
finite positive formula. It avoids cancellation from subtracting a long
partial sum and uses \(O(k)\) rational summands. At first order it reduces
to \(r^{N+1}[(N+1)-Nr]/(1-r)^2\). The value bound alone could not
justify this derivative bound; the analytic disk and coefficient decay do.

```python
from omnibias.core.verified.gaussian_aging_derivative import certify_gaussian_aging_derivative

age_derivative = certify_gaussian_aging_derivative(aging, order=1)
assert age_derivative is not None
heat_age_derivative_box = age_derivative.enclosure
```

The returned `GaussianAgingDerivative` contains the source `aging`, the
derivative `order`, a `polynomial_enclosure` for the derivative of the exact
truncated formula, its Euclidean `approximation_error_bound`, the final
`enclosure`, and a serialized `certificate`. Order zero retains the source
value bound. Other orders use the derivative tail and propagated interval
coefficient errors. The guarantee is uniform over the source target box at
its fixed age; the current call does not enclose a time interval.

## Scope and prior work

The posterior derivative identities are standard exponential-family
calculus; see [Wainwright and Jordan, Proposition 3.1](https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf).
Hodograph transformations of viscous conservation laws are also established;
[Arsie, Lorenzoni and Moro, Section 5](https://arxiv.org/abs/1301.0950)
provides a primary example. The contraction proof and explicit Gaussian
moment identities specialize these methods to the supported inverse family.
They do not establish a new general PDE solver or universal efficiency.
Connections between formal inverse maps, Legendre transforms, and Burgers
equations also appear in [Zhao's recurrent inversion construction](https://arxiv.org/abs/math/0403020).
That work supplies broader context, not an attribution of the specific
uniform predictor bounds proved above.
Formal multivariate coefficient inversion also has classical antecedents in
[Good's generalization of Lagrange inversion](https://doi.org/10.1017/S0305004100034666).
The aging recurrence and conservative analytic disk are derived above for
this family; their novelty has not been established.

The implementation's certificates combine interval arithmetic, exact
metadata, residual bounds, and the analytic arguments stated here. They
are not formal Lean proofs. Failure of a stated sufficient bound, an
inconclusive enclosure, or a resource refusal does not prove that the
original PDE or inverse problem has no solution.
