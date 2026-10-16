# Global inverse of a Gaussian heat field

A positive Gaussian heat mixture gives a Cole–Hopf velocity whose spatial
Jacobian can have either sign. Sufficient heat smoothing makes this field
globally invertible. The inverse can then be enclosed from a rigorous residual
and an explicit contraction constant, without a local root search or a
full-rank assumption on the Gaussian centers.

This uses classical Gaussian score identities, mean-shift fixed points and
the contraction theorem. The bound below is proved directly; no new general
PDE theorem or mathematical novelty is claimed. Unlike the
[prescribed periodic solver](periodic-burgers.md), this construction starts
with a specified Gaussian heat family rather than an arbitrary initial
velocity.

## Gaussian heat flow and its velocity

Fix centers \(a_j\in\mathbb R^d\), finite biases \(b_j\), viscosity
\(\nu>0\), and heat age \(s>0\). Define

\[
\tau(x,s)=(4\pi\nu s)^{-d/2}
\sum_j\exp\left(b_j-\frac{\lVert x-a_j\rVert_2^2}{4\nu s}\right).
\]

The sum is nonempty. Each Gaussian solves the heat equation, and
\(\tau>0\) everywhere. If \(s=t+s_0\), this is a smooth heat solution for
\(t>-s_0\). The associated potential velocity
\(U=-2\nu\nabla_x\log\tau\) solves the irrotational vector Burgers system.
The common Gaussian normalization affects the heat density, but cancels from
the spatial logarithmic derivative.

Let \(p_j(x,s)\) be the normalized positive summand weights and write

\[
\mu(x,s)=\sum_jp_j(x,s)a_j,\qquad
C(x,s)=\sum_jp_j(x,s)(a_j-\mu)(a_j-\mu)^T.
\]

Differentiating the finite sum gives

\[
U(x,s)=\frac{x-\mu(x,s)}s,\qquad
D_x\mu(x,s)=\frac{C(x,s)}{2\nu s},\qquad
D_xU(x,s)=\frac1sI-\frac{C(x,s)}{2\nu s^2}.
\]

The identity term is essential. It is absent from the positive real
exponential family in [the previous inverse map](inverse-pde-map.md), whose
velocity Jacobian is negative semidefinite. Here even a single Gaussian gives
\(U=(x-a_1)/s\), with a positive definite Jacobian.

These are Gaussian score/posterior-mean identities. For the one-dimensional
normal location model, [Efron, *Tweedie's Formula and Selection Bias*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3325056/),
equation (2.8), expresses the posterior mean and variance through the first and
second derivatives of the log density. The formulas above follow by applying
the same calculation to the finite mixture in each direction.

## A global contraction threshold

Let

\[
D^2=\max_{i,j}\lVert a_i-a_j\rVert_2^2,
\qquad q=\frac{D^2}{8\nu s}.
\]

For a unit vector \(z\), the scalars \(z\cdot a_j\) lie in an interval of
length at most \(D\). Their variance is at most \(D^2/4\): the second
moment around the interval midpoint is at most that quantity, and the variance
is no larger than the second moment around any chosen center. Therefore

\[
0\preceq C(x,s)\preceq\frac{D^2}{4}I,
\qquad
\frac{1-q}sI\preceq D_xU(x,s)\preceq\frac1sI.
\]

To solve a prescribed velocity equation \(U(x,s)=v\), one can iterate
\(x\mapsto sv+\mu(x,s)\). The implementation uses the scaled coordinate
\(y=x/s\). In this coordinate the weights are

\[
p_j(y)=\operatorname{softmax}_j\left(
\frac{a_j\cdot y}{2\nu}-\frac{\lVert a_j\rVert_2^2}{4\nu s}+b_j
\right),
\qquad
T_v(y)=v+\frac1s\sum_jp_j(y)a_j.
\]

The derivative \(D_yT_v=C/(2\nu s)\) has Euclidean operator norm at most
\(q\), globally. Hence

\[
s>\frac{D^2}{8\nu}\quad\Longrightarrow\quad q<1
\]

makes \(T_v\) a contraction on the complete space \(\mathbb R^d\). Every
target \(v\in\mathbb R^d\) then has exactly one inverse \(x_*\), and
exact fixed-point iteration converges from every finite starting point.
The argument permits duplicate centers and centers in a proper affine
subspace. The identity term supplies the missing directions.

The inverse is also the unique global minimizer of

\[
\Psi_v(y)=\frac12\lVert y-v\rVert_2^2
-\frac{2\nu}{s}\log\sum_j
\exp\left(\frac{a_j\cdot y}{2\nu}
          -\frac{\lVert a_j\rVert_2^2}{4\nu s}+b_j\right).
\]

Its gradient is \(y-T_v(y)\), and its Hessian is bounded below by
\((1-q)I\). The bound concerns the declared ideal real model; it does not
assume that a floating-point iteration itself is a contraction.

At any inverse, \(x_*=sv+\mu(x_*,s)\), so

\[
x_*\in sv+\operatorname{conv}\{a_j\}.
\]

This gives useful coordinate bounds in addition to the residual certificate.
In fact every target has at least one inverse even without \(q<1\): the
continuous fixed-point map sends the compact convex set
\(sv+\operatorname{conv}\{a_j\}\) into itself. Equivalently, \(\Psi_v\)
is coercive because its quadratic term dominates the at-most-linear growth
of the finite log-sum-exp. Outside the contraction regime, uniqueness is the
property that is no longer supplied by this bound.

## Residual certificate and the choice of norm

For any proposed scaled point \(\widehat y\), contraction gives

\[
\lVert\widehat y-y_*\rVert_2
\le\frac{\lVert\widehat y-T_v(\widehat y)\rVert_2}{1-q},
\qquad
\lVert s\widehat y-x_*\rVert_2
\le\frac{s\lVert\widehat y-T_v(\widehat y)\rVert_2}{1-q}.
\]

To prove the first inequality, insert \(T_v(\widehat y)\) between
\(\widehat y\) and \(T_v(y_*)=y_*\), apply the triangle inequality and
the contraction bound, then bring \(q\lVert\widehat y-y_*\rVert_2\) to
the left-hand side.

Suppose interval evaluation encloses each residual coordinate, and let
\(r_i\) be an upper bound on the absolute value of that coordinate. Then
\(\sum_i r_i^2/(1-q)^2\) bounds the squared scaled error. Exact rational
arithmetic on binary64 endpoints, followed by an upward-bounded square root,
avoids an underestimated Euclidean norm. The physical radius must also include
the outward effect of scaling by the exact positive \(s\).

This is a **Euclidean** contraction proof. The same \(q\) must not simply be
reused as an infinity-norm matrix bound: a spectral norm bound does not imply
the same row-sum bound. A Euclidean ball of radius \(R\) is safely contained
in the axis-aligned box with coordinate radii \(R\), so returning that box
preserves the root enclosure. Intersecting it with the convex-hull coordinate
bounds preserves it as well.

Any finite rigorously bounded residual produces a valid coarse certificate
when \(q<1\). A small residual is needed for a small enclosure, rather than
for existence itself. The denominator \(1-q\) explains the worsening error
bounds near the threshold.

Differentiating the inverse decision with respect to its target gives

\[
D_vx_*=s\left(I-\frac{C(x_*,s)}{2\nu s}\right)^{-1},
\qquad
sI\preceq D_vx_*\preceq\frac{s}{1-q}I.
\]

Thus the inverse is globally Lipschitz with constant \(s/(1-q)\). This is
an analytic sensitivity identity; a numerical sensitivity matrix needs its
own interval linear-system check. The present solver returns a root enclosure,
not a sensitivity matrix.

## Why a strict inequality is necessary for this guarantee

Consider two equal-weight one-dimensional centers \(-a,a\), with \(a>0\).
Then

\[
U(x,s)=\frac{x-a\tanh(ax/(2\nu s))}{s},
\qquad q=\frac{a^2}{2\nu s}.
\]

For target zero, \(x=0\) is always a root. If \(q>1\), the derivative of
\(F(x)=x-a\tanh(ax/(2\nu s))\) is negative at zero, while \(F(a)>0\).
Moreover \(F''(x)>0\) for \(x>0\). There is exactly one positive root
between zero and \(a\), and odd symmetry gives a negative root. Thus this
target has exactly three inverses.

At \(q=1\), \(F'(x)=\tanh^2(x/a)\), so zero remains the only zero, but
the Jacobian vanishes there. The expansion
\(U(x,s)=x^3/(3a^2s)+O(x^5)\) shows that its inverse is not Lipschitz
near target zero. This proves that the strict threshold is sharp as a uniform
guarantee over mixtures. It is not a necessary condition for a particular
target to have a unique inverse.

## Exact model and inverse API

`GaussianHeatMap` is imported from `omnibias.core.verified.gaussian_heat`.
The constructor
`GaussianHeatMap(centers, biases, viscosity=1, time=1, budget=None)` takes
exact integers or `Fraction` objects. The bias \(b_j\) represents the positive
weight \(e^{b_j}\); it is not a signed mixture coefficient. The `time` argument
is the heat age \(s>0\), which can already include an initial smoothing offset.
The frozen model exposes `dimension`, `mode_count`, `diameter_squared` and the
exact rational `contraction_bound`, along with its declared data.

| Method | Coordinate convention |
| --- | --- |
| `velocity(box)` | Encloses \(U(x,s)\) at physical coordinates \(x\). |
| `scaled_velocity(box)` | Encloses \(U(sy,s)\) at scaled coordinates \(y\). |
| `scaled_fixed_point(box, target)` | Encloses \(T_v(y)\). |
| `scaled_jacobian(box)` | Encloses \(D_y[U(sy,s)]=I-C/(2\nu s)\). |
| `scaled_jacobian_action(box, vector)` | Encloses that Jacobian applied to an interval vector without forming the matrix. |
| `float_scaled_fixed_point(point, target)` | Produces an untrusted numerical iterate. |

The verified inverse functions are imported from
`omnibias.core.verified.inverse_gaussian_heat`. Targets also use exact integers
or `Fraction` objects. Interval methods and certificates require the rigorous
transcendental backend, available through `omnibias-core[verified]`.

```python
from fractions import Fraction

from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.inverse_gaussian_heat import inverse_gaussian_heat

model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=1, time=2)
target = (Fraction(1, 3),)
result = inverse_gaussian_heat(
    model, target, initial=(0.0,), absolute_tolerance=1e-10,
)
assert model.contraction_bound == Fraction(1, 4)
assert result.status == "certified" and result.tolerance_met
certificate = result.certificate
assert certificate is not None and certificate.error_bound <= 1e-10
assert model.velocity(certificate.enclosure)[0].contains(target[0])
# The scaled center is about 0.44209; the physical inverse is about 0.88418.

small_time = GaussianHeatMap(
    ((-1,), (1,)), (0, 0), viscosity=1, time=Fraction(1, 4),
)
refusal = inverse_gaussian_heat(small_time, (0,))
assert small_time.contraction_bound == 2
assert refusal.status == "outside_contraction_regime"
```

`inverse_gaussian_heat(model, target, initial=None, max_iterations=128, absolute_tolerance=1e-10)`
seeks a certificate whose **physical Euclidean** error bound is at most the
requested tolerance. Its result has `scaled_point`, `iterations`,
`tolerance_met`, `certificate`, and one of three statuses:

| Status | Meaning |
| --- | --- |
| `"certified"` | A global inverse enclosure was proved. Inspect `tolerance_met` separately. |
| `"not_certified"` | No usable numerical enclosure was produced; this is inconclusive. |
| `"outside_contraction_regime"` | The exact test \(q<1\) fails, so this sufficient uniqueness proof does not apply. |

A finite search may return `"certified"` with `tolerance_met=False` and a
coarser valid enclosure. `scaled_point` is the approximate \(y=x/s\) center;
the physical error is measured relative to the ideal product
\(s\,\texttt{scaled_point}\), rather than an independently rounded product.
The status outside the contraction regime does not assert nonexistence or
multiple roots for the particular target.

`certify_gaussian_inverse(model, target, scaled_point)` certifies a proposed
point directly, returning `GaussianInverseCertificate` or `None`. Its fields
are `scaled_point`, `scaled_enclosure`, physical `enclosure`,
`scaled_error_bound`, physical `error_bound`, exact `contraction_bound`, and
the serialized `certificate`. The enclosures also use the coordinate bounds
from the translated convex hull. Serialization binds the inputs and bounds;
it does not assert formal verification by the Lean kernel.

## Differentiating the inverse

The inverse equation can define a differentiable numerical layer. Set
\(r_j=a_j-\mu\) and

\[
F(x)=x-sv-\mu(x,s),\qquad
H(x)=D_xF=I-\frac{C(x,s)}{2\nu s}.
\]

In the strict contraction regime, \((1-q)I\preceq H\preceq I\), so the
implicit function theorem applies to every target. For a fixed upstream
vector \(g\), solve the adjoint equation \(H\lambda=g\). The
vector–Jacobian products for the physical output \(x_*\) are

\[
\begin{aligned}
\nabla_v L &=s\lambda,\\
\partial_{b_j}L&=p_j(\lambda\cdot r_j),\\
\nabla_{a_j}L&=p_j\left[\lambda+
  \frac{(\lambda\cdot r_j)(x_*-a_j)}{2\nu s}\right],\\
\partial_\nu L&=\sum_j
  \frac{p_j(\lambda\cdot r_j)\lVert x_*-a_j\rVert_2^2}{4\nu^2s},\\
\partial_s L&=\lambda\cdot v+\sum_j
  \frac{p_j(\lambda\cdot r_j)\lVert x_*-a_j\rVert_2^2}{4\nu s^2}.
\end{aligned}
\]

All probabilities and covariances in these identities are evaluated at the
true inverse. They follow by differentiating \(F=0\); in particular,
\(\partial_{b_j}\mu=p_jr_j\). Adding the same scalar to every bias leaves
the output unchanged, giving \(\sum_j\partial_{b_j}L=0\). Translating
all centers translates the inverse by the same amount, giving
\(\sum_j\nabla_{a_j}L=g\). These are useful checks on parameter gradients.

This is classical implicit differentiation, also used in
[Bai, Kolter and Koltun, *Deep Equilibrium Models*](https://vladlen.info/papers/deq.pdf),
Theorem 1 and equation (11). The forward iteration need not be retained on the
autodiff tape. At a finite numerical iterate, evaluating these formulas gives
an approximation to the inverse's derivative; it is generally different from
the derivative of that finite sequence of iterations.

### A smaller covariance solve

Let \(m\) be the number of mixture terms and let the \(j\)-th row of
\(B\in\mathbb R^{m\times d}\) be
\(\sqrt{p_j}\,r_j^T/\sqrt{2\nu s}\). Then

\[
H=I_d-B^TB,\qquad
\lambda=g+B^T(I_m-BB^T)^{-1}Bg.
\]

This is the Woodbury identity. Both matrices have eigenvalues in
\([1-q,1]\). Although \(B\) has rank at most \(m-1\), redundant centers
do not make these shifted systems singular. When \(m<d\), the second
formula replaces a \(d\)-dimensional solve with an \(m\)-dimensional
solve, costing roughly \(O(dm^2+m^3)\) arithmetic rather than forming
and solving a dense \(d\)-dimensional system. The benefit depends on the
number of terms as well as the spatial dimension.
The nonlinear part varies only along the span of the center differences;
orthogonal directions remain affine. A small number of terms therefore gives
both this computational saving and a restriction on the represented field.

A matrix action needs only

\[
Hw=w-\frac1{2\nu s}\sum_jp_jr_j(r_j\cdot w),
\]

which costs \(O(md)\) arithmetic without storing a covariance matrix.
It also supports interval residual checking of a proposed gradient.

### Numerical Torch and JAX API

Both `omnibias.torch.gaussian_heat` and `omnibias.jax.gaussian_heat` expose
`gaussian_heat_inverse(target, centers, biases, viscosity=1, time=1, *, max_iterations=128, tolerance=None, adjoint="auto")`.
Targets have trailing dimension \(d\), with optional leading batch axes;
centers have shape \((m,d)\), and biases have shape \((m,)\). The
viscosity and heat age are positive scalars. Differentiation with respect to
trainable inputs uses the implicit equation and an adjoint solve.
Targets use `float32` or `float64`; the other parameters are cast to the
target's dtype. A subsequent exact-model check must preserve those effective
stored values, including any dtype conversion.

The result contains physical `value`, batchwise `converged`,
`residual_bound`, and `contraction_bound`. These are numerical diagnostics:
`residual_bound` estimates the physical error from the contraction formula,
but it is not an interval certificate. Invalid or unconverged outputs have
NaN values. The default absolute physical tolerance is 64 times the dtype's
machine epsilon; callers should choose a tolerance appropriate to their scale.
The adjoint option is `"dense"`, `"low_rank"`, or `"auto"`; the automatic
choice uses the smaller system. Parameter updates must keep the model in
the valid strict contraction regime to retain this global inverse guarantee.

The Torch backward rule retains dependence on the implicit solution for double
backward. The JAX custom JVP supports `jit`, `vmap`, forward and reverse
differentiation. Tests compare all five parameter gradients between backends,
check mixed second derivatives, and compare scalar target derivatives through
order four against independent high-precision inverse solutions. These are
numerical checks; the certificate below covers a supplied first target
derivative. Torch `torch.func` forward-mode transforms are not implemented by
this custom autograd function.
Check that every entry converged before differentiating a batch. Masking failed
NaN outputs does not provide a shared-parameter gradient guarantee in JAX;
Torch additionally handles failed samples whose upstream gradients are zero.

### Certifying a proposed target gradient

The rigorous layer can check a numerical proposal separately. Suppose a
root certificate proves \(\lVert\widehat x-x_*\rVert_2\le\varepsilon\),
where \(\widehat x=s\widehat y\) uses the exact heat age. For a fixed
upstream vector \(g\), the true target vector–Jacobian product is
\(z_*=sH(x_*)^{-1}g\). Let \(\widehat z\) be a proposed value and
rigorously bound

\[
\eta\ge\lVert H(\widehat x)\widehat z-sg\rVert_2,\qquad
L_H=\frac{D^3}{16\nu^2s^2}.
\]

The covariance derivative is the third centered moment:

\[
DC(x)[h]=\frac1{2\nu s}
  \mathbb E_p[rr^T(r\cdot h)].
\]

Since \(\lVert r_j\rVert_2\le D\) and every directional variance is at
most \(D^2/4\), this gives
\(\lVert DH(x)[h]\rVert_2\le L_H\lVert h\rVert_2\). Thus

\[
\boxed{\quad
\lVert\widehat z-z_*\rVert_2
\le\frac{\eta+L_H\varepsilon\lVert\widehat z\rVert_2}{1-q}.
\quad}
\]

Indeed, multiply the gradient error by \(H(x_*)\), add and subtract
\(H(\widehat x)\widehat z\), and use
\(\lVert H(x_*)^{-1}\rVert_2\le1/(1-q)\). This bound includes both
root-location error and linear-solve error. It concerns a fixed upstream
vector; a claim about an entire loss gradient must also account for any error
in that upstream vector.

`certify_gaussian_target_vjp(model, target, scaled_point, upstream, proposal)`
in `omnibias.core.verified.gaussian_sensitivity` checks this bound with interval
matrix actions and outward norm bounds. It returns a
`GaussianTargetVJPCertificate` or `None`, with `root_certificate`, `proposal`,
`enclosure`, `error_bound`, `residual_bound`, `jacobian_lipschitz_bound`, and
serialized `certificate` fields. The enclosure contains the true physical
target gradient for the declared model and upstream vector. An inaccurate
finite proposal may produce a valid but wide enclosure.

The interval model must describe the same stored parameters as the numerical
layer. Converting each stored float to `Fraction.from_float` preserves its
exact binary value; replacing it with a rounded decimal changes the model.
This check establishes a gradient of the ideal real solution defined by those
stored values. It does not certify hardware autodiff arithmetic, parameter
gradients other than the supplied target product, or a formal Lean proof.

For example, a Torch backward pass can propose a gradient, then the independent
core checker can enclose the ideal target derivative:

```python
import torch

from omnibias.core.verified.gaussian_sensitivity import certify_gaussian_target_vjp
from omnibias.torch.gaussian_heat import gaussian_heat_inverse

centers = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
biases = torch.zeros(2, dtype=torch.float64)
tensor_target = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
numeric = gaussian_heat_inverse(
    tensor_target, centers, biases, viscosity=1.0, time=2.0, tolerance=1e-12,
)
assert bool(numeric.converged)
proposed_gradient, = torch.autograd.grad(numeric.value.sum(), tensor_target)

def exact(value):
    return Fraction.from_float(float(value))

exact_model = GaussianHeatMap(
    tuple(tuple(exact(v) for v in row) for row in centers.tolist()),
    tuple(exact(v) for v in biases.tolist()),
    viscosity=exact(1.0), time=exact(2.0),
)
gradient_certificate = certify_gaussian_target_vjp(
    exact_model,
    tuple(exact(v) for v in tensor_target.detach().tolist()),
    tuple((numeric.value.detach() / 2.0).tolist()),
    upstream=(1.0,),
    proposal=tuple(proposed_gradient.tolist()),
)
assert gradient_certificate is not None
assert gradient_certificate.error_bound < 1e-10
```

## Exact exponential coordinates and bias derivatives

The Gaussian family also has an exact connection to the
[exponential-polynomial identity algebra](nonlinear-pde-identities.md).
Introduce \(y=x/s\), \(z=1/s\), and

\[
T(y,z)=\sum_j\exp\left(b_j+\frac{a_j\cdot y}{2\nu}
                         -\frac{\lVert a_j\rVert_2^2z}{4\nu}\right).
\]

Then \(T_z+\nu\Delta_yT=0\), directly from the frequency of each
exponential, and

\[
\tau=(4\pi\nu s)^{-d/2}
       e^{-\lVert x\rVert_2^2/(4\nu s)}T(x/s,1/s),
\qquad
U=y-2\nu z\nabla_y\log T.
\]

These coordinates turn the mixture into a finite exponential sum with exact
rational rates when the declared metadata is rational. Free biases can be
additional coordinates in that algebra. Each lifted derivative multiplies a
summand by its corresponding rate; for example,

\[
\partial_y^\alpha\partial_z^mT
=\sum_j\left(\frac{a_j}{2\nu}\right)^\alpha
\left(-\frac{\lVert a_j\rVert_2^2}{4\nu}\right)^m
\exp\left(b_j+\frac{a_j\cdot y}{2\nu}
               -\frac{\lVert a_j\rVert_2^2z}{4\nu}\right).
\]

Normalized coalescing bias shifts recover the exponential profile derivatives
used in these packs. To recover physical PDE derivatives, the coordinate
change must also be differentiated. With \(s=t+s_0\), its operators are

\[
\partial_{x_i}=z\partial_{y_i},\qquad
\partial_t=-z\,y\cdot\nabla_y-z^2\partial_z.
\]

The operators act on every factor, including explicit occurrences of \(y\)
and \(z\) in the velocity. Their commutator is zero, as physical mixed
derivatives require. This supplies an exact identity check of the Burgers
equations on \(z>0\), where \(T>0\). It does not turn a bias derivative
alone into a physical time derivative, or allow arbitrary activation
replacement while retaining the heat identity.

## The inverse field satisfies a nonlinear PDE

Keep the centers, biases and viscosity fixed while heat age varies. On
\(s>D^2/(8\nu)\), let \(X(v,s)\) denote the global inverse defined by
\(U(X(v,s),s)=v\), and set

\[
J=D_vX,\qquad G=J^{-1}=D_xU(X(v,s),s).
\]

The inverse field satisfies the hodograph equation

\[
\boxed{\quad
\partial_sX_i=v_i+\nu\sum_{k,l=1}^d(GG^T)_{kl}
                   \partial_{v_kv_l}X_i.
\quad}
\]

This is a coupled nonlinear parabolic system in target coordinates. Its
diffusion matrix depends on the inverse field's first derivatives; changing
coordinates does not turn it into a constant-coefficient linear equation.
To verify the formula, differentiate the inverse identity in time and twice
in the target coordinates:

\[
X_s=-JU_s=v-\nu J\Delta_xU,\qquad
X_{i,kl}=-\sum_{a,b,c}J_{ia}U_{a,bc}J_{bk}J_{cl}.
\]

Contracting the second identity with \(GG^T\) gives
\(\sum_{kl}(GG^T)_{kl}X_{,kl}=-J\Delta_xU\). The first equality uses
the original Burgers equation and \(J(D_xU)v=v\). These identities hold
for any smooth locally invertible vector Burgers field; this Gaussian
family supplies the global inverse and explicit bounds.

There is also a scalar dual description. With the normalized heat density
defined above, put

\[
\phi(x,s)=-2\nu\log\tau(x,s),\qquad
\Psi(v,s)=\sup_x\{v\cdot x-\phi(x,s)\}.
\]

The covariance bound proves

\[
\frac{1-q}{s}I\preceq D_x^2\phi\preceq\frac1sI,
\qquad
sI\preceq D_v^2\Psi\preceq\frac{s}{1-q}I.
\]

Thus the supremum has a unique optimizer \(X(v,s)\) for every target,
\(\nabla_v\Psi=X\), and \(D_v^2\Psi=J\). The potential heat identity
\(\phi_s+\lVert\nabla\phi\rVert_2^2/2=\nu\Delta\phi\) and the envelope
identity \(\Psi_s=-\phi_s(X,s)\) give

\[
\boxed{\quad
\Psi_s=\frac12\lVert v\rVert_2^2
       -\nu\operatorname{tr}\big[(D_v^2\Psi)^{-1}\big].
\quad}
\]

Differentiating this equation in \(v\) recovers the vector hodograph
equation, using symmetry of \(J\) and of the third derivatives of
\(\Psi\). The common heat prefactor matters here: if it is omitted from
\(\tau\), the resulting dual potential gains
\(\nu d\log(4\pi\nu s)\), and its time equation gains \(+\nu d/s\).
Velocity and its inverse remain the same.

Both equations have diffusion matrix \(\nu G^2\) in this gradient
case, with bounds

\[
\frac{\nu(1-q)^2}{s^2}I\preceq\nu G^2\preceq\frac\nu{s^2}I.
\]

They are uniformly parabolic on any compact heat-age interval strictly
above the threshold. This establishes the type and regularity of the
represented solution; it is not a general existence or uniqueness theorem
for arbitrary initial data in the inverse equations.

The [forward inverse Taylor jets](gaussian-inverse-jets.md) evaluate the
diffusion term without storing a third-order tensor. At a fixed target and
age, freeze \(h_r=G_{:,r}\) and compute
\(X(v+\epsilon h_r,s)=x_{0,r}+x_{1,r}\epsilon+x_{2,r}\epsilon^2+\cdots\).
Then the right side is \(v+2\nu\sum_r x_{2,r}\). The factor two is the
ordinary Taylor normalization, and the directions must remain fixed during
each differentiation. Integration tests check this identity for a coupled
two-dimensional mixture at multiple heat ages against the implicit time
derivative and an independent posterior-moment formula. They also check the
dual Hessian and the heat-prefactor term. These floating-point residual
checks complement the derivation; they are not continuum error certificates.

Varying heat age must preserve the original biases. Resetting an equal-logit
anchor or holding its posterior probabilities fixed as age changes defines
a different Gaussian family. The existing numerical layer differentiates
heat age, while the forward inverse-series API takes a target path at a fixed
age. Joint target-and-age Taylor propagation would require an additional
series recurrence.

Hodograph and Legendre transformations are established methods. For primary
examples, [Arsie, Lorenzoni and Moro, *Integrable viscous conservation
laws*](https://arxiv.org/abs/1301.0950), Section 5, writes the hodograph
derivative identities for viscous conservation laws. [Brydges and Marchetti,
*On the Virial Series for a Gas of Particles with Uniformly Repulsive Pairwise
Interaction*](https://arxiv.org/abs/1403.1621) studies a Legendre-transformed
viscous Hamilton–Jacobi equation in a different application. The formulas
here are derived for the declared Gaussian family; no new general
transformation theorem is claimed.

## Scope and prior work

At zero target, the fixed-point update is Gaussian mean shift. See
[Carreira-Perpiñán, *Generalised Blurring Mean-Shift Algorithms for Nonparametric
Clustering*](https://faculty.ucmerced.edu/mcarreira-perpinan/papers/cvpr08c.pdf),
equations (1a)–(1b), for the classical fixed-center Gaussian mean-shift update,
and the author's [review of mean-shift algorithms](https://arxiv.org/abs/1503.00687)
for its relation to Gaussian mixtures, mode finding and multiple inverses.
The centers here remain fixed throughout the inverse solve; the distinct
blurring algorithm updates its centers.

The exact diameter bound deliberately ignores the biases and may be
conservative for a given mixture. It establishes a useful uniform regime, not
the complete parameter region of invertibility. Floating-point stagnation,
underflow, overflow, a nearly unit contraction constant and resource limits
can prevent the requested enclosure accuracy. They do not invalidate the
underlying positive heat solution. This API does not fit a mixture to general
initial data, solve arbitrary PDE systems, or preserve the global inverse claim
after arbitrary activation substitutions.
