# Certified Gaussian inverse approximations on target segments

The [Gaussian heat inverse](gaussian-heat-inverse.md) can be compiled into a
Chebyshev polynomial on a declared real target segment. Construction certifies
the sampled inverse solutions and a uniform analytic interpolation error.
Subsequent polynomial evaluation needs no inverse iteration. The model,
segment and coefficients are fixed in the resulting artifact.

This works with arbitrary declared rational Gaussian biases in the sufficient
regime \(q=D^2/(8\nu s)<1\). It does not require an exact inverse base point
or equal Gaussian logits. The construction combines classical complex
analysis and Chebyshev interpolation; it makes no claim of a new general PDE
solver or a universal speed improvement.

## A global complex domain

Write the physical inverse as

\[
X(v)=sv+\mu(X(v)),\qquad F(v)=X(v)-sv-a_0,
\]

where \(a_0\) is the first Gaussian center. All centers, biases, viscosity
and heat age remain fixed. The affine term is kept exact, so only the bounded
nonlinear correction \(F\) needs polynomial approximation.

`certify_gaussian_inverse_tube(model, spatial_radius=...)` in
`omnibias.core.verified.gaussian_tube` supplies a
`GaussianInverseTubeCertificate` or returns `None` when the requested proof
does not succeed. An omitted spatial radius selects a conservative default.
For a positive rational spatial radius \(r\), put

\[
\beta=\frac{D^2r^2}{16\nu^2s^2}<1-q,\qquad
\kappa=\frac{q}{1-\beta}<1,
\]

\[
B=\frac rs\left[1-q\left(1+\frac{\beta}{3(1-\beta)}\right)\right],
\qquad M^2=\frac{D^2}{1-\beta}.
\]

These exact rational quantities certify a holomorphic inverse for every
complex target with \(\lVert\operatorname{Im}v\rVert_2<B\). Its solution
is unique in the closed spatial tube
\(\lVert\operatorname{Im}X\rVert_2\le r\), and
\(\lVert F(v)\rVert_2\le M\). Other complex roots outside that spatial
tube are not excluded. All real targets are included, without a bound on
their real parts.

Here is the reason the bound is global. At any real spatial location the
tilted Gaussian weights are positive. Adding an imaginary displacement
changes their phases through an arc of half-width at most
\(w=D\lVert\operatorname{Im}X\rVert_2/(4\nu s)\). Rotating that arc
gives a partition modulus at least \(\cos w\); the pairwise covariance
identity bounds the fixed-point derivative by \(q\sec^2w\). The inequalities
\(\cos^2w\ge1-w^2\) and
\(\tan w\le w+w^3/[3(1-w^2)]\) give the stated contraction and imaginary
displacement bounds. The fixed-point map preserves the complete closed
spatial tube. The contraction theorem and holomorphic implicit function
theorem then give the inverse and its holomorphic dependence on the target.
Finally, the numerator for \(\mu-a_0\) has norm at most \(D\), giving
\(M\le D/\sqrt{1-\beta}\).

If all centers coincide, the inverse is the entire affine map
\(X(v)=sv+a_0\). The certificate uses `None` for both tube radii and zero
for `correction_bound_squared`.

## From the tube to a uniform interpolation bound

Choose exact real vectors \(v_0,h\), a half-width \(L>0\), and the target
line \(v(t)=v_0+th\) for \(-L\le t\le L\). Set \(z=t/L\). A rational
Bernstein ellipse parameter \(\rho>1\) is accepted when

\[
L\lVert h\rVert_2\frac{\rho-\rho^{-1}}2<B.
\]

The left side bounds the imaginary target displacement throughout the
ellipse. The implementation checks its squared form with exact rational
arithmetic. Thus \(f(z)=F(v_0+Lzh)\) is holomorphic on a neighborhood of
the closed ellipse and has complex Euclidean norm at most \(M\).

Let \(p_N\) be the degree-\(N\) polynomial interpolating the ideal values
\(f(\cos(j\pi/N))\), \(0\le j\le N\). Then

\[
\sup_{-1\le z\le1}\lVert f(z)-p_N(z)\rVert_2
\le E_{\mathrm{analytic}}
=\frac{4M\rho^{-N}}{\rho-1}.
\]

This is the classical Chebyshev interpolation estimate; see
[Trefethen, *Approximation Theory and Approximation Practice*, Theorem 8.2](https://github.com/chebfun/ATAP/blob/development/chap8.m).
It holds without a dimension factor: the vector-valued contour integral
bounds coefficient \(k\ge1\) by \(2M\rho^{-k}\), and each higher
Chebyshev mode aliases to a mode of degree at most \(N\) on the Lobatto
grid. Their difference is bounded by two on the real interval. Summing that
geometric tail gives the bound above.

## Accounting for computed samples and coefficients

The ideal cosine nodes usually cannot be stored exactly in binary64. The
compiler encloses each ideal node, selects a nearby stored node, and forms
its target with exact rational affine arithmetic. It certifies the inverse
at that target and transports the correction enclosure to the ideal target.
The real global transport bound is

\[
\lVert D_vF\rVert_2=\lVert D_vX-sI\rVert_2
\le\frac{sq}{1-q}.
\]

It follows from \(sI\preceq D_vX\preceq sI/(1-q)\). Multiplying this
constant by a rigorous node-target displacement bound accounts for node
rounding; the root certificate accounts separately for inverse-solve error.

An interval DCT-I then encloses the ideal interpolant's ordinary coefficients
in \(p_N(z)=\sum_{k=0}^N c_kT_k(z)\). With \(\sigma_0=\sigma_N=1/2\)
and other \(\sigma_j=1\), the normalization is

\[
c_k=\frac2N\sigma_k\sum_{j=0}^N
       \sigma_j f(\cos(j\pi/N))\cos(kj\pi/N).
\]

Both sets of endpoint weights matter. The stored coefficients use this
ordinary expansion; evaluation does not apply another endpoint half-weight.
For each stored float coefficient \(\widehat c_{k,i}\), its enclosing
interval gives a rigorous radius \(e_{k,i}\). Since \(|T_k(z)|\le1\),
the coefficient contribution to component \(i\)'s error is bounded by
\(E_i=\sum_k e_{k,i}\). Therefore the stored nominal polynomial obeys

\[
\left|X_i(v_0+th)-\left[s(v_0+th)+a_0+
     \sum_{k=0}^N\widehat c_kT_k(t/L)\right]_i\right|
\le E_{\mathrm{analytic}}+E_i.
\]

The artifact stores these coordinatewise bounds as `uniform_error_bounds`.
They are not a single Euclidean error bound. The analytic interpolation bound
itself is Euclidean; a Euclidean total bound can be obtained separately using
\(E_{\mathrm{analytic}}+\sqrt{\sum_iE_i^2}\).

## Building and evaluating an approximation

```python
from fractions import Fraction

from omnibias.core.verified.gaussian_chebyshev import approximate_gaussian_inverse
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_tube import certify_gaussian_inverse_tube

model = GaussianHeatMap(
    ((-1,), (1,)), (0, Fraction(1, 3)), viscosity=1, time=2,
)
tube = certify_gaussian_inverse_tube(model)
assert tube is not None

approximation = approximate_gaussian_inverse(
    tube, (Fraction(1, 4),), (1,), half_width=Fraction(1, 4),
    degree=24, ellipse_parameter=2,
)
assert approximation is not None
assert max(approximation.uniform_error_bounds) < Fraction(1, 1_000_000)

t = Fraction(1, 8)
nominal = approximation.evaluate(t)  # Exact rational evaluation of stored floats.
enclosure = approximation.enclose(t)  # Contains the true inverse at v0 + t*h.
```

`approximate_gaussian_inverse(tube, base_target, direction, *, half_width=1,
degree=32, ellipse_parameter=None, max_iterations=128,
absolute_tolerance=1e-12)` returns a frozen `GaussianInverseApproximation`
or `None` if construction cannot complete its proof. Invalid argument types
or domains raise. The optional ellipse parameter must pass the tube check;
the automatic choice uses the available tube width and line scale.
Except for the exact affine case, construction requires the rigorous
transcendental backend for the cosine nodes and sampled inverse checks.

The record retains the tube, segment, ellipse, coefficient intervals,
analytic error, coordinatewise coefficient errors, and serialized certificate.
Its `degree` describes the returned polynomial. Identical centers give an
exact degree-zero correction; a constant target line can use a degree-zero
enclosed sample.

Both core evaluation methods accept an exact `int` or `Fraction` parameter
inside the declared segment. `evaluate` evaluates the nominal polynomial
exactly, treating every stored coefficient as its exact binary value.
`enclose` uses outward-rounded interval Clenshaw evaluation and adds the
uniform error bounds, so its result also accounts for evaluation rounding.
Range and resource limits can refuse evaluation.

The tensor adapters expose
`evaluate_gaussian_inverse_approximation(parameter, approximation)`. They
evaluate the fixed approximation with ordinary numerical arithmetic and
differentiate its parameter input. Their `value` and `valid` outputs do not
certify runtime arithmetic or derivatives of the true inverse. The original
model parameters are frozen; this is not differentiation through model
training or approximation construction.

```python
import numpy as np
import torch
from omnibias.torch.gaussian_chebyshev import evaluate_gaussian_inverse_approximation as torch_curve

queries = torch.tensor([-0.25, 0.0, 0.25], dtype=torch.float64, requires_grad=True)
torch_result = torch_curve(queries, approximation)
assert bool(torch_result.valid.all())
torch_result.value.sum().backward()
assert queries.grad is not None and bool(torch.isfinite(queries.grad).all())
```

```python
import jax
import jax.numpy as jnp
from omnibias.jax.gaussian_chebyshev import evaluate_gaussian_inverse_approximation as jax_curve

jax.config.update("jax_enable_x64", True)
evaluate_curve = jax.jit(lambda parameter: jax_curve(parameter, approximation))
jax_result = evaluate_curve(jnp.asarray(queries.detach().numpy()))
assert bool(jax_result.valid.all())
np.testing.assert_allclose(jax_result.value, torch_result.value.detach().numpy(), atol=2e-13)
```

## Certifying derivatives of the true inverse

`certify_gaussian_inverse_derivative(approximation, order)` in
`omnibias.core.verified.gaussian_chebyshev_derivative` constructs a
`GaussianInverseDerivative` with a uniform derivative error on the same
segment. Its `evaluate(t)` gives the exact nominal polynomial derivative;
`enclose(t)` also accounts for approximation error and outward evaluation
rounding. Construction can refuse when its resource or range bounds cannot
be met. Order zero retains the original approximation's value bound.

The quantity enclosed at order \(k\) is the raw directional derivative

\[
\frac{d^k}{dt^k}X(v_0+th),\qquad -L\le t\le L.
\]

It is not divided by \(k!\). In multiple dimensions, one target line gives
one directional derivative, not the full Jacobian or higher derivative tensor.
The original Gaussian model and the line remain fixed. Differentiating the
tensor adapter's nominal polynomial alone does not supply this guarantee.

The proof uses the analytic coefficient bound together with Chebyshev
aliasing. For \(k\ge1\), define

\[
C(n,k)=\max_{-1\le z\le1}|T_n^{(k)}(z)|
=T_n^{(k)}(1)
=\frac{\prod_{j=0}^{k-1}(n^2-j^2)}{(2k-1)!!}
\]

when \(n\ge k\), and set \(C(n,k)=0\) otherwise. The endpoint maximum
follows from the [Chebyshev and Gegenbauer differentiation identities](https://dlmf.nist.gov/18.9)
and the [normalized Gegenbauer integral representation](https://dlmf.nist.gov/18.10.E4).
The derivative series converges uniformly on the real interval because its
coefficient decay is geometric while these derivative factors grow polynomially.

Let \(r=\rho^{-1}\), \(K=\max(N+1,k)\), and
\(S_C=\sum_{n=K}^{\infty}C(n,k)r^n\). Every omitted mode aliases to
a mode of degree at most \(N\), so a valid Euclidean interpolation bound is

\[
E_{\mathrm{analytic},k}
=\frac{2M}{L^k}
\left[S_C+C(N,k)\frac{r^K}{1-r}\right].
\]

The implementation bounds \(M\) upwards and computes the remaining
expression with rational arithmetic. If \(k>N\), the alias contribution
vanishes. The analytic tail generally remains nonzero even though the stored
polynomial's derivative is zero. There is no extra \(\lVert h\rVert^k\)
factor: the direction already enters \(f(z)=F(v_0+Lzh)\), and \(L^{-k}\)
converts derivatives in \(z\) to derivatives in \(t\).

The infinite sum has a finite positive formula. Put \(B=K+k-1\). Since
\(C(n,k)=2^{k-1}(k-1)!n\binom{n+k-1}{2k-1}\), Vandermonde's identity
and the differentiated geometric series give

\[
S_C=2^{k-1}(k-1)!r^K
\sum_{j=0}^{2k-1}\binom{B}{2k-1-j}
\left[
\frac{(K+j)r^j}{(1-r)^{j+1}}
+\frac{(j+1)r^{j+1}}{(1-r)^{j+2}}
\right].
\]

All summands are nonnegative. This avoids subtracting a large finite sum
from an infinite-series total and is tighter than replacing \(C(n,k)\)
with \(n^{2k}/(2k-1)!!\).

The ideal interpolant's coefficient uncertainty contributes the additional
coordinatewise bound

\[
E_{k,i}^{\mathrm{coeff}}
=L^{-k}\sum_{n=k}^N e_{n,i}C(n,k).
\]

The returned `uniform_error_bounds` are
\(E_{\mathrm{analytic},k}+E_{k,i}^{\mathrm{coeff}}\), one per coordinate.
The analytic part remains Euclidean. A Euclidean total bound is
\(E_{\mathrm{analytic},k}+\sqrt{\sum_i(E_{k,i}^{\mathrm{coeff}})^2}\).
The nominal coefficient transformation is exact rational arithmetic. The
affine trend contributes \(sh\) at first order and zero at higher orders.

```python
from omnibias.core.verified.gaussian_chebyshev_derivative import (
    certify_gaussian_inverse_derivative,
)

derivative = certify_gaussian_inverse_derivative(approximation, 1)
assert derivative is not None
nominal_slope = derivative.evaluate(t)
slope_enclosure = derivative.enclose(t)  # Contains dX(v0 + t*h)/dt.
```

Small value error alone does not imply small derivative error. Endpoint
derivative factors amplify both interpolation and coefficient uncertainty;
high orders can yield wide enclosures despite accurate values. Increasing
degree can reduce the analytic tail, while more accurate sampled roots and
coefficients may be needed to reduce the other error. The requested order
is subject to explicit arithmetic and resource limits.

The proof ellipse also matters. For the same model and segment above,
\(\rho=4\) still fits inside the certified tube. Using that ellipse and
more accurate sampled roots gives useful bounds through order three:

```python
derivative_approximation = approximate_gaussian_inverse(
    tube, (Fraction(1, 4),), (1,), half_width=Fraction(1, 4),
    degree=24, ellipse_parameter=4, absolute_tolerance=1e-14,
)
assert derivative_approximation is not None
for order, limit in enumerate(
    (Fraction("3e-10"), Fraction("2e-7"), Fraction("5e-5")), start=1,
):
    bound = certify_gaussian_inverse_derivative(derivative_approximation, order)
    assert bound is not None
    assert max(bound.uniform_error_bounds) < limit
```

These are absolute uniform errors for raw derivatives, before the additional
outward evaluation rounding in `enclose`. They concern this fixed scalar
example. Increasing degree alone can worsen the certificate once uncertainty
in the sampled coefficients dominates; the derivative endpoint weights then
amplify more uncertain coefficients. A requested sample tolerance also does
not promise that floating-point root proposals will attain that tolerance.

## Measured cost at the same requested enclosure accuracy

The [benchmark script](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/gaussian_inverse_approximation.py)
compares `approximation.enclose(t)` with a fresh `inverse_gaussian_heat`
certificate on the scalar example above. Both must contain an independent
80-digit reference and have coordinate half-width at most \(10^{-6}\).
This times rigorous point enclosures, including their interval arithmetic;
it does not time the numerical tensor adapters.

The [recorded run](benchmarks/gaussian_inverse_approximation.json) used 17 exact
rational queries spread across the segment. Each timed call evaluates the
whole set. The following per-query figures divide the median batch time by 17,
using nine samples after two warmups. The first timed batch, all raw samples,
build costs, references, enclosure widths, environment and source hashes are
retained in the artifact. Timing began after accuracy validation; process
startup and the independent reference calculation are excluded.

| Method | Median per query | Largest coordinate half-width |
| --- | ---: | ---: |
| Certified polynomial enclosure | 0.0908 ms | \(5.2902\times10^{-7}\) |
| Fresh certified inverse | 3.1698 ms | \(9.9238\times10^{-7}\) |

The polynomial query was about 34.9 times faster in this run. Building its
tube and approximation cost 170.67 ms, giving an estimated break-even point
of 56 queries at these measured rates. The common model-construction cost is
recorded separately and cancels from that comparison. The fresh solver starts
from its default zero proposal at each target, with no warm start.

These measurements cover one fixed scalar family and segment. They do not
predict training speed, tensor-kernel speed, changing-model costs, or the
degree needed on other domains. Changing the family or segment requires a
new approximation and a new proof.

## Scope and prior work

The original approximation certifies values on the specified real target
segment; the separate derivative artifact adds the derivative bounds above.
Increasing degree reduces the analytic tail, but node, solve and coefficient
errors can limit the useful accuracy. A narrow complex tube or a long segment
can require a large degree. Construction cost includes the certified samples;
the value timings above do not measure derivative certification or evaluation.

The positive-exponential half-plane nonvanishing argument appears explicitly
in [Sao, *Ghosts of Softmax*, Lemma 4.5 and Theorem 4.6](https://arxiv.org/html/2603.13552v1#S4.SS3).
Only that geometric result is relevant here. The two-center inverse also
rescales to the classical Curie–Weiss self-consistency equation; see
[Deger and Flindt, equation (9)](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.033009).
Those connections do not establish novelty of the inverse-tube bound.
Certificates record checked arithmetic and stated analytic arguments; they
are not formal Lean proofs or guarantees for arbitrary PDE families.
