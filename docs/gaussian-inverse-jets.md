# Forward Taylor jets of a Gaussian inverse

The [Gaussian heat inverse](gaussian-heat-inverse.md) can be differentiated to
arbitrary finite order by propagating Taylor coefficients through its implicit
equation. One matrix factorization is reused across orders. The exact core
primitive computes rational coefficients for a declared family and target
polynomial; a separate analytic certificate bounds the remaining Taylor tail.
This supplies derivatives in a forward pass without recursively differentiating
a numerical solver's iteration history.

Implicit Taylor differentiation and factorization reuse are classical. See
[Wagner, Walther and Schaefer, *On the efficient computation of high-order
derivatives for implicitly defined functions*, §4.1](https://arxiv.org/pdf/0912.2208)
and [Griewank, Utke and Walther, *Evaluating higher derivative tensors by forward
propagation of univariate Taylor series*](https://doi.org/10.1090/S0025-5718-00-01120-0).
The implementation combines that method with the Gaussian covariance bound and
an explicit complex domain. It makes no claim of a new general calculus theorem.

## The exact family and its base point

Fix rational centers \(a_j\), a rational base point \(x_0\), viscosity
\(\nu>0\), and heat age \(s>0\). Supply strictly positive rational
posterior weights \(p_j\) with \(\sum_jp_j=1\). These data define the
Gaussian log weights, up to a common additive constant, by

\[
b_j=\log p_j+\frac{\lVert x_0-a_j\rVert_2^2}{4\nu s}.
\]

The log weights need not be rational, and the implementation does not round
them or evaluate their logarithms. At \(x_0\), the posterior is exactly the
supplied \(p\). Thus

\[
\mu_0=\sum_jp_ja_j,\qquad v_0=\frac{x_0-\mu_0}{s}
\]

is an exact inverse pair. The declared target is a **polynomial path**
\(v(t)=\sum_{n=0}^Nv_nt^n\); coefficients beyond its supplied order are
zero. The time parameter \(t\) in this path is a decision parameter, with
the heat age \(s\) held fixed.

`gaussian_inverse_series(centers, posterior, base_point, target_coefficients, *, viscosity=1, time=1, budget=None)`
from `omnibias.core.gaussian_jet` constructs a frozen `GaussianInverseSeries`.
Inputs must be integers or `Fraction` objects, excluding booleans. The
constant target must equal the exact \(v_0\) above. Construction requires
\(q=D^2/(8\nu s)<1\), where \(D\) is the center diameter.

The returned `coefficients[n]` is \(x_n=x^{(n)}(0)/n!\), an ordinary Taylor
coefficient. `evaluate(t)` evaluates the finite polynomial exactly and does
not add a remainder. The artifact also retains `centers`, `posterior`,
`base_point`, `target_coefficients`, `viscosity`, `time`, `diameter_squared`,
`contraction_bound`, `budget`, and the `order`, `dimension`, `mode_count`
properties.

Changing or rounding the posterior changes the represented family. To bind an
existing `GaussianHeatMap`, use
`GaussianInverseSeries.from_equal_logits(model, base_point, higher_target_coefficients)`.
It checks equality of the exact Gaussian logits at the base point, then uses
the exact posterior \(p_j=1/m\). It computes the constant target itself.
Unequal logits are refused. This preserves the existing velocity field, up to
the irrelevant common heat-potential normalization.

```python
from fractions import Fraction

from omnibias.core.gaussian_jet import GaussianInverseSeries
from omnibias.core.verified.gaussian_heat import GaussianHeatMap

model = GaussianHeatMap(
    ((0, 0), (1, 0), (0, 1)),
    (0, Fraction(1, 4), Fraction(1, 4)),
)
# v(t)=(-1/3,-1/3)+(t,0); trailing zeros request eight inverse orders.
series = GaussianInverseSeries.from_equal_logits(
    model, (0, 0), ((1, 0),) + ((0, 0),) * 7,
)
assert series.order == 8
assert series.posterior == (Fraction(1, 3),) * 3
assert series.coefficients[0] == (0, 0)
assert series.coefficients[1] == (Fraction(96, 85), Fraction(-6, 85))
```

The nonzero second coordinate of the first derivative exhibits the coupling:
changing only the first target component moves both coordinates of its inverse.

## Why one linear solve per order suffices

At the exact base point define

\[
C_0=\sum_jp_j(a_j-\mu_0)(a_j-\mu_0)^T,\qquad
H=I-\frac{C_0}{2\nu s}.
\]

The contraction bound gives \(H\succeq(1-q)I\), so it has a reusable
positive-definite factorization. To propagate the inverse series, introduce

\[
g_j(t)=\frac{a_j\cdot(x(t)-x_0)}{2\nu s},\quad
E_j(t)=e^{g_j(t)},\quad
Z(t)=\sum_jp_jE_j(t),\quad
M(t)=\sum_jp_ja_jE_j(t).
\]

Then \(\mu(x(t))=M(t)/Z(t)\). A common center can be subtracted from all
rates to reduce intermediate sizes; the normalized mean is unchanged after
restoring that center. If coefficients below order \(n\) are known, first
set the unknown \(g_{j,n}\) to zero and compute

\[
\widehat E_{j,n}=\frac1n\sum_{k=1}^{n-1}k\,g_{j,k}E_{j,n-k}.
\]

Use these provisional coefficients in the quotient recurrence
\(\mu_n=M_n-\sum_{k=1}^nZ_k\mu_{n-k}\), since \(Z_0=1\).
This produces \(\widehat\mu_n\). The missing dependence on \(x_n\)
is linear and equals \(C_0x_n/(2\nu s)\). Hence

\[
Hx_n=sv_n+\widehat\mu_n.
\]

After solving, commit \(g_{j,n}=a_j\cdot x_n/(2\nu s)\) and
\(E_{j,n}=\widehat E_{j,n}+g_{j,n}\), then update the moments for the
next order. Lower coefficients never need to be rebuilt. The exact core uses
one rational LDLT factorization and budgeted rational arithmetic.

For \(m\) terms, spatial dimension \(d\), and order \(N\), coefficient
work is \(O((m+d)N^2+mdN)\), in addition to factorization and \(N\)
linear solves. The exact implementation uses a dense factorization; the tensor
jet API below also supports the
[smaller covariance solve](gaussian-heat-inverse.md#a-smaller-covariance-solve).
Integer bit growth can dominate these scalar-operation
counts. A complete mixed derivative tensor still needs additional directions
and its associated storage.

The exponential derivatives appearing here also arise from normalized
coalescing bias shifts. Computing the inverse derivatives additionally needs
the quotient rule, composition and the implicit linear solve. A raw collapsed
difference is not a replacement for those operations.

## Trainable Torch and JAX jets

Both `omnibias.torch.gaussian_jet` and `omnibias.jax.gaussian_jet` expose
`gaussian_heat_inverse_jet(target_coefficients, centers, biases, viscosity=1, time=1, *, max_iterations=128, tolerance=None, adjoint="auto")`.
The input and output have shape `(N+1, ..., d)`, with ordinary Taylor
coefficients along the first axis. Shared centers have shape `(m, d)` and
biases `(m,)`; viscosity and heat age are scalars held fixed along the target
path. All supplied coefficients and parameters remain trainable.

For positive order, one Cholesky factorization is reused for every coefficient.
`adjoint="dense"` uses the spatial matrix, `"low_rank"` uses a mode-space
Woodbury solve, and `"auto"` chooses the smaller matrix. The low-rank path
does not form a `d` by `d` matrix. Order zero skips the jet factorization.

The result contains `coefficients`, `base_result`, `anchor_target`, and
`base_target_residual`. The numerical base point is `x0=coefficients[0]`;
`anchor_target=U(x0)` and `base_target_residual=U(x0)-target_coefficients[0]`.
Higher coefficients expand the inverse at this **actual anchor**, so their
formal target path has constant term `U(x0)`. Tightening the base tolerance
reduces this discrepancy but does not make the coefficients exact.

```python
import numpy as np
import torch
from omnibias.torch.gaussian_jet import gaussian_heat_inverse_jet as torch_inverse_jet

target_rows = [[float(value) for value in row] for row in series.target_coefficients]
torch_targets = torch.tensor(target_rows, dtype=torch.float64, requires_grad=True)
torch_centers = torch.tensor(model.centers, dtype=torch.float64)
torch_biases = torch.tensor([float(value) for value in model.biases], dtype=torch.float64)
torch_jet = torch_inverse_jet(torch_targets, torch_centers, torch_biases)
assert bool(torch_jet.base_result.converged)
assert bool(torch.isfinite(torch_jet.coefficients).all())
reference_coefficients = np.array([[float(value) for value in row] for row in series.coefficients])
np.testing.assert_allclose(torch_jet.coefficients.detach().numpy(), reference_coefficients, atol=2e-13)
torch_jet.coefficients[2].square().sum().backward()
assert torch_targets.grad is not None and bool(torch.isfinite(torch_targets.grad).all())
```

```python
import jax
import jax.numpy as jnp
from omnibias.jax.gaussian_jet import gaussian_heat_inverse_jet as jax_inverse_jet

jax.config.update("jax_enable_x64", True)
jax_jet = jax.jit(jax_inverse_jet)(
    jnp.asarray(target_rows), jnp.asarray(torch_centers.numpy()), jnp.asarray(torch_biases.numpy()),
)
assert bool(jax_jet.base_result.converged)
np.testing.assert_allclose(np.asarray(jax_jet.coefficients), reference_coefficients, atol=2e-13)
```

Torch supports reverse-mode and higher parameter derivatives; its custom base
inverse does not support `torch.func` forward-mode transforms. JAX supports
JIT, `vmap`, and higher differentiation through its custom JVP. The
[base inverse's batch-gradient contract](gaussian-heat-inverse.md) also applies.
A failed base yields NaN coefficients. Higher-order overflow yields NaNs from
that order onward even if `base_result.converged` is true, so inspect all
coefficients for finiteness. These numerical results do not carry the analytic
certificate below; that certificate requires the exact core artifact.

### Measured derivative evaluation

The [reproduction script](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/gaussian_inverse_jets.py)
compares the forward recurrence with recursively applying Torch autograd to
the same implicit inverse. Both return every ordinary coefficient through the
requested order and include a fresh base solve. One asymmetric, three-mode
scalar family has an exactly known rational base and a nonlinear target path.
Every returned coefficient is checked against the exact core reference.

The [recorded run](benchmarks/gaussian_inverse_jets.json) used CPU float64,
Torch eager execution, one computation thread, and a fresh process for each
method/order. Medians below use nine samples after two warmups; first execution,
all raw samples, resource limits, operands, software versions and source hashes
are retained in the artifact. All runs completed and passed the accuracy gate;
the largest absolute coefficient error was below \(5.56\times10^{-17}\).

| Highest order | Forward jet | Nested implicit AD |
| ---: | ---: | ---: |
| 1 | 9.914 ms | 9.887 ms |
| 2 | 10.582 ms | 12.332 ms |
| 4 | 10.624 ms | 17.852 ms |
| 6 | 10.199 ms | 45.687 ms |
| 8 | 10.271 ms | 346.560 ms |

Nested AD was slightly faster at order one; the forward recurrence was about
33.7 times faster at order eight in this run. These results describe this
scalar workload and exclude parameter backward, training, JIT compilation and
certificate construction. They do not establish a speedup for arbitrary PDEs.

## A complex domain and a rigorous Taylor tail

Real contraction proves global real invertibility, but the Taylor series can
meet complex singularities. A quantitative complex contraction supplies a
domain on which the series really converges.

Write a complex displacement as \(h=u+iy\). Absorb the real part into
positive tilted weights
\(\widetilde p_j\propto p_j\exp(a_j\cdot u/(2\nu s))\).
Set \(\theta_j=a_j\cdot y/(2\nu s)\) and
\(Z=\sum_j\widetilde p_je^{i\theta_j}\). The phase range has width at
most \(D\|y\|_2/(2\nu s)\). Rotation by its midpoint gives

\[
|Z|\ge\cos w>0,\qquad w=\frac{D\|y\|_2}{4\nu s}<\frac\pi2.
\]

This half-plane nonvanishing argument is also given in
[Sao, *Ghosts of Softmax*, Lemma 4.5 and Theorem 4.6](https://arxiv.org/html/2603.13552v1#S4.SS3).
Here the logits are exactly affine in space; no logit linearization is used.
The covariance additionally has the pairwise identity, with
\(d_{jk}=a_j-a_k\),

\[
C_{\mathbb C}=
\frac{1}{2Z^2}\sum_{j,k}\widetilde p_j\widetilde p_k
  d_{jk}d_{jk}^{T}e^{i(\theta_j+\theta_k)}.
\]

For complex unit vectors \(z,t\), Cauchy–Schwarz bounds the numerator's
bilinear form by
\(\sqrt{(z^*C_{\widetilde p}z)(t^*C_{\widetilde p}t)}\), where
\(C_{\widetilde p}\) is the real covariance under the positive tilted
weights. Its norm is at most \(D^2/4\). Consequently,

\[
\left\|\frac{C_{\mathbb C}}{2\nu s}\right\|_2
\le q\sec^2 w.
\]

The bound is independent of the real displacement. Two equally weighted
centers, with imaginary displacement along their separation, attain it.
It is therefore sharp as a uniform bound using only the diameter. This is a
bound on the complex fixed-point derivative; it does not assert positivity
of complex posterior weights or identify every family's nearest singularity.

To keep the certificate purely rational, obtain a checked binary64 upper
bound \(\overline L\) on \(\sqrt{D^2(1-q)}\) by exact integer
comparisons, and treat that upper bound as an exact rational. Choose

\[
r=\frac{2\nu s(1-q)}{\overline L},\qquad
\kappa=\frac{4q}{3+q}<1.
\]

On the complex ball \(\|h\|_2\le r\),
\(w\le\sqrt{1-q}/2\). The elementary inequality
\(\cos^2w\ge1-w^2\ge(3+q)/4\) proves the stated contraction bound,
with \(1-\kappa=3(1-q)/(3+q)>0\). No numerical sine, cosine or
exponential evaluation enters the certificate.

Let \(a_n\ge\|v_n\|_2\) be checked rational upper bounds. Any positive
rational parameter radius \(R\) satisfying

\[
s\sum_{n=1}^Na_nR^n\le(1-\kappa)r
\]

makes the fixed-point map preserve the complex ball and contract there
uniformly for \(|t|\le R\). Iteration gives a holomorphic inverse with
\(\|x(t)-x_0\|_2\le r\). The
[Cauchy derivative formula](https://dlmf.nist.gov/1.9.E31) then yields

\[
\|x_n\|_2\le\frac r{R^n},\qquad
\left\|x(t)-\sum_{n=0}^Nx_nt^n\right\|_2
\le r\frac{(\rho/R)^{N+1}}{1-\rho/R},\quad |t|\le\rho<R.
\]

This is a truncation bound for the exact Taylor coefficients. The enclosure
API additionally accounts for conversion of rational polynomial values to
binary64 endpoints.

## Certificate API

`certify_gaussian_inverse_series(series, *, parameter_radius=None)` from
`omnibias.core.verified.gaussian_analytic` returns a frozen
`GaussianSeriesCertificate` or inconclusive `None`. A supplied radius must
be a positive integer or `Fraction`. The automatic choice is
\(R=\min(1,(1-\kappa)r/(s\sum a_n))\), which satisfies the condition
because \(R^n\le R\) for \(R\le1\). Identical centers, or a constant
target path, yield an entire polynomial inverse and need no finite radius.

The result stores `series`, `parameter_radius`, `spatial_radius`,
`complex_contraction_bound` and a serialized `certificate` binding the
family, target path and computed coefficients. `parameter_radius=None` denotes
the entire case. `tail_bound(rho)` returns an exact rational Euclidean tail
bound; `enclose(t)` returns physical coordinate intervals for one exact real
parameter. The finite-domain case requires \(|t|<R\), strictly.

```python
from omnibias.core.verified.gaussian_analytic import certify_gaussian_inverse_series

analytic = certify_gaussian_inverse_series(series)
assert analytic is not None and analytic.parameter_radius is not None
t = analytic.parameter_radius / 4
tail = analytic.tail_bound(t)
solution_box = analytic.enclose(t)
assert tail > 0 and len(solution_box) == 2
```

No transcendental evaluation is needed to produce these bounds. The record
uses `transcend_backend="not_used"`; an explicit global strict certificate
mode still retains its own backend requirements. A digest checks record
integrity and is not a Lean-kernel proof. Invalid scalar types and evaluation
domains raise. Unproved requested radii, arithmetic range failures and
certification budget exhaustion return `None`; evaluation budget or endpoint
overflow raises rather than producing an infinite enclosure.

## Limits and useful checks

An approximate base point cannot be silently substituted: its derivatives are
anchored at the velocity of that point, which may differ from the requested
target. The exact posterior/base-point representation avoids this ambiguity.
It does not provide exact jets for every existing rational-bias model; the
equal-logit factory deliberately supports only cases it can bind exactly.

For equal one-dimensional centers \(\pm a\),
\(U(x)=(x-a\tanh(cx))/s\), where \(c=a/(2\nu s)\). Even if
\(q=ac<1\), putting \(\theta=\arccos\sqrt q\) gives the critical
point \(x_c=i\theta/c\), with target

\[
v_c=i\frac as\left(\frac\theta q-\tan\theta\right).
\]

The inverse branch from zero develops a square-root branch point there.
Thus global real invertibility does not justify unrestricted Taylor stepping.
The supplied radius is a sufficient, often conservative bound, not the exact
distance to the nearest singularity. The focused tests check independent
high-precision scalar and coupled roots, nonlinear polynomial target paths,
tail reduction with order, exact affine cases and outward rational rounding.
