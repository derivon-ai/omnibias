# omnibias-measure

Autograd-native **measure-theoretic integration**. omnibias has *three distinct
senses of "integral"* (see the [operator-surface capability
matrix](../operator-surface.md)); this package is the third:

1. the closed-form activation **antiderivative window** `S(z+b_hi) - S(z+b_lo)`
   with `S' = sigma` (`OperatorBlock(op="integral")`, `omnibias-torch`);
2. **domain quadrature** `sum_q w_q u(x_q)` over a field's collocation grid
   (`omnibias.fields.{torch,jax}.integrate`);
3. the **measure integral** `int f dmu` against an abstract measure -- *this
   package*.

Everything here is differentiable: the integrand `f`, the measure weights, and
the layer-cake softness `beta` all carry gradients, so a measure integral drops
into a network as a trainable component. The numpy `_core` is the bit-identical
reference; the `torch` and `jax` twins mirror it exactly (parity to `rtol=1e-9`
in float64) and ship trainable `nn.Module` / functional layer wrappers.

!!! note "This is quadrature-backed, not a symbolic Lebesgue integral"
    A `Measure` is a *discrete* (atomic) measure `sum_i w_i delta_{x_i}`, so
    `int f dmu = sum_i w_i f(x_i)` is exact for the counting / empirical / Dirac
    measures and a convergent quadrature for the Lebesgue / Gaussian ones (the
    weights come from `omnibias.fields`'s Gauss-Legendre / Gauss-Hermite /
    Monte-Carlo rules). The value-add over `scipy.integrate` is that the whole
    pipeline is autograd-native and cross-backend bit-identical.

## The Measure abstraction

A `Measure` generalizes `omnibias.fields`'s `QuadratureSpec` (nodes + weights
over a box) to arbitrary supports and total mass, and adds the measure-algebra
operations a measure integral needs: `pushforward` (image measure `T_# mu`),
`product` (`mu (x) nu`), `reweight` (Radon-Nikodym / importance reweighting) and
`normalize`. Constructors `lebesgue` / `gaussian` / `uniform_mc` / `empirical` /
`counting` / `dirac` / `from_quadrature` reuse the fields quadrature rules rather
than reimplementing them.

::: omnibias.measure._core.measure
    options:
      show_root_heading: false
      heading_level: 3

## Measure-integral primitives (numpy reference)

The bit-identical numpy reference the parity tests check against:

- `lebesgue_integral(f, measure)` -- the measure integral `int f dmu` as the
  weight contraction `sum_i w_i f(x_i)` (an expectation for a probability
  measure, `int_box f dx` for the Lebesgue box measure);
- `importance_expectation` -- (self-normalized) importance-sampling expectation
  `E_p[f]` from proposal samples;
- `superlevel_measure` -- the soft superlevel-set measure
  `mu({f>t}) ~= sum_i w_i sigmoid(beta (f(x_i) - t))` (the sigmoid's derivative
  tower is exactly the omnibias sigmoid tower), the shared building block of the
  next two;
- `layer_cake_integral` -- the distribution-function identity
  `int f dmu = int_0^inf mu({f>t}) dt` (signed:
  `int_0^inf [mu({f>t}) - mu({f<-t})] dt`), differentiable through a
  non-smooth / thresholded integrand;
- `simple_function_approx` -- the monotone from-below simple-function
  construction `int s dmu = sum_k level_k * mu(band_k)`.

::: omnibias.measure._core.integrate
    options:
      show_root_heading: false
      heading_level: 3

## Public API

::: omnibias.measure
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source

## Differentiable primitives (torch)

The functional twin of `omnibias.measure._core`. Each op accepts the measure's
`nodes` / `weights` as tensors (so gradients flow into learnable weights) and is
bit-identical to the numpy reference.

::: omnibias.measure.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## Trainable layers (torch)

`nn.Module` wrappers that pin a measure's nodes as a buffer and expose its
weights (and the layer-cake softness `beta = exp(log_beta) > 0`) as optional
learnable parameters, so an integral against a measure is a trainable network
component. Each `forward` accepts either a callable `f` (evaluated at the pinned
nodes -- e.g. a sub-network) or a precomputed value tensor.

::: omnibias.measure.torch.layers
    options:
      show_root_heading: false
      heading_level: 3

## Temporal point processes & survival analysis

`omnibias.measure.{torch,jax}.pointprocess` builds the likelihood of an
inhomogeneous Poisson / temporal point process and of right-censored survival
data on top of the measure integral. The log-likelihood

\[
\log L = \sum_i \log \lambda(t_i) - \underbrace{\int_{t_0}^{T} \lambda(t)\,dt}_{\text{compensator }\Lambda}
\]

hinges on the **compensator** \(\Lambda\) (for survival, the cumulative hazard
\(H(d) = \int_0^{d} h\)) -- the term usually Monte-Carlo / quadrature
approximated. omnibias evaluates it two ways, both differentiable in the
intensity parameters:

- `compensator(intensity, t0, t1)` -- the general **numerical** route: the
  measure integral `int lambda dmu` over a Gauss-Legendre (or Monte-Carlo)
  measure on `[t0, T]`, exact for polynomial intensities;
- `closed_form_compensator(activation, w, b, t0, t1, scale)` -- the
  omnibias-native **closed form** for an OMBU intensity
  `lambda(t) = scale * activation(w t + b)`: the exact antiderivative window
  `(scale/w) [S(w T + b) - S(w t0 + b)]` with `S' = activation`, reusing the
  registered antiderivative kernel (`sigmoid` -> `softplus`, `exp` -> `exp`,
  ...). This is the fundamental-theorem twin of the derivative tower.

`poisson_nll` / `survival_nll` assemble the two log-likelihoods; the
`TemporalPointProcess` wrapper (a torch `nn.Module`; a functional holder in jax)
trains an intensity network by maximum likelihood. See
[`docs/examples/measure_pointprocess.py`](https://github.com/derivon-ai/omnibias/blob/main/docs/examples/measure_pointprocess.py)
for the validated smoke (analytic oracles, a best-in-class comparison against a
coarse left-Riemann baseline, and train-through).

::: omnibias.measure.torch.pointprocess
    options:
      show_root_heading: false
      heading_level: 3

## Fredholm & Volterra integral equations

`omnibias.measure.{_core,torch,jax}.integraleq` solves the integral equation of
the **second kind**

\[
u(x) = f(x) + \lambda \int_\Omega K(x, t)\, u(t)\, d\mu(t),
\]

the mirror image of a differential equation: a derivative reads the solution
locally, an integral operator reads all of it at once. **Nystrom discretisation**
replaces the integral with the quadrature the measure already carries, so the
equation becomes the dense linear system \((I - \lambda K W) u = f\) in the nodal
values -- and since a `Measure` *is* nodes and weights, that matrix is one outer
product away from the measure integral the rest of this package is built on. This
is the same seam the [symbolic integral
columns](symbolic.md) use to *discover* such a law from data; here it is solved.

Four routes, with deliberately different promises:

| Solver | Route | Honesty label |
| --- | --- | --- |
| `nystrom_solve` | direct solve of \((I - \lambda K W) u = f\) | **numerical**: the quadrature error of the measure's own rule, so spectral for Gauss-Legendre on a smooth kernel |
| `volterra_solve` | causal \(\int_a^x\), lower triangular | **numerical**, second order |
| `neumann_series` | \(\sum_k \lambda^k (KW)^k f\) | **numerical**, and valid only inside \(\lvert\lambda\rvert \rho(KW) < 1\) |
| `degenerate_kernel_solve` | finite-rank collapse to an \(r \times r\) system | **exact in the kernel** -- only scalar moments are quadrature |

`degenerate_kernel_solve` is the one to reach for when the kernel really is
separable, \(K(x,t) = \sum_r a_r(x) b_r(t)\): it costs an \(r \times r\) solve
instead of \(n \times n\), the solution \(u = f + \lambda \sum_r c_r a_r\) is a
closed form evaluable *anywhere* rather than only at the nodes, and it is the
analytic oracle the other three are tested against.

`neumann_series` returns a `NeumannResult` rather than a bare array, for a
specific reason: a Neumann series outside its convergence radius does not fail
loudly, it returns a finite array of garbage. The dataclass carries the measured
`residual`, the estimated `spectral_radius`, and a `converged` flag *earned* by
the residual rather than assumed from the iteration count -- and
`raise_if_diverged=True` turns a divergence into an exception.

The two direct solvers screen for a **Fredholm alternative** before solving. If
\(1/\lambda\) is an eigenvalue of the discretised operator the equation has no
unique solution, and the failure mode in floating point is nastier than an
outright singular matrix: one rounding error away from the eigenvalue the solve
succeeds and hands back a vector of order \(10^{15}\) that looks like an answer.
The test is therefore `solvability_margin`, \(\sigma_\min / \max(1, \sigma_\max)\)
-- not the reciprocal condition number, which is scale-invariant and so cannot
see a \(1 \times 1\) moment system collapse. Sweeping `lam` through
`solvability_margin` locates the operator's spectrum. Pass
`check_conditioning=False` to skip the screen; under a `jax` transform it stands
down automatically, since a tracer has no condition number to read.

`fredholm_residual` gives the pointwise residual \(u - f - \lambda \int K u\,
d\mu\) of a *candidate* solution -- the quantity a PINN drives to zero, and the
way to check any solver without a reference solution. The PINN-side twins are
`Fredholm` and `Volterra` in `omnibias.pinn.{torch,jax}.equations`, which solve
for the solution as a differentiable *function* rather than as nodal values; see
[the PINN equations page](pinn.md).

Everything is differentiable in the **kernel parameters**, the **source** and
\(\lambda\), and the measure weights may be passed as tensors, so the quadrature
itself is learnable. An integral equation therefore composes with a network on
either side: a learned kernel fitted to data, or a learned source.

!!! warning "The Fredholm routes are dense"
    \(O(n^2)\) memory and \(O(n^3)\) per solve in the node count, and the
    backward pass through `linalg.solve` costs another solve. Refining the
    quadrature is cheap in accuracy terms and expensive in wall clock. That
    asymmetry is exactly why the degenerate-kernel path exists, and why a
    Gauss-Legendre measure (spectral convergence, so few nodes) is worth far more
    here than in a local problem.

See
[`docs/examples/measure_integraleq.py`](https://github.com/derivon-ai/omnibias/blob/main/docs/examples/measure_integraleq.py)
for the validated smoke: the separable analytic oracle, Gauss-Legendre versus a
trapezoid Nystrom baseline at equal node budget, second-order Volterra
convergence, both honest-failure paths, a PINN train-through calibrated against a
supervised fit of the same architecture, and cross-backend parity.

::: omnibias.measure._core.integraleq
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend (`omnibias.measure.jax.ops` and the
`register_pytree_node_class` layers in `omnibias.measure.jax.layers`, plus the
`omnibias.measure.jax.pointprocess` and `omnibias.measure.jax.integraleq` twins)
mirrors the torch surface; the layers carry their `nodes` / `weight` / `log_beta`
as differentiable pytree leaves. Cross-backend agreement (values and gradients)
is asserted to `rtol=1e-9` in float64.

Status: Alpha (`0.1.0a1`).
