# Guarantees and limits

## Derivatives

For supported smooth activations, shared polynomial recurrences evaluate
activation derivatives without recursive autodiff. Taylor composition yields
mathematical derivatives of the represented network through the requested
finite order. Floating-point evaluation still introduces rounding error and
can suffer cancellation or overflow at high order.

A tower provider reuses an activation evaluation across orders where that
activation implements the provider. A custom activation may instead supply
only individual derivative fastpaths. Unsupported orders fail explicitly.
The PyTorch and JAX jet APIs support affine layers and registered activations;
they do not automatically transform an arbitrary framework module into a jet.

Directional jets contain `order + 1` coefficient rows. Full mixed jets in
`dim` dimensions contain `binomial(dim + order, order)` rows. This output
size remains even when the derivatives are computed without nested autodiff.
The Riccati directional composition path is quadratic in order; the general
composition path is cubic. Benchmark the actual model and requested orders.
For a straight input path and an activated first layer, the default general
MLP jet uses `sigma^(n)(W x + b) (W v)^n / n!` directly at that layer. This
omits zero higher input coefficients; later nonlinear layers still require
composition. The explicit Riccati path and arbitrary curved input jets keep
their existing recurrences.

## Laplacians without the mixed-jet dimension ceiling

The specialized one-hidden-layer operator uses
`Delta^k f(x) = sum_h c_h sigma^(2k)(w_h.x + beta_h) ||w_h||^(2k)`.
It contracts the requested operator directly, avoiding a dense Hessian or a
full order-`2k` derivative tensor. Computing the preactivations still costs
work proportional to the input dimension. Tanh's individual derivative
fastpath evaluates one tanh and a polynomial of degree `n + 1`: one activation
evaluation is not constant total arithmetic as `n` grows.

Deep linear-chain MLPs also have a direct `deep_field_laplacian` path. It
propagates activation Jacobians and Laplacians without allocating a `D x D`
identity or enumerating mixed partials. For fixed depth and layer widths, its
work and memory grow linearly with input dimension `D`. Both backends include
a 5,000-dimensional regression beyond the full mixed-jet allocation budget.
There is no fixed input-dimension cap on this path; available memory and
compute still bound a practical run. Parameter gradients are retained.

Deep repeated Laplacians have four explicit modes:

| Requested operator | Method | Meaning of the result |
| --- | --- | --- |
| `k = 1` | Forward Laplacian | Exact differentiation, subject to floating-point rounding |
| `k >= 2`, `support` | Jets over coordinate supports of size at most `k` | Exact differentiation; work grows with the number of supports |
| `k >= 2`, `estimator` | Sampled directional derivatives | Unbiased in expectation; a finite run is an estimate |
| `k` in `{2, 3}`, `structured` (JAX and PyTorch) | Projected program over the first-layer factor | Exact differentiation, subject to floating-point rounding; refuses activations it cannot differentiate faithfully |

`auto` selects support enumeration while the support count fits the budget,
except that from a measured input dimension it runs the exact `structured`
program in its place wherever that applies (`k` in `{2, 3}`, a supported smooth
activation, an affine readout): `STRUCTURED_MIN_DIM` on CPU (`k = 2` from `D = 2`;
`k = 3` from `D = 4` on JAX and `D = 6` on PyTorch) and `STRUCTURED_MIN_DIM_CUDA`
on an accelerator (`k = 3` from `D = 3`), where it measured faster; both are
module constants of `omnibias.jax.laplacian` and `omnibias.torch.laplacian`.
Both backends return the same exact value either way, within rounding, and the
PyTorch result stays connected to the readout bias, which `Delta^k` (`k >= 2`)
does not depend on, so its gradient is an exact zero as under support
enumeration (`torch.autograd.grad` without `allow_unused` does not start to raise).
Past the budget, the JAX backend uses the exact `structured` program when it
applies and samples only otherwise; the PyTorch backend samples. Explicitly forcing
`support` bypasses the budget. Use
`deep_field_polylaplacian_with_report` to inspect the selected method. Its
`concentration` is `None` by default. To request a conditional Hoeffding bound,
pass `value_range=(lo, hi)` bounding every possible normalized directional
integrand for the first batch element and output, and optionally `delta`.
Observed sample extrema do not establish this population bound. The bound
also assumes independent draws for a field and point fixed independently of
those draws; an adaptively fitted field needs fresh independent report samples.
The report is constructed on the host; use `deep_field_polylaplacian` inside JIT or
gradient transforms. Sampling is not an exact finite-sample value or an
interval-arithmetic proof. Full Hessians,
full mixed jets and other architectures do not inherit the contraction path's
dimension behavior.

## Precision and training

Core coefficients are shared across backends. Framework functions, devices,
compiler transformations and precision can change rounding; cross-backend
identity is a tested property of specific paths, not a universal guarantee.
Use matching dtypes and enable JAX 64-bit mode for float64 comparisons.

Spatial jets retain parameter gradients. A differentiable residual does not
ensure a well-conditioned optimization problem, good collocation coverage,
or convergence to the correct PDE solution. Measure independent validation
errors and enforce boundary and initial conditions explicitly.

## Certified computations

`omnibias.core.verified` provides outward-rounded interval and Taylor-model
operations. A certificate applies only to its stated model, domain and
assumptions. A certificate digest establishes integrity; it does not establish
mathematical validity. Checker-earned flags require a successful checker run.
Numerical quadrature and finite-difference approximations must be described as
approximations, with their error bounds when available.
