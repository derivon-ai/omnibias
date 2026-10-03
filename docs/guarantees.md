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

Deep repeated Laplacians have three explicit modes:

| Requested operator | Method | Meaning of the result |
| --- | --- | --- |
| `k = 1` | Forward Laplacian | Exact differentiation, subject to floating-point rounding |
| `k >= 2`, `support` | Jets over coordinate supports of size at most `k` | Exact differentiation; work grows with the number of supports |
| `k >= 2`, `estimator` | Sampled directional derivatives | Unbiased in expectation; a finite run is an estimate |

`auto` selects support enumeration or sampling according to the budget;
explicitly forcing `support` bypasses that budget. Use
`deep_field_polylaplacian_with_report` to inspect the selected method. Its
current concentration report uses the observed range from the first batch
element and output. Treat that report as a diagnostic, not a validated
confidence bound for unseen draws or other outputs. Sampling is not an exact
finite-sample value or an interval-arithmetic proof. Full Hessians,
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
