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
