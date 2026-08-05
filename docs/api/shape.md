# omnibias-shape

Differentiable **soft shape / occupancy fields** and **soft-coverage** (soft-OR /
log-sum-exp union) operators with a **closed-form derivative tower**, bit-identical
across the torch and jax backends.

A 1-D soft interval indicator is a difference of two sigmoids; a soft axis-aligned box
is the separable product across axes; a soft *cover* of `K` shapes with existence gates
`alpha` is the probabilistic-OR union `C = 1 - prod_k (1 - alpha_k m_k)`. Because every
shape is built purely from `sigmoid`, every center-derivative is a closed-form polynomial
in `sigmoid` via the shared Riccati tower
(`omnibias.core.polynomials.sigmoid_polynomial_coeffs`). This yields the **exact gradient
and Hessian** of a coverage energy with respect to the shape centers -- no autodiff --
which is what turns a discrete geometric-covering problem into a second-order-optimizable
energy.

!!! warning "Scope: differentiable relaxation, not an exact combinatorial solver"
    The soft cover is a *continuous relaxation* of the discrete set-cover ILP. The soft-OR
    coverage `C` lower-bounds the hard coverage, and a rounded soft solution must still be
    checked / completed to a feasible discrete cover; the certified optimality gap comes
    from an area / LP lower bound (`omnibias-convex`), not from the relaxation itself. See
    [`examples/min_square_cover`](https://github.com/derivon-ai/omnibias/tree/main/examples/min_square_cover).

## Oracles

* **Hardening**: as `beta -> inf`, `soft_box` converges to the hard box indicator and its
  integer-pixel sum to the box area.
* **Closed-form vs autodiff**: `coverage_energy_grad` / `coverage_energy_hessian` match
  `torch.autograd.functional.jacobian` / `hessian` to `< 1e-9` in float64.
* **Gauss-Newton**: the `gauss_newton=True` Hessian is the PSD `J^T J` metric (the
  residual-curvature term dropped), verified positive semidefinite.

## Ops (torch)

::: omnibias.shape.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend (`omnibias.shape.jax.ops`) is the bit-identical twin: occupancy, soft-OR /
log-sum-exp coverage, the closed-form gradient / Hessian, and the cardinality surrogates all
match the torch backend to `atol=1e-10` in float64 (cross-backend parity tests). Coefficients
come from the same pure-Python Riccati tower in `omnibias-core`, so parity holds by
construction.

Status: Alpha (`0.1.0a1`).
