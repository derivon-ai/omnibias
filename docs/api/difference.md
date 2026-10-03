# omnibias-difference

Finite-difference primitives.

`omnibias.difference` supplies finite-difference stencils, Taylor-error
bounds and exact coefficient algebra.

- `StencilRequest`, `IrregularStencil`, `solve_irregular_stencil`:
  construct irregular stencils.
- `apply_irregular_stencil`: evaluate a stencil on sampled values.
- `certified_irregular_error`: bound truncation error under derivative bounds.
- `finite_difference_estimate`: a finite-step derivative estimate.

Finite steps are approximations. Separate truncation bounds, supplied
regularity assumptions and floating-point error. Use closed-form activation
jets when the represented network supports them.

Install this distribution with `pip install omnibias-difference`; select its
backend extras when needed. See [guarantees](../guarantees.md).
