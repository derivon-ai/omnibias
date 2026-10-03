# omnibias-fields

Field calculus for PINN integration.

- `omnibias.fields.FieldState`: an evaluated field, coordinates and shared cache.
- `ComponentSpec`, `CoordinateSpec`: named components and coordinate axes.
- `SigmaCache`: reuse activation derivatives within an evaluation.
- `omnibias.fields.torch.ops` and `omnibias.fields.jax.ops`: value, derivative,
  gradient, divergence, curl, Laplacian, Hessian and integration operators.

Operations consume a `FieldState`, not an arbitrary tensor and coordinates.
A consumer field provides parameters and derivative methods through the
`DISPATCH_ATTR` contract. Rebuild a state after changing coordinates or
parameters; cached evaluations must remain consistent.

This package supplies shared state and operators. Field model constructors
and the higher-level solver live in the external `omnibias-pinn` consumer.
Use [network jets](../derivatives.md) directly for a small self-contained PINN.
Quadrature results are numerical approximations unless exactness is established
for the particular integrand and rule.

Install this distribution with `pip install omnibias-fields`; select its
backend extras when needed. See [guarantees](../guarantees.md).
