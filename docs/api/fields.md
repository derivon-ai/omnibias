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

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.2.0rc1** · Python **>=3.10** · **4 - Beta** · Apache-2.0

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-fields/src/omnibias/fields). Modules below are relative to `omnibias.fields`; underscored modules are internal.

`jax`, `jax.ops`, `jax.ops.basic`, `jax.ops.complex`, `jax.ops.conservation`, `jax.ops.high_order`, `jax.ops.integral`, `jax.ops.nonlinear`, `jax.ops.norms`, `jax.ops.registry`, `jax.ops.tensor`, `jax.ops.vector`, `locus`, `locus.jax`, `locus.torch`, `scale`, `singularity`, `torch`, `torch.ops`, `torch.ops.basic`, `torch.ops.complex`, `torch.ops.conservation`, `torch.ops.high_order`, `torch.ops.integral`, `torch.ops.nonlinear`, `torch.ops.norms`, `torch.ops.registry`, `torch.ops.tensor`, `torch.ops.vector`, `weak`, `weak.jax`, `weak.torch`.

Exports from `omnibias.fields`:

`AffineSet`, `ComponentSpec`, `ComponentView`, `CoordinateSpec`, `DISPATCH_ATTR`, `DOMAINS`, `EqualitySystem`, `FieldBase`, `FieldState`, `NewtonResult`, `OperatorInfo`, `READOUT_INDEPENDENT_ATTR`, `SigmaCache`, `TestFunctionSpace`, `UnitTerm`, `VectorView`, `WeakForm`, `affine_locus`, `boundary_bound`, `certify_locus_point`, `did_you_mean`, `exact_moment`, `get_operator`, `list_operators`, `operator_names`, `ops_registry`.

</details>

<!-- END GENERATED API INVENTORY -->
