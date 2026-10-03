# omnibias-jax

JAX derivatives and trainable operators.

- `omnibias.jax.activations.get_activation`: activation specifications.
- `omnibias.jax.jet`: directional network jets and Taylor conversions.
- `omnibias.jax.jet_mv`: full mixed jets and derivative extraction.

The [derivative guide](../derivatives.md) demonstrates `jax.grad` over a
fourth-order residual. Keep array operations traceable and derivative orders
static when compiling. Set `JAX_ENABLE_X64=1` before array creation when
comparing to float64 reference values. Shared coefficient definitions do not
remove compiler or device rounding differences.

Install this distribution with `pip install omnibias-jax`; select its
backend extras when needed. See [guarantees](../guarantees.md).
