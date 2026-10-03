---
name: omnibias-fields
description: Maintain field state, dispatch, caching and differential-operator composition in omnibias-fields.
---

# Maintaining omnibias-fields

Owned implementation: [source](../../../packages/omnibias-fields/src/omnibias/fields);
public surface: [API](../../../docs/api/fields.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`_core/` owns the common field state, view objects, sigma cache and dispatch
registry. `torch/` and `jax/` implement the backend field operators. Preserve a
single state protocol: downstream PINN shims depend on class identity as well as
matching attributes. Dispatch uses the shared marker contract so the substrate
can recognize supported fields without importing consumer classes.

Follow an operator through state construction, cached quantities and its backend
implementation. A cache key must distinguish every input or operator choice that
changes its mathematical value, and cached tensors must preserve the expected
gradient lifetime. Vector calculus needs explicit component and coordinate axes;
a numerically plausible reduction over the wrong axis can pass scalar examples.

Keep integration, weak forms and norms explicit about quadrature, orientation
and measure assumptions. New operators should exercise batch shapes, vector or
complex fields where applicable, and composition with another field operation.
Use the catalog tests when adding a dispatch entry, Wirtinger tests for complex
semantics, and integration/norm tests for aggregate quantities. Solver training
policies belong in the separate PINN repository.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-fields/tests/test_catalog.py packages/omnibias-fields/tests/test_vector_calculus.py packages/omnibias-fields/tests/test_integration_norms.py -q
```
