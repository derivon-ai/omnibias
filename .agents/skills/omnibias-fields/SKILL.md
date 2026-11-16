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

Caches and routing (audit of 2026-10-26). A planned operator is reused (`OPERATOR_CACHE_SIZE` entries) only for leaf parameter tensors, keyed by structure, shapes, dtypes,
devices and the identity of those tensors through weak references: an optimizer step updates the same tensors in place and the operator
reads them, a replaced parameter, a non-leaf tensor or a diffusion that trains or lives on a device builds a new one. The sigma cache is
invalidated by `Tensor._version`; a zero-copy numpy view of a JAX array is copied before it is kept.
A field that offers `planned_chain` is routed to the planner's projected programs for `laplacian`, `biharmonic`, `polylaplacian`,
`divergence` and the vector variants (one planner call serves all components) in both back ends (4 to 10 times faster on PyTorch, with
the same gradient leaves as before: `omnibias_experiments/perf_audit_20261026/none_mask.py` compares them); the generic Hessian block
computes one triangle and mirrors it, and the device or dtype conversion of a quadrature rule is remembered for as long as the rule
lives (never for a traced value).
Anything new that is cached needs tests for what must miss (a replaced parameter, a changed
structure, a traced input) and for what must see the new value (an in-place update).

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
