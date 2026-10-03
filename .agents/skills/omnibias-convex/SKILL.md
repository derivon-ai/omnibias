---
name: omnibias-convex
description: Maintain LP/QP problem definitions, barrier solves, implicit gradients and optimality bounds.
---

# Maintaining omnibias-convex

Owned implementation: [source](../../../packages/omnibias-convex/src/omnibias/convex);
public surface: [API](../../../docs/api/convex.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`problem.py` defines the optimization problem, backend directories implement
solves and differentiable layers, and `certify.py` checks bounds. Preserve the
objective normalization and inequality sign convention at each boundary; a
factor-of-two Hessian error can change both the solution and implicit gradient.

A numerical candidate and a verified enclosure are separate outputs. Keep
infeasibility, rank deficiency, failed line searches and unconverged iterations
visible rather than returning a successful certificate. Warm starts need validation
against the new problem dimensions and active constraints before reuse.

For implicit differentiation, inspect the KKT system, regularization and active-set
assumptions. Test gradients away from and near active-set changes, and avoid
promising classical differentiability at a nonsmooth transition. Equality and
inequality penalties have different feasibility behavior. Small analytic LP/QP
problems and independent reference solves check values; layer tests check gradients;
certificate tests check enclosure direction and soundness. Arrangement and warm-start
integrations should reuse this solver boundary instead of duplicating it in a
consumer.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-convex/tests/test_solver_torch.py packages/omnibias-convex/tests/test_layer_jax.py packages/omnibias-convex/tests/test_certify.py -q
```
