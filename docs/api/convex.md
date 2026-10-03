# omnibias-convex

Convex optimization and certificate backend.

- `omnibias.convex.torch` and `.jax`: differentiable LP/QP solvers and layers.
- `BarrierOptions`, `ConvexSolution`: solver configuration and result objects.
- `certify_lp_optimum`, `certify_qp_optimum`: verified optimality enclosures.
- `lp_dual_lower_bound`: a dual objective bound.

Use solver status and certificate fields explicitly. A floating-point iterate
is not automatically feasible or certified. Gradients require the assumptions
of the selected implicit differentiation path, especially around changing
active constraints.

Install this distribution with `pip install omnibias-convex`; select its
backend extras when needed. See [guarantees](../guarantees.md).
