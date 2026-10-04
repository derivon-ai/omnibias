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

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.1.0a2** · Python **>=3.10** · **3 - Alpha** · AGPL-3.0-or-later **or commercial**

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-convex/src/omnibias/convex). Modules below are relative to `omnibias.convex`; underscored modules are internal.

`arrangement`, `arrangement.jax`, `arrangement.torch`, `certify`, `inequality`, `jax`, `jax.layer`, `jax.penalty`, `jax.solver`, `problem`, `torch`, `torch.layer`, `torch.penalty`, `torch.solver`, `warm_start`.

Exports from `omnibias.convex`:

`BarrierOptions`, `Certificate`, `CertificationError`, `ConvexSolution`, `LinearInequalityBackend`, `active_set_warm_start`, `certify_lp_optimum`, `certify_qp_optimum`, `geometry_warm_start`, `lp_dual_lower_bound`, `predicted_vertex`.

</details>

<!-- END GENERATED API INVENTORY -->
