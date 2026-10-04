# omnibias-graph

Differentiable graph primitives.

- `omnibias.graph.torch.ops` and `omnibias.graph.jax.ops`: graph spectral
  operators and differentiable combinatorial relaxations.
- Spectral operations include graph Laplacians, embeddings and heat kernels.
- Relaxations include Sinkhorn normalization, soft sorting and soft top-k.

These are reusable operators for consumer models. Temperature-based outputs
are continuous relaxations; a discrete interpretation needs an explicit
rounding rule. Repeated eigenvalues and ties require care when interpreting
parameter gradients.

Install this distribution with `pip install omnibias-graph`; select its
backend extras when needed. See [guarantees](../guarantees.md).

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.1.0a2** · Python **>=3.10** · **3 - Alpha** · AGPL-3.0-or-later **or commercial**

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-graph/src/omnibias/graph). Modules below are relative to `omnibias.graph`; underscored modules are internal.

`arrangement`, `arrangement.jax`, `arrangement.torch`, `jax`, `jax.ops`, `jax.ops.relaxation`, `jax.ops.spectral`, `torch`, `torch.ops`, `torch.ops.relaxation`, `torch.ops.spectral`.

</details>

<!-- END GENERATED API INVENTORY -->
