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
