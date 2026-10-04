---
name: omnibias-graph
description: Maintain spectral graph operators, matrix relaxations and their numerical boundary cases.
---

# Maintaining omnibias-graph

Owned implementation: [source](../../../packages/omnibias-graph/src/omnibias/graph);
public surface: [API](../../../docs/api/graph.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

The backend directories implement spectral operations and continuous relaxations;
`arrangement/` owns its geometric integration. Inspect whether an operator expects
an adjacency matrix, weighted edge list, normalized Laplacian or unnormalized
Laplacian before changing its inputs. Isolated vertices, disconnected components
and repeated eigenvalues need explicit behavior.

Eigenvectors are basis-dependent within a repeated eigenspace. Compare invariant
subspaces or resulting operators where appropriate instead of treating arbitrary
sign or basis changes as numerical failures. Heat-kernel and diffusion operations
must preserve the documented normalization and time-domain assumptions.

Sinkhorn, sorting and top-k relaxations require their own tests for marginals,
temperature limits and gradients. A continuous doubly stochastic result is not
an integral matching; decoding and feasibility claims belong to the appropriate
consumer. Run spectral tests for spectral changes, relaxation tests for matrix
transforms, and `test_audit_limits.py` for problematic regimes. Keep backend parity
focused on mathematically comparable outputs, and retain numerical refusal or
regularization behavior around degenerate inputs.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-graph/tests/test_spectral_torch.py packages/omnibias-graph/tests/test_relaxation_jax.py packages/omnibias-graph/tests/test_audit_limits.py -q
```
