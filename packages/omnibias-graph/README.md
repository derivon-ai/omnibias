# omnibias-graph

## Graph structure inside a learning loop.

**Spectral graph operators and smooth matrix relaxations for differentiable models.**

[API reference](https://omnibias.ai/api/graph/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-graph/src/omnibias/graph) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-graph/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## Why this package exists

Graph Laplacians describe connectivity and diffusion; smooth assignment matrices provide a route from discrete matching choices to trainable scores. Graph puts these reusable operations in one primitive so application packages can focus on their objective and decoding policy.

## What you can build

- Laplacians, spectral embeddings and heat-kernel operations.
- Sinkhorn normalization, soft sorting and soft top-k.
- Torch and JAX realizations plus arrangement-related operators.

Use spectral operators for geometry-aware representations and smooth relaxations when a downstream loss must influence assignment scores. Combinatorial applications, routing workflows and shape models live in consumers. A matrix relaxation is a component of those products rather than a complete optimizer.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-graph[torch]"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-graph/pyproject.toml).

## A working example

```python
import torch
from omnibias.graph.torch.ops import graph_laplacian, sinkhorn_normalize

adjacency = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
L = graph_laplacian(adjacency)
assert torch.allclose(L.sum(dim=1), torch.zeros(3))
scores = torch.eye(3, requires_grad=True)
assignment = sinkhorn_normalize(scores, n_iters=30)
(assignment * torch.arange(9).reshape(3, 3)).sum().backward()
assert scores.grad is not None
```

## Choose the right contract

Repeated eigenvalues can make eigenvector gradients ambiguous. Sinkhorn uses a finite iteration budget, so row and column constraints need tolerances. A doubly stochastic matrix is not a permutation until decoded, and a low-temperature choice still needs handling of ties and numerical range.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/graph.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-graph/tests -q
```

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-graph/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-graph/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
