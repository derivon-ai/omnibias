# omnibias-graph

**From relations to soft assignments.** Spectral operators and matrix relaxations expose graph structure.

![From relations to soft assignments.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-graph/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-graph/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-graph/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-graph/docs/visuals/scene.py)

An affinity or cost matrix enters; spectral graph operators, embeddings, soft orderings or assignment matrices leave. Graph-aware models can compose these outputs with trainable features while respecting each relaxation’s constraints.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://omnibias.ai/api/graph/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-graph/src/omnibias/graph) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-graph/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

Temperature-like sharpening controls selected soft assignment and sorting operators. Ties and finite iteration residuals must remain visible when interpreting a hardened result. Spectral Laplacians are a separate graph-calculus operation, not automatically a temperature limit. Bias collapse is consumed only where a differentiated model explicitly uses the shared derivative tower.

## Run this README

The examples use `omnibias-graph[torch]` on Python >=3.10. Their installed-wheel
profile is [wheel-tests.toml](https://github.com/derivon-ai/omnibias/blob/codex/pinn-substrate-split/packages/omnibias-graph/wheel-tests.toml); it selects runtime features, not
an editable workspace. Build the coordinated wheelhouse using the
[release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md), then run
from that main checkout:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-graph[torch]"
```

After this opt-in prerelease is published, the equivalent index command is:

```bash
python -m pip install --pre "omnibias-graph[torch]==0.1.0a2"
```

Existing published consumers may need the historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

Graph Laplacians describe connectivity and diffusion; smooth assignment matrices provide a route from discrete matching choices to trainable scores. Graph puts these reusable operations in one primitive so application packages can focus on their objective and decoding policy.

## What you can build

- Laplacians, spectral embeddings and heat-kernel operations.
- Sinkhorn normalization, soft sorting and soft top-k.
- Torch and JAX realizations plus arrangement-related operators.

Use spectral operators for geometry-aware representations and smooth relaxations when a downstream loss must influence assignment scores. Combinatorial applications, routing workflows and shape models live in consumers. A matrix relaxation is a component of those products rather than a complete optimizer.

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

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-graph/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-graph/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
