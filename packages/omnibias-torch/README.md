# omnibias-torch

**Derivatives that keep training.** Spatial jets forward. Parameter gradients backward.

![Derivatives that keep training.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-torch/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-torch/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-torch/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-torch/docs/visuals/scene.py)

Torch tensors and supported network layers enter; activation derivatives, directional jets and specialized contractions leave as tensors still connected to model parameters. Use this backend to train a high-order residual without constructing a nested spatial backward graph.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://github.com/derivon-ai/omnibias/blob/main/docs/api/torch.md) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-torch/src/omnibias/torch) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-torch/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

Bias collapse becomes an analytic tensor kernel here: evaluate the shared activation polynomial, then propagate normalized Taylor coefficients through layers. Parameter autograd remains available through that calculation. Temperature collapse is realized by separate soft-gate primitives; this backend provides their trainable arithmetic rather than a tree-learning algorithm.

## Run this README

The examples use `omnibias-torch` on Python >=3.10. Their installed-wheel
profile selects runtime features, not an editable workspace. Install the prepared
prerelease from PyPI:

```bash
python -m pip install --pre "omnibias-torch==0.5.0rc1"
```

For local development before publication, build and test the coordinated wheelhouse
using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md).
The package's [wheel profile](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-torch/wheel-tests.toml)
executes the examples below outside the source checkout.

Existing published consumers may need historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

A fourth-order residual can require more derivative work than evaluating the network itself. Instead of recursively asking spatial autograd for another gradient, carry Taylor coefficients through supported layers. The result remains a tensor expression, so ordinary parameter autograd can optimize the residual.

## What you can build

- Direct activation derivatives and composable directional Taylor jets.
- Mixed partials for coupled PDE residuals and direct Laplacian contractions when the full Hessian is unnecessary.
- Trainable operator layers, reusable architectures and residual-optimizer interfaces.

Use directional jets for a few directions, mixed jets when you need the complete selected order, and specialized Laplacians for trace-like operators. These are different algorithms with different output sizes. Choose the operator your residual consumes before choosing a general Hessian implementation.

## A working example

```python
import torch
from omnibias.torch.jet import mlp_jet, jet_to_tower

w = torch.tensor([[0.5], [-0.3]], dtype=torch.float64, requires_grad=True)
layers = [(w, torch.tensor([0.1, 0.2]), "tanh"),
          (torch.tensor([[0.6, -0.4]], dtype=torch.float64), None, None)]
x = torch.tensor([0.3], dtype=torch.float64)
tower = jet_to_tower(mlp_jet(x, torch.ones_like(x), layers, order=4))
residual = tower[4] + tower[0]  # an illustrative fourth-order operator
residual.square().mean().backward()
assert w.grad is not None and torch.isfinite(w.grad).all()
```

## Choose the right contract

Jet row k stores the derivative divided by k!. Convert with jet_to_tower at derivative-facing boundaries. Mixed coefficients use multi-index factorials. Shared algebra gives comparable mathematics, not universal bit identity across devices. Dense mixed jets still grow combinatorially; no fixed dimension ceiling means no hard-coded cap, not constant memory.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/torch.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-torch/tests -q
```

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-torch/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
