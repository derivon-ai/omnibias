# omnibias-torch

## Differentiate deeper. Keep training.

**PyTorch activation towers, directional and mixed jets, and direct Laplacian paths with gradients into model parameters.**

[API reference](https://omnibias.ai/api/torch/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-torch/src/omnibias/torch) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-torch/tests) · [Talk to Derivon](mailto:info@derivon.ai)

![Differentiate deeper. Keep training.](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/bias-collapse.svg)

<details>
<summary>Watch the mechanism change continuously</summary>

![Animated mathematical explanation](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/bias-collapse.gif)

The animation illustrates the mechanism; it is not a speed benchmark.
</details>

## Why this package exists

A fourth-order residual can require more derivative work than evaluating the network itself. Instead of recursively asking spatial autograd for another gradient, carry Taylor coefficients through supported layers. The result remains a tensor expression, so ordinary parameter autograd can optimize the residual.

## What you can build

- Direct activation derivatives and composable directional Taylor jets.
- Mixed partials for coupled PDE residuals and direct Laplacian contractions when the full Hessian is unnecessary.
- Trainable operator layers, reusable architectures and residual-optimizer interfaces.

Use directional jets for a few directions, mixed jets when you need the complete selected order, and specialized Laplacians for trace-like operators. These are different algorithms with different output sizes. Choose the operator your residual consumes before choosing a general Hessian implementation.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-torch"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-torch/pyproject.toml).

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

## Evidence, not a universal speed claim

The [performance guide](https://github.com/derivon-ai/omnibias/blob/main/docs/performance.md) separates activation derivatives,
specialized contractions and general deep-network jets. Its artifacts record
workloads, precision, compilation and independent accuracy checks, including
baseline wins. The reported speedups do not automatically transfer to an entire
training loop, another activation, a different device or this package’s every API.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/torch.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-torch/tests -q
```

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-torch/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
