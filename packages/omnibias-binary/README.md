# omnibias-binary

**Hard values. A trainable backward path.** Forward representation and backward optimization have separate contracts.

![Hard values. A trainable backward path.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-binary/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-binary/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-binary/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-binary/docs/visuals/scene.py)

Real-valued tensors enter; hard binary, ternary or quantized values leave in the forward pass. The backward pass follows an explicitly chosen smooth surrogate, allowing optimization through a discrete representation.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://omnibias.ai/api/binary/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-binary/src/omnibias/binary) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-binary/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

Temperature controls the tanh-based surrogate: higher β concentrates its gradient near the threshold. The hard forward operation remains hard; the surrogate is not its classical derivative. Bias-collapse activation polynomials supply higher derivative formulas for the smooth path. Threshold and tie conventions belong to each quantizer.

## Run this README

The examples use `omnibias-binary[torch]` on Python >=3.10. Their installed-wheel
profile selects runtime features, not an editable workspace. Install the prepared
prerelease from PyPI:

```bash
python -m pip install --pre "omnibias-binary[torch]==0.1.0a2"
```

For local development before publication, build and test the coordinated wheelhouse
using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md).
The package's [wheel profile](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-binary/wheel-tests.toml)
executes the examples below outside the source checkout.

Existing published consumers may need historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

A hard quantizer provides the output representation you want but loses an ordinary useful derivative almost everywhere. Binary makes the optimization choice explicit: keep hard forward values and use a temperature-controlled Riccati derivative in the backward path. This gives quantized models a reusable gradient mechanism.

## What you can build

- Signed and zero/one binarization, ternary and k-bit quantizers.
- Torch and JAX surrogate-gradient implementations.
- Beta schedules and shared activation-polynomial derivatives.

Use binary for quantization-aware experiments, differentiable Boolean realizations and models with hard-valued intermediate states. Compare forward accuracy, surrogate conditioning and deployment behavior separately; a useful training surrogate is a modeling decision that should be evaluated on the target task.

## A working example

```python
import torch
from omnibias.binary.torch.ops import binarize

z = torch.tensor([-0.8, 0.3, 1.2], requires_grad=True)
q = binarize(z, beta=2.0)
q.sum().backward()
assert torch.all((q == -1) | (q == 1))
assert z.grad is not None and torch.isfinite(z.grad).all()
```

## Choose the right contract

The backward formula is not the classical derivative of a discontinuous quantizer. A sharper temperature can produce narrow or saturated gradient regions. Keep the surrogate definition in experiment provenance, and test higher parameter derivatives when an optimizer relies on them.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/binary.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-binary/tests -q
```

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-binary/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
