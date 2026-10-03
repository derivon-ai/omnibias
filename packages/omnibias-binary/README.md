# omnibias-binary

## Discrete forward values. Trainable surrogate gradients.

**Binary, ternary and k-bit quantizers with an explicit smooth backward model.**

[API reference](https://omnibias.ai/api/binary/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-binary/src/omnibias/binary) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-binary/tests) · [Talk to Derivon](mailto:info@derivon.ai)

![Discrete forward values. Trainable surrogate gradients.](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/temperature-collapse.svg)

<details>
<summary>Watch the mechanism change continuously</summary>

![Animated mathematical explanation](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/temperature-collapse.gif)

The animation illustrates the mechanism; it is not a speed benchmark.
</details>

## Why this package exists

A hard quantizer provides the output representation you want but loses an ordinary useful derivative almost everywhere. Binary makes the optimization choice explicit: keep hard forward values and use a temperature-controlled Riccati derivative in the backward path. This gives quantized models a reusable gradient mechanism.

## What you can build

- Signed and zero/one binarization, ternary and k-bit quantizers.
- Torch and JAX surrogate-gradient implementations.
- Beta schedules and shared activation-polynomial derivatives.

Use binary for quantization-aware experiments, differentiable Boolean realizations and models with hard-valued intermediate states. Compare forward accuracy, surrogate conditioning and deployment behavior separately; a useful training surrogate is a modeling decision that should be evaluated on the target task.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-binary[torch]"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-binary/pyproject.toml).

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

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-binary/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
