# omnibias-partition

## Make regions—and their decisions—trainable.

**Smooth partitions of unity that route inputs into regional models with gradients through the gates.**

[API reference](https://omnibias.ai/api/partition/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-partition/src/omnibias/partition) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-partition/tests) · [Talk to Derivon](mailto:info@derivon.ai)

![Make regions—and their decisions—trainable.](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/temperature-collapse.svg)

<details>
<summary>Watch the mechanism change continuously</summary>

![Animated mathematical explanation](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/temperature-collapse.gif)

The animation illustrates the mechanism; it is not a speed benchmark.
</details>

## Why this package exists

Hard routing makes it difficult to learn where one expert should stop and another should begin. A soft partition assigns nonnegative weights that sum to one, so regional outputs can be blended while learning thresholds and features. Increasing inverse temperature sharpens the decision after or during training.

## What you can build

- Oblique, axis and sparse split configurations.
- NumPy reference weights with Torch, JAX and Keras realizations.
- Regional composition, hard assignments, readable rules and scoped gap certificates.

Build mixtures of regional physics models, trainable soft trees or piecewise surrogates. This is a routing primitive: stagewise boosting and complete tabular training pipelines live in consumers. The certified add-on supplies integrations for Apache PINN and geometry users who explicitly choose the advanced engines.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-partition"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-partition/pyproject.toml).

## A working example

```python
import numpy as np
from omnibias.partition import PartitionConfig, init_params, partition_weights

params = init_params(PartitionConfig(n_features=1, depth=1), rng=0)
x = np.linspace(-2, 2, 9)[:, None]
weights = partition_weights(params, x, beta=3.0)
experts = np.concatenate([x**2, 1 + x], axis=1)
prediction = (weights * experts).sum(axis=1)
assert np.allclose(weights.sum(axis=1), 1.0)
assert prediction.shape == (9,)
```

## Choose the right contract

Finite-temperature routing is differentiable; exact hard if/else is not made smooth by renaming it. At a split boundary a sigmoid gate stays at one half, and hard assignments need a tie policy. A depth-d binary partition has 2**d regions, so deeper routing has a real representation cost.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/partition.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-partition/tests -q
```

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-partition/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-partition/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
