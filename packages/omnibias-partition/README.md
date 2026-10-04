# omnibias-partition

**Learn where each expert applies.** Regions, memberships and expert outputs form one trainable composition.

![Learn where each expert applies.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-partition/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-partition/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-partition/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-partition/docs/visuals/scene.py)

Coordinates, oblique gate parameters and regional experts enter; nonnegative region weights and blended outputs leave. Gradients can update splits and expert parameters together, making region choice part of the learned model.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://github.com/derivon-ai/omnibias/blob/main/docs/api/partition.md) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-partition/src/omnibias/partition) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-partition/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

Temperature collapse is the central mechanism: sigmoid gates sharpen as β grows. Away from split boundaries they approach hard routing; on a boundary the sigmoid remains one half, so tie conventions matter. Bias-collapse derivative kernels support the smooth experts and gates. A finite-temperature mixture is differentiable; an exact hard if/else discontinuity is not.

## Run this README

The examples use `omnibias-partition` on Python >=3.10. Their installed-wheel
profile selects runtime features, not an editable workspace. Install the prepared
prerelease from PyPI:

```bash
python -m pip install --pre "omnibias-partition==0.1.0a2"
```

For local development before publication, build and test the coordinated wheelhouse
using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md).
The package's [wheel profile](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-partition/wheel-tests.toml)
executes the examples below outside the source checkout.

Existing published consumers may need historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

Hard routing makes it difficult to learn where one expert should stop and another should begin. A soft partition assigns nonnegative weights that sum to one, so regional outputs can be blended while learning thresholds and features. Increasing inverse temperature sharpens the decision after or during training.

## What you can build

- Oblique, axis and sparse split configurations.
- NumPy reference weights with Torch, JAX and Keras realizations.
- Regional composition, hard assignments, readable rules and scoped gap certificates.

Build mixtures of regional physics models, trainable soft trees or piecewise surrogates. This is a routing primitive: stagewise boosting and complete tabular training pipelines live in consumers. The certified add-on supplies integrations for Apache PINN and geometry users who explicitly choose the advanced engines.

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

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-partition/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-partition/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
