# omnibias-keras

**Make operators part of the layer.** A Keras model can consume analytic derivative outputs.

![Make operators part of the layer.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-keras/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-keras/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-keras/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-keras/docs/visuals/scene.py)

A Keras tensor enters an operator-aware layer; a value or analytic activation derivative leaves with the same channel organization. Model builders can integrate the layer using their selected Keras backend and serialization workflow.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://github.com/derivon-ai/omnibias/blob/main/docs/api/keras.md) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-keras/src/omnibias/keras) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-keras/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

OMBU encodes bias collapse: a normalized pack of nearby activations converges to an activation derivative, which the analytic path evaluates directly. Its finite-bias and derivative contracts should be chosen explicitly. Temperature collapse is not a property of every Keras layer; smooth decision models need their own gates and schedule.

## Run this README

The examples use `omnibias-keras[torch]` on Python >=3.10. Their installed-wheel
profile selects runtime features, not an editable workspace. Install the prepared
prerelease from PyPI:

```bash
python -m pip install --pre "omnibias-keras[torch]==0.0.2a1"
```

For local development before publication, build and test the coordinated wheelhouse
using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md).
The package's [wheel profile](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-keras/wheel-tests.toml)
executes the examples below outside the source checkout.

Existing published consumers may need historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

Keep the Keras model-building workflow while replacing supported activation derivatives with shared analytic formulas. Multi-bias units and operator blocks let an activation carry a selected derivative or other supported operator into a trainable layer. The mathematical coefficients come from core; tensor execution follows the chosen Keras backend.

## What you can build

- A shared activation registry through keras.ops.
- OperatorMultiBiasUnit / OMBU and operator-typed blocks.
- Dense and convolutional adapters plus identity-initialized and growable units.

Choose this package for an existing Keras stack that needs activation-level operators. Choose the dedicated Torch or JAX distributions for the network-level directional and mixed jet APIs. A Keras backend choice is an installation and process-start decision, not something to switch after importing the library.

## A working example

```python
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
from keras import ops
from omnibias.keras import OMBU

unit = OMBU(num_channels=2, K=3, base="tanh")
z = ops.convert_to_tensor([[0.2, -0.3]])
value = unit(z)
second = unit.analytic_derivative(z, order=2)
assert tuple(value.shape) == tuple(second.shape) == (1, 2)
```

## Choose the right contract

Set KERAS_BACKEND before the first Keras import. Dtypes and gradients follow that backend, and serialization should be checked in the environment used for deployment. Activation-level derivative support does not imply that every arbitrary Keras model has an automatic whole-network jet conversion.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/keras.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-keras/tests -q
```

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-keras/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
