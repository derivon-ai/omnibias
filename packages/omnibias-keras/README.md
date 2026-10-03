# omnibias-keras

## Operator-aware layers. Your Keras backend.

**Keras 3 activation derivatives and trainable operator blocks on Torch, JAX or TensorFlow.**

[API reference](https://omnibias.ai/api/keras/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-keras/src/omnibias/keras) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-keras/tests) · [Talk to Derivon](mailto:info@derivon.ai)

![Operator-aware layers. Your Keras backend.](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/bias-collapse.svg)

<details>
<summary>Watch the mechanism change continuously</summary>

![Animated mathematical explanation](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/bias-collapse.gif)

The animation illustrates the mechanism; it is not a speed benchmark.
</details>

## Why this package exists

Keep the Keras model-building workflow while replacing supported activation derivatives with shared analytic formulas. Multi-bias units and operator blocks let an activation carry a selected derivative or other supported operator into a trainable layer. The mathematical coefficients come from core; tensor execution follows the chosen Keras backend.

## What you can build

- A shared activation registry through keras.ops.
- OperatorMultiBiasUnit / OMBU and operator-typed blocks.
- Dense and convolutional adapters plus identity-initialized and growable units.

Choose this package for an existing Keras stack that needs activation-level operators. Choose the dedicated Torch or JAX distributions for the network-level directional and mixed jet APIs. A Keras backend choice is an installation and process-start decision, not something to switch after importing the library.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-keras[torch]"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-keras/pyproject.toml).

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

## Evidence, not a universal speed claim

The [performance guide](https://github.com/derivon-ai/omnibias/blob/main/docs/performance.md) separates activation derivatives,
specialized contractions and general deep-network jets. Its artifacts record
workloads, precision, compilation and independent accuracy checks, including
baseline wins. The reported speedups do not automatically transfer to an entire
training loop, another activation, a different device or this package’s every API.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/keras.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-keras/tests -q
```

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-keras/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
