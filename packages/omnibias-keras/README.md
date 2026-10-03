# omnibias-keras

Keras 3 activation derivatives and operator blocks for the JAX, PyTorch, and TensorFlow backends. Set `KERAS_BACKEND` explicitly before importing Keras.

Install with `pip install omnibias-keras`; tensor backends are optional where
listed in [pyproject.toml](pyproject.toml).

Run package tests from the repository root:

```bash
python -m pytest packages/omnibias-keras/tests -q
```

See the [main guide](../../README.md) for architecture and supported workflows.
License: Apache-2.0; see [LICENSE](LICENSE).
