# omnibias-keras

Keras 3 activation and operator layers.

- `omnibias.keras.get_activation`: activation registry access.
- `omnibias.keras.OperatorBlock`: select identity, derivative, gradient,
  Laplacian, band or antiderivative-window behavior.
- `omnibias.keras.cmbDense`, `cmbConv1D`, `cmbConv2D`: trainable layer adapters.

Set `KERAS_BACKEND` to `torch`, `jax` or `tensorflow` before importing Keras.
Activation kernels share core coefficients and use `keras.ops`; tensor dtype
and device behavior follow the chosen backend. The network jet APIs documented
here are implemented in the dedicated PyTorch and JAX distributions.

Install this distribution with `pip install omnibias-keras`; select its
backend extras when needed. See [guarantees](../guarantees.md).
