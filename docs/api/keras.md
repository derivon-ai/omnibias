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

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.0.2a1** · Python **>=3.10** · **3 - Alpha** · Apache-2.0

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-keras/src/omnibias/keras). Modules below are relative to `omnibias.keras`; underscored modules are internal.

`activations`, `activations.classical`, `activations.nqs`, `activations.piecewise`, `activations.proximal`, `activations.registry`, `activations.smooth`, `activations.tempered`, `activations.trigonometric`, `blocks`, `blocks.conv`, `blocks.linear`, `blocks.operator`, `fastpath`, `fastpath.dispatch`, `fastpath.eulerian`, `fastpath.hermite`, `fastpath.legendre`, `growable`, `identity_init`, `stencil`, `tempered_blocks`, `training`, `training.k_scheduler`, `unit`.

Exports from `omnibias.keras`:

`ActivationSpec`, `AnalyticGaussianConv1D`, `AnalyticGaussianConv2D`, `GrowStrategy`, `GrowableOMBU`, `GrowableOperatorMultiBiasUnit`, `KGrowthScheduler`, `LearnablePReLU`, `OMBU`, `OpName`, `OperatorBlock`, `OperatorMultiBiasUnit`, `TemperedActivation`, `analytic_gaussian_taps`, `cmbConv1D`, `cmbConv2D`, `cmbDense`, `get_activation`, `is_registered`, `list_activations`, `register_activation`.

</details>

<!-- END GENERATED API INVENTORY -->
