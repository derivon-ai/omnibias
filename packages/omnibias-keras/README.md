# omnibias-keras

**Keras 3 unified backend for omnibias.**

The same code runs on **TensorFlow, JAX, or PyTorch** because every kernel
is written against `keras.ops`. The closed-form n-th derivative towers
(`sigma^(n)(z)`) use the *same* polynomial coefficients as
`omnibias-torch` and `omnibias-jax` — they are imported from the shared
pure-Python `omnibias.core.polynomials`, so the closed-form activation /
derivative math is **bit-identical across backends by construction**.
End-to-end layer numerics otherwise follow the selected Keras backend.

## Install

```bash
pip install omnibias-keras[jax]         # or [tf] / [torch]
```

Select the Keras backend *before importing keras*:

```bash
export KERAS_BACKEND=jax                 # tensorflow | jax | torch
```

or in Python:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras  # noqa: E402
```

## Quickstart

```python
import keras
from omnibias.keras import OMBU, OperatorBlock, cmbDense

# Trainable K-bias scalar operator (drop-in for an activation).
ombu = OMBU(num_channels=4, K=2, base="tanh")
y = ombu(keras.ops.zeros((8, 4)))

# Typed operator: closed-form 2nd derivative of the base activation.
lap = OperatorBlock(channels=4, op="laplacian", base="gaussian")

# Dense + inline operator, a drop-in for keras.layers.Dense.
model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    cmbDense(units=64, op="identity", base="tanh"),
    cmbDense(units=1, op="identity", base="tanh"),
])
```

## Public API

| Symbol | Role |
|---|---|
| `OperatorMultiBiasUnit` (`OMBU`) | trainable K-bias scalar operator |
| `GrowableOperatorMultiBiasUnit` (`GrowableOMBU`) | OMBU with a growable K |
| `OperatorBlock` | typed wrapper: `identity / grad / laplacian / derivative / band / integral` |
| `cmbDense`, `cmbConv1D`, `cmbConv2D` | drop-in `Dense` / `Conv1D` / `Conv2D` with an inline operator |
| `KGrowthScheduler` | plateau-triggered K-growth controller |
| `get_activation`, `list_activations`, `register_activation`, `is_registered` | activation registry |

## Activation dictionary

Same names as the other backends: `sigmoid`, `tanh`, `softplus`,
`gaussian`, `exp`, `relu`, `silu`, `gelu`, `huber`, `arctan`, `log1pu2`,
`sin`, `cos`, `sinh`, `cosh`, `tan`, `cot`, `coth`, `sech`, `log_cosh`,
`softabs`, `smooth_sign`, `mish`.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
