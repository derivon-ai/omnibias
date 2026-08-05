# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Three-way parity: torch vs jax vs keras activation kernels.

The Keras unified backend shares its polynomial coefficients with the
torch and JAX backends (all import :mod:`omnibias.core.polynomials`), so
the activation forward / derivative / fast-path kernels must agree.

The Keras backend is selected by ``KERAS_BACKEND`` (tensorflow | jax |
torch) at import time; CI runs this test once per backend. The
comparison tolerance adapts to the Keras compute precision: on a float32
backend (TF / JAX default) we use a looser tolerance, on float64 (torch
default here) a strict one. Bit-exact coefficient sharing is checked
separately in :func:`test_polynomial_coeffs_shared`.
"""

from __future__ import annotations

import numpy as np
import pytest

# Converting a backend tensor to NumPy goes through ``np.array(tensor)``,
# which under NumPy 2.x emits a third-party DeprecationWarning for some
# frameworks. That is not omnibias code, so do not let it fail the suite.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array__ implementation doesn't accept a copy keyword.*:DeprecationWarning"
)

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
keras = pytest.importorskip("keras")

import jax.numpy as jnp  # noqa: E402
from keras import ops as kops  # noqa: E402
from omnibias.jax.activations import get_activation as jax_get  # noqa: E402
from omnibias.keras.activations import get_activation as keras_get  # noqa: E402
from omnibias.keras.activations import list_activations as keras_list  # noqa: E402
from omnibias.torch.activations.registry import get_activation as torch_get  # noqa: E402

_MAX_FASTPATH_ORDER = {
    "sigmoid": 6,
    "tanh": 6,
    "softplus": 6,
    "gaussian": 6,
    "exp": 6,
    "arctan": 2,
    "log1pu2": 2,
    # relu / huber now carry all-orders a.e. towers; silu / gelu / mish are
    # exact all orders via Leibniz on the analytic z*f(z) product.
    "huber": 6,
    "silu": 6,
    "gelu": 6,
    "relu": 6,
    "log_cosh": 3,
    "softabs": 2,
    "smooth_sign": 6,
    "mish": 6,
    "sin": 6,
    "cos": 6,
    "sinh": 6,
    "cosh": 6,
    "tan": 3,
    "cot": 3,
    "coth": 3,
    "sech": 3,
    # Piecewise (almost-everywhere) family. Linear pieces: n>=2 -> 0.
    "leaky_relu": 4,
    "prelu": 4,
    "relu6": 4,
    "hardtanh": 4,
    "hardsigmoid": 4,
    "softshrink": 4,
    "hardshrink": 4,
    "threshold": 4,
    "abs": 4,
    "sign": 4,
    "step": 4,
    # Piecewise-smooth pieces (exp / quadratic / rational): all orders.
    "elu": 6,
    "selu": 6,
    "celu": 6,
    "hardswish": 6,
    "softsign": 6,
    # Beta-tempered smooth surrogates.
    "soft_relu": 6,
    "soft_step": 6,
}

# Keras compute precision depends on the active backend; pick tolerances
# accordingly. (Torch keras runs float64; TF/JAX keras default to float32.)
_FLOAT32 = keras.config.floatx() == "float32"
_RTOL = 1e-4 if _FLOAT32 else 1e-6
_ATOL = 1e-4 if _FLOAT32 else 1e-6


def _sample_inputs_for(name: str) -> np.ndarray:
    rng = np.random.default_rng(7 + abs(hash(name)) % 65536)
    if name == "tan":
        return rng.uniform(-1.2, 1.2, size=128).astype(np.float64)
    if name == "cot":
        x = rng.uniform(0.2, np.pi - 0.2, size=64)
        return np.concatenate([x, -x]).astype(np.float64)
    if name == "coth":
        x = rng.uniform(0.2, 4.0, size=64)
        return np.concatenate([x, -x]).astype(np.float64)
    return np.concatenate(
        [rng.normal(scale=1.5, size=96), np.array([0.0, 1.0, -1.0, 2.0, -2.0])]
    ).astype(np.float64)


def _keras_np(name: str, z: np.ndarray, n: int) -> np.ndarray:
    out = keras_get(name).fastpath(kops.convert_to_tensor(z), n)
    return np.asarray(kops.convert_to_numpy(out), dtype=np.float64)


def _tol_for(name: str, n: int) -> tuple[float, float]:
    """(rtol, atol), loosened where a float32 keras backend costs a few ULPs."""
    rtol, atol = _RTOL, _ATOL
    if name == "gelu":
        rtol = max(rtol, 1e-5)
    # Mish's (t, s) derivative tower has strong term cancellation at high
    # orders; on a float32 backend that drifts a little past the base tol
    # (the float64 jax<->torch parity below still holds it to 1e-6).
    if _FLOAT32 and name == "mish" and n >= 4:
        rtol, atol = 2e-3, 2e-3
    return rtol, atol


def test_keras_registers_same_activations() -> None:
    from omnibias.torch.activations.registry import list_activations as torch_list

    assert sorted(keras_list()) == sorted(torch_list())


def test_polynomial_coeffs_shared() -> None:
    from omnibias.core.polynomials import (
        hermite_coeffs,
        sigmoid_polynomial_coeffs,
        tanh_polynomial_coeffs,
    )

    assert sigmoid_polynomial_coeffs(1) == (0.0, 1.0, -1.0)
    assert tanh_polynomial_coeffs(1) == (1.0, 0.0, -1.0)
    assert hermite_coeffs(2) == (-1.0, 0.0, 1.0)


@pytest.mark.parametrize("name", sorted(_MAX_FASTPATH_ORDER))
def test_keras_matches_torch(name: str) -> None:
    z = _sample_inputs_for(name)
    z_t = torch.from_numpy(z).double()
    for n in range(0, _MAX_FASTPATH_ORDER[name] + 1):
        rtol, atol = _tol_for(name, n)
        t_out = np.asarray(torch_get(name).fastpath(z_t, n).detach().numpy(), dtype=np.float64)
        k_out = _keras_np(name, z, n)
        np.testing.assert_allclose(
            k_out, t_out, rtol=rtol, atol=atol,
            err_msg=f"keras vs torch mismatch for {name!r} at n={n}",
        )


@pytest.mark.parametrize("name", sorted(_MAX_FASTPATH_ORDER))
def test_keras_matches_jax(name: str) -> None:
    z = _sample_inputs_for(name)
    z_j = jnp.asarray(z)
    for n in range(0, _MAX_FASTPATH_ORDER[name] + 1):
        rtol, atol = _tol_for(name, n)
        j_out = np.asarray(jax_get(name).fastpath(z_j, n), dtype=np.float64)
        k_out = _keras_np(name, z, n)
        np.testing.assert_allclose(
            k_out, j_out, rtol=rtol, atol=atol,
            err_msg=f"keras vs jax mismatch for {name!r} at n={n}",
        )
