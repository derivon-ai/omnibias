# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form quantization gradients: forward hard quantize, Riccati surrogate backward.

Cross-backend parity (torch autograd vs jax.grad) on float64 inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from omnibias.binary.jax import ops as jq  # noqa: E402
from omnibias.binary.torch import ops as tq  # noqa: E402

BETA = 10.0
DELTA = 0.5
LO, HI = -1.0, 1.0
Z_1D = np.array([-2.0, -0.6, -0.1, 0.0, 0.1, 0.6, 2.0], dtype=np.float64)
Z_2D = np.reshape(Z_1D, (7, 1))


def _np(v):  # type: ignore[no-untyped-def]
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def _torch_grad(fn, z_np, *args):  # type: ignore[no-untyped-def]
    z = torch.tensor(z_np, dtype=torch.float64, requires_grad=True)
    out = fn(z, *args)
    out.sum().backward()
    return _np(out), _np(z.grad)


def _jax_grad(fn, z_np, *args):  # type: ignore[no-untyped-def]
    z = jnp.asarray(z_np, dtype=jnp.float64)

    def loss(zz):  # type: ignore[no-untyped-def]
        return jnp.sum(fn(zz, *args))

    out = fn(z, *args)
    grad = jax.grad(loss)(z)
    return _np(out), _np(grad)


def _expected_binarize(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0, -1.0)


def _expected_ternarize(z: np.ndarray, delta: float = DELTA) -> np.ndarray:
    return np.where(z > delta, 1.0, np.where(z < -delta, -1.0, 0.0))


def _kbit_levels(bits: int, lo: float = LO, hi: float = HI) -> np.ndarray:
    n_levels = 2**bits
    step = (hi - lo) / (n_levels - 1)
    return lo + step * np.arange(n_levels, dtype=np.float64)


def _expected_kbit(z: np.ndarray, bits: int, lo: float = LO, hi: float = HI) -> np.ndarray:
    levels = _kbit_levels(bits, lo, hi)
    step = levels[1] - levels[0]
    idx = np.round((z - lo) / step)
    idx = np.clip(idx, 0, len(levels) - 1)
    return lo + idx * step


def _surrogate_binarize_grad(z: np.ndarray, beta: float = BETA) -> np.ndarray:
    t = np.tanh(beta * z)
    return beta * (1.0 - t * t)


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


def test_binarize_forward() -> None:
    z = torch.as_tensor(Z_1D, dtype=torch.float64)
    out = _np(tq.binarize(z, beta=BETA))
    assert np.allclose(out, _expected_binarize(Z_1D))
    assert np.all(np.abs(out) == 1.0)
    assert out[Z_1D == 0.0][0] == 1.0


def test_ternarize_forward_deadzone() -> None:
    z = torch.as_tensor(Z_1D, dtype=torch.float64)
    out = _np(tq.ternarize(z, beta=BETA, delta=DELTA))
    expected = _expected_ternarize(Z_1D)
    assert np.allclose(out, expected)
    assert out[np.abs(Z_1D) < DELTA].tolist() == [0.0, 0.0, 0.0]


def test_kbit_forward_levels_and_bounds() -> None:
    z = torch.as_tensor(Z_1D, dtype=torch.float64)
    out1 = _np(tq.kbit_quantize(z, bits=1, lo=LO, hi=HI, beta=BETA))
    assert np.all(np.abs(out1) == 1.0)

    out2 = _np(tq.kbit_quantize(z, bits=2, lo=LO, hi=HI, beta=BETA))
    levels = _kbit_levels(2, LO, HI)
    assert np.all(out2 >= LO - 1e-15)
    assert np.all(out2 <= HI + 1e-15)
    for v in out2:
        assert np.min(np.abs(levels - v)) < 1e-12


def test_shape_preservation() -> None:
    z = torch.as_tensor(Z_2D, dtype=torch.float64)
    assert tq.binarize(z).shape == Z_2D.shape
    assert tq.ternarize(z).shape == Z_2D.shape
    assert tq.kbit_quantize(z, bits=2).shape == Z_2D.shape


def test_monotonic_kbit_levels() -> None:
    grid = np.linspace(LO, HI, 31, dtype=np.float64)
    z = torch.as_tensor(grid, dtype=torch.float64)
    q = _np(tq.kbit_quantize(z, bits=2, lo=LO, hi=HI, beta=BETA))
    assert np.all(np.diff(q) >= -1e-15)


# ---------------------------------------------------------------------------
# Backward correctness
# ---------------------------------------------------------------------------


def test_binarize_backward_formula() -> None:
    z = torch.tensor(Z_1D, dtype=torch.float64, requires_grad=True)
    out = tq.binarize(z, beta=BETA)
    out.sum().backward()
    grad = _np(z.grad)
    assert grad is not None
    assert np.allclose(grad, _surrogate_binarize_grad(Z_1D, BETA), rtol=1e-12, atol=1e-14)


def test_binarize_high_beta_concentrates_at_origin() -> None:
    grad_origin = _surrogate_binarize_grad(np.array([0.0]), beta=1000.0)[0]
    grad_far = _surrogate_binarize_grad(np.array([5.0]), beta=1000.0)[0]
    assert abs(grad_far) < abs(grad_origin)
    assert abs(grad_far) < 1e-6


def test_riccati_matches_direct_tanh_prime() -> None:
    z = torch.as_tensor(Z_1D, dtype=torch.float64)
    t = torch.tanh(BETA * z)
    ric = _np(tq.riccati_tanh_derivative(t, order=1))
    direct = _np(1.0 - t * t)
    assert np.allclose(ric, direct, rtol=1e-15, atol=1e-15)


# ---------------------------------------------------------------------------
# Cross-backend parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn_name", "args"),
    [
        ("binarize", (BETA,)),
        ("ternarize", (BETA, DELTA)),
        ("kbit_quantize", (2, LO, HI, BETA)),
    ],
)
def test_cross_backend_parity(fn_name: str, args: tuple) -> None:
    tfn = getattr(tq, fn_name)
    jfn = getattr(jq, fn_name)
    tout, tgrad = _torch_grad(tfn, Z_1D, *args)
    jout, jgrad = _jax_grad(jfn, Z_1D, *args)
    assert np.allclose(tout, jout, rtol=1e-9, atol=1e-11)
    assert np.allclose(tgrad, jgrad, rtol=1e-9, atol=1e-11)
