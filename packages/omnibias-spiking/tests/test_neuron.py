# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""LIF / IF neuron primitives: forward dynamics, surrogate grads, parity.

All computations use float64. JAX runs with x64 enabled.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from omnibias.core.polynomials import hermite_coeffs, sigmoid_polynomial_coeffs
from omnibias.spiking.jax import ops as jops
from omnibias.spiking.torch import ops as tops

THRESHOLD = 1.0
DECAY = 0.9
SCALE = 4.0
V_INIT = np.array([0.2, 0.8, 1.0, 1.4, -0.3], dtype=np.float64)
X_IN = np.array([0.5, 0.1, 0.0, -0.6, 1.5], dtype=np.float64)
U_GRID = np.linspace(-2.0, 2.0, 11, dtype=np.float64)


def _np(v):  # type: ignore[no-untyped-def]
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def _ref_fast_sigmoid_deriv(u: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-u))
    c0, c1, c2 = sigmoid_polynomial_coeffs(1)
    return c2 * s * s + c1 * s + c0


def _ref_gaussian_deriv(u: np.ndarray) -> np.ndarray:
    g = np.exp(-0.5 * u * u)
    he0 = hermite_coeffs(0)[0]
    return g * he0 / math.sqrt(2.0 * math.pi)


def _expected_spike_grad(v_pre: np.ndarray, *, kind: str, scale: float) -> np.ndarray:
    u = scale * (v_pre - THRESHOLD)
    ref = _ref_fast_sigmoid_deriv if kind == "fast_sigmoid" else _ref_gaussian_deriv
    return scale * ref(u)


# ---------------------------------------------------------------------------
# Forward dynamics
# ---------------------------------------------------------------------------


def test_heaviside_spike_fires_at_threshold() -> None:
    v = torch.as_tensor(V_INIT, dtype=torch.float64)
    spikes = tops.heaviside_spike(v, THRESHOLD, surrogate_scale=SCALE)
    expected = (V_INIT >= THRESHOLD).astype(np.float64)
    assert np.allclose(_np(spikes), expected)


def test_lif_leak_and_soft_reset() -> None:
    v = torch.zeros_like(torch.as_tensor(V_INIT, dtype=torch.float64))
    x = torch.as_tensor(X_IN, dtype=torch.float64)
    s, v_out = tops.lif_step(
        v, x, decay=DECAY, threshold=THRESHOLD, surrogate_scale=SCALE,
    )
    v_pre = DECAY * 0.0 + X_IN
    expected_s = (v_pre >= THRESHOLD).astype(np.float64)
    expected_v = v_pre - expected_s * THRESHOLD
    assert np.allclose(_np(s), expected_s)
    assert np.allclose(_np(v_out), expected_v)


def test_if_step_matches_lif_decay_one() -> None:
    v = torch.as_tensor(V_INIT, dtype=torch.float64)
    x = torch.as_tensor(X_IN, dtype=torch.float64)
    s_if, v_if = tops.if_step(v, x, threshold=THRESHOLD, surrogate_scale=SCALE)
    s_lif, v_lif = tops.lif_step(
        v, x, decay=1.0, threshold=THRESHOLD, surrogate_scale=SCALE,
    )
    assert np.allclose(_np(s_if), _np(s_lif))
    assert np.allclose(_np(v_if), _np(v_lif))


# ---------------------------------------------------------------------------
# Surrogate derivative shape / closed form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["fast_sigmoid", "gaussian"])
def test_surrogate_derivative_positive_peaks_at_zero(kind: str) -> None:
    u = torch.as_tensor(U_GRID, dtype=torch.float64)
    ds = tops.surrogate_derivative(u, kind)
    ds_np = _np(ds)
    assert np.all(ds_np > 0.0)
    peak_idx = int(np.argmax(ds_np))
    assert U_GRID[peak_idx] == pytest.approx(0.0, abs=1e-12)
    assert ds_np[peak_idx] > ds_np[0]
    assert ds_np[peak_idx] > ds_np[-1]


@pytest.mark.parametrize("kind", ["fast_sigmoid", "gaussian"])
def test_surrogate_matches_closed_form(kind: str) -> None:
    u = torch.as_tensor(U_GRID, dtype=torch.float64)
    ds = _np(tops.surrogate_derivative(u, kind))
    ref = _ref_fast_sigmoid_deriv(U_GRID) if kind == "fast_sigmoid" else _ref_gaussian_deriv(U_GRID)
    assert np.allclose(ds, ref, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("kind", ["fast_sigmoid", "gaussian"])
def test_surrogate_matches_closed_form_dense_grid_and_random(kind: str) -> None:
    """Dense-grid + random-sample soundness: the surrogate derivative equals its
    closed form and matches across backends on a dense grid AND random points
    (the fixed test above checks only an 11-point grid)."""
    rng = np.random.default_rng(31)
    dense = np.linspace(-6.0, 6.0, 201, dtype=np.float64)
    rand = rng.uniform(-6.0, 6.0, size=200).astype(np.float64)
    u = np.concatenate([dense, rand])
    ref = _ref_fast_sigmoid_deriv(u) if kind == "fast_sigmoid" else _ref_gaussian_deriv(u)

    ds_t = _np(tops.surrogate_derivative(torch.as_tensor(u, dtype=torch.float64), kind))
    ds_j = _np(jops.surrogate_derivative(jnp.asarray(u), kind))
    assert np.allclose(ds_t, ref, rtol=1e-12, atol=1e-14)
    assert np.allclose(ds_j, ref, rtol=1e-12, atol=1e-14)
    assert np.allclose(ds_t, ds_j, rtol=1e-9, atol=1e-11)
    # The surrogate is a nonnegative bump that decays away from the threshold.
    assert np.all(ds_t >= 0.0)


def test_surrogate_scale_sharpens_peak() -> None:
    v = torch.tensor([THRESHOLD], dtype=torch.float64, requires_grad=True)
    loss_lo = tops.heaviside_spike(v, THRESHOLD, surrogate_scale=2.0).sum()
    loss_hi = tops.heaviside_spike(v, THRESHOLD, surrogate_scale=8.0).sum()
    g_lo, = torch.autograd.grad(loss_lo, v)
    g_hi, = torch.autograd.grad(loss_hi, v)
    assert float(g_hi) > float(g_lo)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_torch_lif_grad_matches_surrogate() -> None:
    v = torch.tensor(V_INIT, dtype=torch.float64, requires_grad=True)
    x = torch.tensor(X_IN, dtype=torch.float64, requires_grad=True)
    s, _ = tops.lif_step(
        v, x, decay=DECAY, threshold=THRESHOLD, surrogate_scale=SCALE,
    )
    loss = s.sum()
    gv, gx = torch.autograd.grad(loss, (v, x))
    v_pre = DECAY * V_INIT + X_IN
    expected = _expected_spike_grad(v_pre, kind="fast_sigmoid", scale=SCALE)
    assert np.allclose(_np(gv), expected * DECAY, rtol=1e-12, atol=1e-12)
    assert np.allclose(_np(gx), expected, rtol=1e-12, atol=1e-12)


def test_jax_lif_grad_matches_surrogate() -> None:
    v = jnp.asarray(V_INIT)
    x = jnp.asarray(X_IN)

    def loss_fn(v, x):  # type: ignore[no-untyped-def]
        s, _ = jops.lif_step(
            v, x, decay=DECAY, threshold=THRESHOLD, surrogate_scale=SCALE,
        )
        return s.sum()

    gv, gx = jax.grad(loss_fn, argnums=(0, 1))(v, x)
    v_pre = DECAY * V_INIT + X_IN
    expected = _expected_spike_grad(v_pre, kind="fast_sigmoid", scale=SCALE)
    assert np.allclose(np.asarray(gv), expected * DECAY, rtol=1e-12, atol=1e-12)
    assert np.allclose(np.asarray(gx), expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Cross-backend parity
# ---------------------------------------------------------------------------


def test_cross_backend_forward_and_surrogate() -> None:
    v_t = torch.as_tensor(V_INIT, dtype=torch.float64)
    x_t = torch.as_tensor(X_IN, dtype=torch.float64)
    v_j = jnp.asarray(V_INIT)
    x_j = jnp.asarray(X_IN)

    for kind in ("fast_sigmoid", "gaussian"):
        u_t = torch.as_tensor(U_GRID, dtype=torch.float64)
        ds_t = _np(tops.surrogate_derivative(u_t, kind))
        ds_j = _np(jops.surrogate_derivative(jnp.asarray(U_GRID), kind))
        assert np.allclose(ds_t, ds_j, rtol=1e-9, atol=1e-11)

    s_t, v_t_out = tops.lif_step(
        v_t, x_t, decay=DECAY, threshold=THRESHOLD, surrogate_scale=SCALE,
    )
    s_j, v_j_out = jops.lif_step(
        v_j, x_j, decay=DECAY, threshold=THRESHOLD, surrogate_scale=SCALE,
    )
    assert np.allclose(_np(s_t), _np(s_j), rtol=1e-9, atol=1e-11)
    assert np.allclose(_np(v_t_out), _np(v_j_out), rtol=1e-9, atol=1e-11)

    v_t_g = torch.tensor(V_INIT, dtype=torch.float64, requires_grad=True)
    x_t_g = torch.tensor(X_IN, dtype=torch.float64, requires_grad=True)
    loss_t = tops.lif_step(
        v_t_g, x_t_g, decay=DECAY, threshold=THRESHOLD, surrogate_scale=SCALE,
    )[0].sum()
    gv_t, gx_t = torch.autograd.grad(loss_t, (v_t_g, x_t_g))

    def jloss(v, x):  # type: ignore[no-untyped-def]
        return jops.lif_step(
            v, x, decay=DECAY, threshold=THRESHOLD, surrogate_scale=SCALE,
        )[0].sum()

    gv_j, gx_j = jax.grad(jloss, argnums=(0, 1))(v_j, x_j)
    assert np.allclose(_np(gv_t), np.asarray(gv_j), rtol=1e-9, atol=1e-11)
    assert np.allclose(_np(gx_t), np.asarray(gx_j), rtol=1e-9, atol=1e-11)
