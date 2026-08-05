# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smooth-surrogate towers / jets + curvature-aware backward (torch <-> jax)."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from omnibias.binary.jax import ops as jq  # noqa: E402
from omnibias.binary.torch import ops as tq  # noqa: E402

BETA = 3.0
Z = np.array([-1.3, -0.4, 0.0, 0.25, 0.9, 2.1], dtype=np.float64)


def _np(v):  # type: ignore[no-untyped-def]
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


# ----- surrogate_tower ------------------------------------------------------


def test_surrogate_tower_row0_is_tanh_and_row1_is_slope() -> None:
    tower = _np(tq.surrogate_tower(torch.as_tensor(Z), BETA, order=3))
    t = np.tanh(BETA * Z)
    assert np.allclose(tower[0], t, rtol=1e-13, atol=1e-14)
    assert np.allclose(tower[1], BETA * (1.0 - t * t), rtol=1e-13, atol=1e-14)


def test_surrogate_tower_matches_finite_difference() -> None:
    z = torch.as_tensor(Z)
    tower = _np(tq.surrogate_tower(z, BETA, order=1))
    h = 1e-6
    fd = (np.tanh(BETA * (Z + h)) - np.tanh(BETA * (Z - h))) / (2 * h)
    assert np.allclose(tower[1], fd, rtol=1e-6, atol=1e-7)


def test_surrogate_tower_parity() -> None:
    tt = _np(tq.surrogate_tower(torch.as_tensor(Z), BETA, order=4))
    jt = _np(jq.surrogate_tower(jnp.asarray(Z), BETA, order=4))
    assert np.allclose(tt, jt, rtol=1e-12, atol=1e-13)


# ----- surrogate_jet (jet-STE via compose_jet) ------------------------------


def test_surrogate_jet_equals_tower_over_factorial() -> None:
    order = 4
    jet = _np(tq.surrogate_jet(torch.as_tensor(Z), BETA, order=order))
    tower = _np(tq.surrogate_tower(torch.as_tensor(Z), BETA, order=order))
    facts = np.array([math.factorial(k) for k in range(order + 1)], dtype=np.float64)
    assert np.allclose(jet, tower / facts[:, None], rtol=1e-12, atol=1e-13)


def test_surrogate_jet_carries_curvature_and_matches_finite_diff() -> None:
    # jet[1] = s'(z); jet[2] = s''(z)/2.  Compare against finite differences.
    jet = _np(jq.surrogate_jet(jnp.asarray(Z), BETA, order=2))
    h = 1e-4
    s = lambda x: np.tanh(BETA * x)  # noqa: E731
    sp = (s(Z + h) - s(Z - h)) / (2 * h)
    spp = (s(Z + h) - 2 * s(Z) + s(Z - h)) / (h * h)
    assert np.allclose(jet[1], sp, rtol=1e-5, atol=1e-6)
    assert np.allclose(jet[2], 0.5 * spp, rtol=1e-4, atol=1e-5)


def test_surrogate_jet_parity() -> None:
    tj = _np(tq.surrogate_jet(torch.as_tensor(Z), BETA, order=4))
    jj = _np(jq.surrogate_jet(jnp.asarray(Z), BETA, order=4))
    assert np.allclose(tj, jj, rtol=1e-12, atol=1e-13)


# ----- curvature-corrected slope --------------------------------------------


def test_curvature_corrected_slope_approximates_windowed_secant() -> None:
    h = 0.1
    corrected = _np(tq.curvature_corrected_slope(torch.as_tensor(Z), BETA, window=h))
    s = lambda x: np.tanh(BETA * x)  # noqa: E731
    secant = (s(Z + h) - s(Z - h)) / (2 * h)
    # Both equal the windowed-average slope to O(h^4).
    assert np.allclose(corrected, secant, rtol=1e-2, atol=1e-3)


def test_curvature_corrected_slope_reduces_to_point_slope_for_tiny_window() -> None:
    z = torch.as_tensor(Z)
    corrected = _np(tq.curvature_corrected_slope(z, BETA, window=1e-9))
    point = _np(tq.surrogate_tower(z, BETA, order=1))[1]
    assert np.allclose(corrected, point, rtol=1e-10, atol=1e-12)


def test_curvature_corrected_slope_parity() -> None:
    tc = _np(tq.curvature_corrected_slope(torch.as_tensor(Z), BETA))
    jc = _np(jq.curvature_corrected_slope(jnp.asarray(Z), BETA))
    assert np.allclose(tc, jc, rtol=1e-12, atol=1e-13)


# ----- binarize_curvature quantizer -----------------------------------------


def test_binarize_curvature_forward_is_hard_sign() -> None:
    out = _np(tq.binarize_curvature(torch.as_tensor(Z), beta=BETA))
    assert np.allclose(out, np.where(Z >= 0, 1.0, -1.0))


def test_binarize_curvature_backward_uses_corrected_slope() -> None:
    z = torch.tensor(Z, dtype=torch.float64, requires_grad=True)
    tq.binarize_curvature(z, beta=BETA).sum().backward()
    expected = _np(tq.curvature_corrected_slope(torch.as_tensor(Z), BETA))
    assert np.allclose(_np(z.grad), expected, rtol=1e-12, atol=1e-13)


def test_binarize_curvature_parity() -> None:
    zt = torch.tensor(Z, dtype=torch.float64, requires_grad=True)
    tq.binarize_curvature(zt, beta=BETA).sum().backward()
    tgrad = _np(zt.grad)
    jgrad = _np(jax.grad(lambda zz: jnp.sum(jq.binarize_curvature(zz, BETA)))(jnp.asarray(Z)))
    assert np.allclose(tgrad, jgrad, rtol=1e-11, atol=1e-12)


# ----- beta -> inf ties to the Phase 1 saturation limit ---------------------


def test_high_beta_forward_approaches_tanh_saturation_limit() -> None:
    from omnibias.jax.activations import get_activation

    limit = get_activation("tanh").limit_pos_inf
    big = _np(jq.surrogate_tower(jnp.asarray([0.5]), 1000.0, order=0))[0, 0]
    assert big == pytest.approx(limit, abs=1e-9)
