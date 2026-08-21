# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Composed-curvature algebra (theory 08-02): chain rule, Jacobi, G4."""

from __future__ import annotations

import cmath
import math

import pytest
from omnibias.core.composed_curvature import (
    ComposedCurvatureConfig,
    eigh_symmetric,
    eval_tanh_derivative,
    reject_full_parameter_jacobian,
    scalar_nest_hessian,
    select_composed_step,
    solve_dense,
)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def test_tanh_first_two_derivatives_match_closed_form() -> None:
    z = 0.2
    t = math.tanh(z)
    sech2 = 1.0 - t * t
    assert eval_tanh_derivative(z, 0) == pytest.approx(t)
    assert eval_tanh_derivative(z, 1) == pytest.approx(sech2)
    assert eval_tanh_derivative(z, 2) == pytest.approx(-2.0 * t * sech2)


def test_g1_scalar_nest_matches_high_precision_fd() -> None:
    w, v, x = 0.2, 0.1, 1.0
    h_ww, h_wv, h_vv = scalar_nest_hessian(w, v, x)

    def loss(ww: float, vv: float) -> float:
        h = math.tanh(ww * x)
        r = vv * h - 1.0
        return 0.5 * r * r

    def loss_c(ww: complex, vv: complex) -> complex:
        h = cmath.tanh(ww * x)
        r = vv * h - 1.0
        return 0.5 * r * r

    def fourth_ww(h: float) -> float:
        return (
            -loss(w + 2 * h, v)
            + 16.0 * loss(w + h, v)
            - 30.0 * loss(w, v)
            + 16.0 * loss(w - h, v)
            - loss(w - 2 * h, v)
        ) / (12.0 * h * h)

    def fourth_vv(h: float) -> float:
        return (
            -loss(w, v + 2 * h)
            + 16.0 * loss(w, v + h)
            - 30.0 * loss(w, v)
            + 16.0 * loss(w, v - h)
            - loss(w, v - 2 * h)
        ) / (12.0 * h * h)

    h_fd = 0.01
    fd_ww = (16.0 * fourth_ww(h_fd / 2.0) - fourth_ww(h_fd)) / 15.0
    fd_vv = (16.0 * fourth_vv(h_fd / 2.0) - fourth_vv(h_fd)) / 15.0
    step = 1e-8

    def dldw(ww: float, vv: float) -> float:
        return loss_c(ww + 1j * step, vv).imag / step

    def fourth_mixed(h: float) -> float:
        return (dldw(w, v + h) - dldw(w, v - h)) / (2.0 * h)

    fd_wv = (16.0 * fourth_mixed(h_fd / 2.0) - fourth_mixed(h_fd)) / 15.0
    assert _rel(h_ww, fd_ww) <= 1e-10
    assert _rel(h_vv, fd_vv) <= 1e-10
    assert _rel(h_wv, fd_wv) <= 1e-10


def test_eigh_2x2_matches_closed_form() -> None:
    h = ((2.0, 1.0), (1.0, 2.0))
    evals, vecs = eigh_symmetric(h)
    assert evals[0] == pytest.approx(1.0)
    assert evals[1] == pytest.approx(3.0)
    # Reconstruct H = V diag V^T.
    recon = [[0.0, 0.0], [0.0, 0.0]]
    for i in range(2):
        for j in range(2):
            recon[i][j] = (
                evals[0] * vecs[i][0] * vecs[j][0]
                + evals[1] * vecs[i][1] * vecs[j][1]
            )
    assert recon[0][0] == pytest.approx(2.0)
    assert recon[0][1] == pytest.approx(1.0)


def test_solve_dense_recovers_identity_system() -> None:
    x = solve_dense(((2.0, 0.0), (0.0, 3.0)), (4.0, 9.0), damping=0.0)
    assert x[0] == pytest.approx(2.0)
    assert x[1] == pytest.approx(3.0)


def test_g4_rejects_full_parameter_jacobian() -> None:
    with pytest.raises(ValueError, match="allow_full"):
        reject_full_parameter_jacobian(2, 2, allow_full=False)
    reject_full_parameter_jacobian(2, 2, allow_full=True)
    reject_full_parameter_jacobian(1, 4, allow_full=False)


def test_escape_picks_negative_mode() -> None:
    # Slice PD, joint saddle: H_vv = 1, mixed = 2, H_ww = 1 => det = -3.
    h_slice = ((1.0,),)
    h_joint = ((1.0, 2.0), (2.0, 1.0))
    step = select_composed_step(h_slice, h_joint, (0.0, 0.0))
    assert step.lambda_min_slice >= 0.0
    assert step.lambda_min_joint < 0.0
    assert step.escaped
    nrm = math.sqrt(sum(c * c for c in step.coeffs))
    assert nrm == pytest.approx(1.0)
