# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the discrete adjoint / Riccati recursion (theory 10-02)."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.core.adjoint import (
    adjoint_recursion,
    adjoint_skill,
    adjoint_step,
    closed_loop_jacobian,
    discrete_riccati_sweep,
    honesty_payload,
    lqr_costate_from_riccati,
    n_step_adjoint_bootstrap,
    policy_gradient,
    td_lambda_mix,
    total_state_cost_gradient,
    worked_example,
)


def test_honesty_payload_all_false() -> None:
    payload = honesty_payload()
    assert not any(payload.values())


def test_closed_loop_jacobian_matches_manual() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0], [0.0]])
    dpi_dy = np.array([[-0.5, -0.2]])
    m = closed_loop_jacobian(a, b, dpi_dy)
    expected = a + b @ dpi_dy
    assert np.allclose(m, expected)


def test_adjoint_step_matches_manual() -> None:
    m = np.array([[2.0, 0.0], [0.0, 0.5]])
    c = np.array([1.0, -1.0])
    lam_next = np.array([0.5, 0.5])
    out = adjoint_step(m, c, lam_next)
    assert np.allclose(out, m.T @ lam_next + c)


def test_adjoint_recursion_length_and_terminal() -> None:
    m = np.eye(2)
    c = np.zeros(2)
    terminal = np.array([1.0, 2.0])
    seq = adjoint_recursion([m, m, m], [c, c, c], terminal)
    assert len(seq) == 4
    assert np.allclose(seq[-1], terminal)
    assert np.allclose(seq[0], terminal)  # identity closed-loop, zero cost


def test_adjoint_recursion_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        adjoint_recursion([np.eye(2)], [np.zeros(2), np.zeros(2)], np.zeros(2))


def test_worked_example_g1_g2() -> None:
    result = worked_example()
    assert result["g2_earned"] is True
    assert result["max_abs_err"] < 1e-10


def test_adjoint_skill_g2() -> None:
    result = adjoint_skill(n=30, seed=1)
    assert result["all_finite"] is True
    assert result["singular_raised"] is True
    assert result["g2_earned"] is True
    assert result["median_err"] < 1e-9


def test_discrete_riccati_sweep_shapes() -> None:
    a = np.eye(2)
    b = np.array([[0.0], [1.0]])
    q = np.eye(2)
    r = np.array([[1.0]])
    qf = 2.0 * np.eye(2)
    sweep = discrete_riccati_sweep([a] * 4, [b] * 4, [q] * 4, [r] * 4, qf)
    assert len(sweep.p_seq) == 5
    assert len(sweep.k_seq) == 4
    assert np.allclose(sweep.p_seq[-1], qf)
    for p in sweep.p_seq:
        assert np.allclose(p, p.T)  # symmetric cost-to-go


def test_discrete_riccati_sweep_singular_raises() -> None:
    with pytest.raises(ValueError, match="singular"):
        discrete_riccati_sweep(
            [np.eye(1)], [np.zeros((1, 1))], [np.zeros((1, 1))], [np.zeros((1, 1))], np.eye(1)
        )


def test_lqr_costate_from_riccati_length_mismatch() -> None:
    with pytest.raises(ValueError):
        lqr_costate_from_riccati([np.eye(2)], [np.zeros(2), np.zeros(2)])


def test_policy_gradient_matches_manual_sum() -> None:
    dl_du = [np.array([1.0]), np.array([2.0])]
    b = [np.array([[1.0], [0.0]]), np.array([[1.0], [0.0]])]
    dpi_dtheta = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    lam_seq = [np.array([0.0, 0.0]), np.array([1.0, -1.0]), np.array([0.5, 0.5])]
    grad = policy_gradient(dl_du, b, dpi_dtheta, lam_seq)
    # step 0: h_u = dl_du[0] + b[0]^T lam_seq[1] = 1 + 1 = 2; term = dpi_dtheta[0]^T @ [2] = [2,0]
    # step 1: h_u = 2 + 0.5 = 2.5; term = dpi_dtheta[1]^T @ [2.5] = [0, 2.5]
    assert np.allclose(grad, np.array([2.0, 2.5]))


def test_policy_gradient_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        policy_gradient(
            [np.array([1.0])],
            [np.array([[1.0]]), np.array([[1.0]])],
            [np.array([[1.0]])],
            [np.zeros(1), np.zeros(1)],
        )


def test_td_lambda_mix_zero_recovers_first_bootstrap() -> None:
    g1 = np.array([1.0, 2.0])
    g2 = np.array([5.0, 5.0])
    out = td_lambda_mix([g1, g2], td_lambda=0.0)
    assert np.allclose(out, g1)


def test_td_lambda_mix_one_recovers_last_bootstrap() -> None:
    g1 = np.array([1.0, 2.0])
    g2 = np.array([5.0, 5.0])
    out = td_lambda_mix([g1, g2], td_lambda=1.0)
    assert np.allclose(out, g2)


def test_td_lambda_mix_single_bootstrap() -> None:
    g = np.array([3.0])
    assert np.allclose(td_lambda_mix([g], td_lambda=0.7), g)


def test_td_lambda_mix_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        td_lambda_mix([np.zeros(1)], td_lambda=1.5)


def test_n_step_adjoint_bootstrap_matches_recursion() -> None:
    m = np.array([[0.9, 0.0], [0.0, 0.9]])
    c = np.zeros(2)
    terminal_pred = np.array([1.0, -1.0])
    out = n_step_adjoint_bootstrap([m, m], [c, c], terminal_pred)
    expected = m.T @ (m.T @ terminal_pred)
    assert np.allclose(out, expected)


def test_total_state_cost_gradient_matches_manual() -> None:
    dl_dy = np.array([1.0, 0.0])
    dl_du = np.array([2.0])
    dpi_dy = np.array([[0.5, -0.5]])
    out = total_state_cost_gradient(dl_dy, dl_du, dpi_dy)
    assert np.allclose(out, dl_dy + dpi_dy.T @ dl_du)
