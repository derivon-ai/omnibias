# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
import numpy as np
import pytest
from omnibias.geometry.neuromanifold import (
    ObservationMetric,
    affine_quotient,
    escape_candidates,
    extrinsic_geometry,
    hidden_permutation,
    hidden_sign,
    higher_order_visibility,
    homogeneous_anchor_chart,
    least_squares_hessian,
    quotient_step,
    rank_report,
)


def test_visibility_is_not_intrinsic_singularity():
    assert higher_order_visibility([np.array([0.0]), np.array([2.0])]).first_visible_order == 2
    cube = higher_order_visibility([np.array([0.0]), np.array([0.0]), np.array([6.0])])
    assert cube.first_visible_order == 3
    assert higher_order_visibility([np.zeros(2)] * 4).status.startswith("inconclusive")
    # Cusp (t^2,t^3): finite jet visibility alone does not prove image regularity.
    assert (
        higher_order_visibility(
            [np.zeros(2), np.array([2.0, 0.0]), np.array([0.0, 6.0])]
        ).first_visible_order
        == 2
    )
    assert rank_report(np.zeros((1, 3))).certified_upper is None
    candidates = escape_candidates(np.array([[1.0]]), np.array([[0.0, 0.0, -12.0]]))
    np.testing.assert_array_equal(candidates[0], [1.0])


def test_affine_quotient_and_reduced_step():
    chart = affine_quotient([[1, 1, 0], [0, 0, 1]], observation_scope="complete_coefficients")
    theta = np.array([2.0, 3.0, -1.0])
    a = np.asarray(chart.source, dtype=float)
    np.testing.assert_array_equal(a @ theta, a @ chart.embed(chart.retract(theta)))
    assert chart.valid_at(a)
    assert not chart.valid_at(a * 0)
    assert chart.dim == 2 and len(chart.kernel) == 1
    step = quotient_step(
        np.array([3.0, -1.0]), np.array([3.0, -1.0]), np.eye(2), lambda u: 0.5 * float(u @ u)
    )
    assert step.accepted and step.loss == 0


def test_circle_second_fundamental_and_loss_curvature():
    t = 0.4
    j = np.array([[-np.sin(t)], [np.cos(t)]])
    h = np.array([[[-np.cos(t)]], [[-np.sin(t)]]])
    metric = ObservationMetric(np.ones(2))
    geometry = extrinsic_geometry(j, metric, h)
    np.testing.assert_allclose(geometry.second_fundamental, h, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        geometry.tangent_projection @ geometry.normal_projection, 0, atol=1e-14
    )
    residual = np.array([np.cos(t) - 2.0, np.sin(t)])
    np.testing.assert_allclose(least_squares_hessian(j, h, residual, metric), [[2 * np.cos(t)]])
    with pytest.raises(ValueError, match="regular"):
        extrinsic_geometry(np.zeros((2, 1)), metric)


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
def test_hidden_actions(activation):
    w = np.array([[0.3, -0.5], [0.7, 0.2]])
    b = np.array([0.1, -0.2])
    v = np.array([[0.4, -0.6]])
    c = np.array([0.8])
    x = np.array([0.7, 0.9])
    act = np.tanh if activation == "tanh" else lambda z: 1 / (1 + np.exp(-z))
    original = v @ act(w @ x + b) + c
    for transformed in (
        hidden_permutation(w, b, v, c, (1, 0)),
        hidden_sign(w, b, v, c, np.array([-1.0, 1.0]), activation=activation),
    ):
        np.testing.assert_allclose(
            transformed.outgoing @ act(transformed.incoming @ x + transformed.bias)
            + transformed.next_bias,
            original,
            atol=1e-14,
        )
    chart = homogeneous_anchor_chart(w, b, degree=2)
    normalized = chart.normalize(w, b, v, c)
    np.testing.assert_allclose(
        normalized.outgoing @ (normalized.incoming @ x + normalized.bias) ** 2 + c,
        v @ (w @ x + b) ** 2 + c,
    )


def test_exact_controls_and_undersampling_do_not_invent_symmetries():
    from dataclasses import replace
    from fractions import Fraction

    from omnibias.geometry.neuromanifold.strata import (
        diagnose_stratum,
        exact_rank_report,
        monomial_curve_stratum,
    )

    assert monomial_curve_stratum((2,)).real_image_has_boundary
    assert monomial_curve_stratum((3,)).intrinsic_class == "smooth_curve"
    assert monomial_curve_stratum((2, 3)).intrinsic_class == "intrinsic_algebraic_curve_singularity"
    assert monomial_curve_stratum((3, 6)).intrinsic_class == "smooth_curve"
    chart = affine_quotient([[1, 1]])
    # The same Jacobian occurs at zero for a+b+b^3; a fiber is not inferred.
    diagnostic = diagnose_stratum(np.array([[1.0, 1.0]]), known_affine_chart=chart)
    assert any("identity_unresolved" in reason for reason in diagnostic.reasons)
    bad = replace(chart, projection=((Fraction(1), Fraction(0)),))
    assert not bad.valid_at(np.array([[1.0, 1.0]]))
    with pytest.raises(ValueError):
        bad.retract(np.ones(2))
    report, witness = exact_rank_report(np.array([[1.0, 1.0], [1.0, 1.0 + 2**-50]]))
    assert report.certified_lower == report.certified_upper == 2 and witness.verify()
    undersampled = diagnose_stratum(np.array([[1.0, 1.0]]), complete_jacobian=np.eye(2))
    assert "numerical_observation_deficiency" in undersampled.reasons


def test_metric_symmetry_budget_and_existing_natural_solver_bridge():
    import torch
    from omnibias.torch.optim import NaturalGradient

    weight = np.array([[2.0, 1.0 + 2e-15], [1.0, 3.0]])
    observation_metric = ObservationMetric(weight)
    np.testing.assert_array_equal(observation_metric.weight, observation_metric.weight.T)
    with pytest.raises(ValueError, match="budget"):
        affine_quotient([[1, 2]] * 100, max_entries=20)
    with pytest.raises(ValueError, match="direction"):
        quotient_step(
            np.ones(2),
            np.ones(2),
            np.eye(2),
            lambda q: float(q @ q),
            solve=lambda _a, _b: np.zeros(3),
        )

    # The known affine realization has a complete exact fiber kernel. Train
    # its regular coordinates with the already-shipped metric optimizer.
    chart = affine_quotient([[1, 1, 0], [0, 0, 1]])
    q = torch.nn.Parameter(torch.tensor([3.0, -1.0], dtype=torch.float64))
    observed = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=q.dtype)
    target = torch.tensor([1.0, 2.0], dtype=q.dtype)
    optimizer = NaturalGradient([q], damping=0.0, metric=lambda _: observed.T @ observed)
    optimizer.step(lambda: 0.5 * (observed @ q - target).square().sum())
    torch.testing.assert_close(q, torch.linalg.solve(observed, target), rtol=1e-12, atol=1e-12)
    full_parameters = chart.embed(q.detach().numpy())
    np.testing.assert_allclose(chart.retract(full_parameters), q.detach().numpy())

    calls = []

    def existing_solve(matrix, rhs):
        calls.append(matrix.copy())
        return torch.linalg.solve(torch.from_numpy(matrix), torch.from_numpy(rhs)).numpy()

    reduced = quotient_step(
        np.ones(2),
        np.ones(2),
        np.eye(2),
        lambda u: 0.5 * float(u @ u),
        damping=0.1,
        solve=existing_solve,
    )
    assert reduced.accepted and len(calls) == 1
    np.testing.assert_array_equal(calls[0], 1.1 * np.eye(2))
