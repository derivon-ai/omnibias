# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent equilibrium branches test the generic switch and its refusals."""

from dataclasses import replace

import numpy as np
import pytest
from omnibias.geometry.continuation import ImplicitFamily, continue_branch
from omnibias.geometry.continuation_switch import switch_equilibrium_branch


def pitchfork():
    return ImplicitFamily(
        lambda z: np.array([z[0] * (z[1] - z[0] ** 2)]),
        lambda z: np.array([[z[1] - 3 * z[0] ** 2, z[0]]]),
        lambda z: np.array([[[-6 * z[0], 1.0], [1.0, 0.0]]]),
        lambda z: np.array([[[[-6.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
    )


@pytest.mark.parametrize("side", [-1, 1])
def test_pitchfork_switch_leaves_parent_and_continues_independent_branch(side):
    family = pitchfork()
    switched = switch_equilibrium_branch(
        family, np.zeros(2), np.array([0.0, 1.0]), side=side, step=0.02
    )
    assert switched.status == "switched" and not switched.certified
    assert switched.seed is not None and switched.tangent is not None
    x, parameter = switched.seed
    assert side * x > 0.019
    assert parameter == pytest.approx(x * x, abs=1e-12)
    np.testing.assert_allclose(switched.branch_curvature, [0.0, 2.0], atol=1e-12)
    branch = continue_branch(
        family, switched.seed, direction=switched.tangent, step=0.02, n_steps=20
    )
    assert branch.status == "complete" and side * branch.points[-1, 0] > 0.3
    np.testing.assert_allclose(branch.points[:, 1], branch.points[:, 0] ** 2, atol=1e-9)


@pytest.mark.parametrize("side", [-1, 1])
def test_transcritical_switch_selects_diagonal_instead_of_parent_axis(side):
    family = ImplicitFamily(
        lambda z: np.array([z[0] * (z[1] - z[0])]),
        lambda z: np.array([[z[1] - 2 * z[0], z[0]]]),
        lambda z: np.array([[[-2.0, 1.0], [1.0, 0.0]]]),
        lambda z: np.zeros((1, 2, 2, 2)),
    )
    result = switch_equilibrium_branch(family, np.zeros(2), np.array([0.0, 1.0]), side=side)
    assert result.status == "switched" and result.seed is not None
    assert result.seed[0] == pytest.approx(side * 0.01)
    assert result.seed[1] == pytest.approx(result.seed[0], abs=1e-12)


def test_coupled_state_rotated_residual_and_nonlinear_corrector():
    # Exact non-parent branch: lambda=x²+x⁴, auxiliary=x².
    # A nonsingular output mixing prevents dependence on residual ordering.
    rotation = np.array([[1.0, 2.0], [-2.0, 1.0]])

    def value(z):
        x, auxiliary, parameter = z
        return rotation @ np.array([auxiliary - x * x, x * (parameter - x * x - x**4)])

    def jac(z):
        x, _, parameter = z
        return rotation @ np.array([[-2 * x, 1.0, 0.0], [parameter - 3 * x * x - 5 * x**4, 0.0, x]])

    def second(z):
        x = z[0]
        h = np.zeros((2, 3, 3))
        h[0, 0, 0] = -2
        h[1, 0, 0] = -6 * x - 20 * x**3
        h[1, 0, 2] = h[1, 2, 0] = 1
        return np.einsum("ab,bij->aij", rotation, h)

    def third(z):
        t = np.zeros((2, 3, 3, 3))
        t[1, 0, 0, 0] = -6 - 60 * z[0] ** 2
        return np.einsum("ab,bijk->aijk", rotation, t)

    result = switch_equilibrium_branch(
        ImplicitFamily(value, jac, second, third), np.zeros(3), np.array([0.0, 0.0, 1.0]), step=0.04
    )
    assert result.status == "switched" and result.seed is not None
    x, auxiliary, parameter = result.seed
    assert abs(x) > 0.039
    assert auxiliary == pytest.approx(x * x, abs=1e-12)
    assert parameter == pytest.approx(x * x + x**4, abs=1e-12)
    assert abs(parameter - x * x) > 1e-6  # the corrector improved the cubic predictor


def test_switch_is_valid_after_state_parameter_coordinate_mixing():
    original = pitchfork()
    transform = np.array([[1.0, -0.35], [0.0, 1.0]])
    family = ImplicitFamily(
        lambda z: original.value(transform @ z),
        lambda z: original.jacobian(transform @ z) @ transform,
        lambda z: np.einsum("ijk,ja,kb->iab", original.second(transform @ z), transform, transform),
        lambda z: np.einsum(
            "ijkl,ja,kb,lc->iabc", original.third(transform @ z), transform, transform, transform
        ),
    )
    result = switch_equilibrium_branch(family, np.zeros(2), np.array([0.35, 1.0]), step=0.03)
    assert result.status == "switched" and result.seed is not None
    x, parameter = result.seed
    transverse_state = x - 0.35 * parameter
    assert transverse_state > 0.02
    assert parameter == pytest.approx(transverse_state**2, abs=1e-11)


def test_unsupported_degeneracy_parent_and_corrector_budget_are_explicit():
    family = pitchfork()
    parent = np.array([0.0, 1.0])
    assert switch_equilibrium_branch(
        replace(family, third=None), np.zeros(2), parent
    ).reason.startswith("second_and_third")
    assert (
        switch_equilibrium_branch(family, np.zeros(2), np.array([1.0, 0.0])).reason
        == "parent_tangent_not_supported"
    )
    assert (
        switch_equilibrium_branch(family, np.array([1.0, 0.0]), parent).reason
        == "point_does_not_solve_residual"
    )
    higher = ImplicitFamily(
        lambda z: np.zeros(2),
        lambda z: np.zeros((2, 3)),
        lambda z: np.zeros((2, 3, 3)),
        lambda z: np.zeros((2, 3, 3, 3)),
    )
    assert (
        switch_equilibrium_branch(higher, np.zeros(3), np.array([0.0, 0.0, 1.0])).reason
        == "unsupported_rank_or_higher_codimension"
    )
    # The analytic branch solves lambda+lambda²=x². One Newton step cannot
    # reach this tolerance from its cubic predictor at the requested spread.
    tensor = np.zeros((1, 2, 2, 2))
    tensor[0, 0, 0, 0] = -6
    tensor[0, 0, 1, 1] = tensor[0, 1, 0, 1] = tensor[0, 1, 1, 0] = 2
    degenerate = ImplicitFamily(
        lambda z: np.array([z[0] * (z[1] ** 2 - z[0] ** 2)]),
        lambda z: np.array([[z[1] ** 2 - 3 * z[0] ** 2, 2 * z[0] * z[1]]]),
        lambda z: np.array([[[-6 * z[0], 2 * z[1]], [2 * z[1], 2 * z[0]]]]),
        lambda z: tensor,
    )
    assert (
        switch_equilibrium_branch(degenerate, np.zeros(2), parent).reason
        == "degenerate_mixed_quadratic_crossing"
    )
    curved = ImplicitFamily(
        lambda z: np.array([z[0] * (z[1] + z[1] ** 2 - z[0] ** 2)]),
        lambda z: np.array([[z[1] + z[1] ** 2 - 3 * z[0] ** 2, z[0] * (1 + 2 * z[1])]]),
        lambda z: np.array([[[-6 * z[0], 1 + 2 * z[1]], [1 + 2 * z[1], 2 * z[0]]]]),
        lambda z: tensor,
    )
    result = switch_equilibrium_branch(
        curved, np.zeros(2), parent, step=0.25, max_corrector=1, max_halvings=0, tol=1e-12
    )
    assert result.status == "inconclusive" and result.reason == "off_parent_corrector_failed"
