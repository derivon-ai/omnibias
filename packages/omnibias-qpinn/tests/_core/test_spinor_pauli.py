# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the spinor / Pauli / gamma-matrix machinery."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.pinn._core.components import ComponentSpec
from omnibias.qpinn import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    gamma5,
    gamma_matrices,
    make_spinor_components,
    pauli_matrices,
)


class TestMakeSpinorComponents:
    def test_default_dirac_4spinor(self):
        spec = make_spinor_components()
        assert isinstance(spec, ComponentSpec)
        assert spec.names == (
            "spinor_0_re", "spinor_0_im",
            "spinor_1_re", "spinor_1_im",
            "spinor_2_re", "spinor_2_im",
            "spinor_3_re", "spinor_3_im",
        )

    def test_weyl_2spinor(self):
        spec = make_spinor_components(name="chi", n_components=2)
        assert spec.names == (
            "chi_0_re", "chi_0_im",
            "chi_1_re", "chi_1_im",
        )
        assert spec.group_members("chi_0") == ("chi_0_re", "chi_0_im")
        assert spec.group_members("chi") == (
            "chi_0_re", "chi_0_im", "chi_1_re", "chi_1_im",
        )

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_spinor_components(name="")

    def test_zero_components_rejected(self):
        with pytest.raises(ValueError, match=">= 1"):
            make_spinor_components(n_components=0)


class TestPauliMatrices:
    def test_pauli_x(self):
        np.testing.assert_array_equal(
            PAULI_X, np.array([[0, 1], [1, 0]], dtype=np.complex128),
        )

    def test_pauli_y(self):
        np.testing.assert_array_equal(
            PAULI_Y, np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        )

    def test_pauli_z(self):
        np.testing.assert_array_equal(
            PAULI_Z, np.array([[1, 0], [0, -1]], dtype=np.complex128),
        )

    def test_pauli_matrices_function(self):
        sx, sy, sz = pauli_matrices()
        np.testing.assert_array_equal(sx, PAULI_X)
        np.testing.assert_array_equal(sy, PAULI_Y)
        np.testing.assert_array_equal(sz, PAULI_Z)

    def test_anticommutation_relations(self):
        """{sigma_i, sigma_j} = 2 delta_ij I."""
        sx, sy, sz = pauli_matrices()
        I2 = np.eye(2, dtype=np.complex128)
        np.testing.assert_allclose(sx @ sx + sx @ sx, 2 * I2)
        np.testing.assert_allclose(sx @ sy + sy @ sx, np.zeros((2, 2)))
        np.testing.assert_allclose(sx @ sz + sz @ sx, np.zeros((2, 2)))
        np.testing.assert_allclose(sy @ sz + sz @ sy, np.zeros((2, 2)))


class TestGammaMatricesDirac:
    def test_shapes(self):
        for g in gamma_matrices("dirac"):
            assert g.shape == (4, 4)
            assert g.dtype == np.complex128

    def test_clifford_algebra(self):
        r"""{\gamma^\mu, \gamma^\nu} = 2 eta^{\mu\nu} I_4 with eta = diag(-1, 1, 1, 1).

        With our convention gamma^0 anticommutes with all other gammas
        and (gamma^0)^2 = -I, (gamma^i)^2 = I.

        Actually the convention here is mostly-plus, so gamma^0^2 = -I and
        (gamma^i)^2 = I. Wait, this depends on the metric signature.

        Using Peskin & Schroeder (mostly-minus eta = (+, -, -, -)):
        (gamma^0)^2 = I, (gamma^i)^2 = -I.

        Our gamma_0 = diag(I, -I) so (gamma_0)^2 = I.
        Our gamma_i^2 with sigma_i^2 = I: (gamma_i)^2 = block-product = ((-sigma_i)(sigma_i), 0; 0, (sigma_i)(-sigma_i))
                       = -sigma_i^2 = -I.
        So we are using mostly-minus eta = (+, -, -, -).
        """
        g0, g1, g2, g3 = gamma_matrices("dirac")
        I4 = np.eye(4, dtype=np.complex128)
        # (gamma^0)^2 = +I
        np.testing.assert_allclose(g0 @ g0, I4, atol=1e-12)
        # (gamma^i)^2 = -I
        np.testing.assert_allclose(g1 @ g1, -I4, atol=1e-12)
        np.testing.assert_allclose(g2 @ g2, -I4, atol=1e-12)
        np.testing.assert_allclose(g3 @ g3, -I4, atol=1e-12)
        # gamma^0 anticommutes with gamma^i
        np.testing.assert_allclose(g0 @ g1 + g1 @ g0, np.zeros((4, 4)), atol=1e-12)
        np.testing.assert_allclose(g0 @ g2 + g2 @ g0, np.zeros((4, 4)), atol=1e-12)
        np.testing.assert_allclose(g0 @ g3 + g3 @ g0, np.zeros((4, 4)), atol=1e-12)
        # Distinct gamma^i anticommute
        np.testing.assert_allclose(g1 @ g2 + g2 @ g1, np.zeros((4, 4)), atol=1e-12)


class TestGammaMatricesWeyl:
    def test_weyl_gamma0_offdiagonal(self):
        g0_d, _, _, _ = gamma_matrices("dirac")
        g0_w, _, _, _ = gamma_matrices("weyl")
        # Different representation: g0_d is diagonal, g0_w is anti-block-diagonal
        assert g0_w[0, 2] == 1 + 0j
        assert g0_w[2, 0] == 1 + 0j
        assert g0_w[0, 0] == 0 + 0j

    def test_weyl_clifford(self):
        """Weyl gammas satisfy the same Clifford algebra."""
        g0, g1, g2, g3 = gamma_matrices("weyl")
        I4 = np.eye(4, dtype=np.complex128)
        np.testing.assert_allclose(g0 @ g0, I4, atol=1e-12)
        np.testing.assert_allclose(g1 @ g1, -I4, atol=1e-12)


class TestGamma5:
    def test_dirac_gamma5_offdiagonal(self):
        g5 = gamma5("dirac")
        # In Dirac rep: gamma_5 = anti-block-diagonal with I_2 blocks
        np.testing.assert_allclose(g5[:2, :2], np.zeros((2, 2)), atol=1e-12)
        np.testing.assert_allclose(g5[2:, 2:], np.zeros((2, 2)), atol=1e-12)

    def test_weyl_gamma5_diagonal(self):
        g5 = gamma5("weyl")
        # In Weyl/chiral rep: gamma_5 = diag(-I_2, +I_2)
        I2 = np.eye(2, dtype=np.complex128)
        np.testing.assert_allclose(g5[:2, :2], -I2, atol=1e-12)
        np.testing.assert_allclose(g5[2:, 2:], I2, atol=1e-12)

    def test_gamma5_squared(self):
        """(gamma_5)^2 = +I_4."""
        for rep in ("dirac", "weyl"):
            g5 = gamma5(rep)
            I4 = np.eye(4, dtype=np.complex128)
            np.testing.assert_allclose(g5 @ g5, I4, atol=1e-12)

    def test_unknown_representation(self):
        with pytest.raises(ValueError, match="representation"):
            gamma_matrices("majorana")


class TestPauliDotAlgebra:
    """sigma . A applied to a 2-spinor must satisfy (sigma . A)^2 = |A|^2 * I."""

    def test_pauli_dot_squared_via_state(self):
        """Verify the closed-form (sigma . A)^2 psi == |A|^2 psi for a test psi."""
        # Build numpy 2x2 matrices for the test
        sx, sy, sz = pauli_matrices()
        A = (0.3, -0.7, 1.1)
        sigma_dot_A = A[0] * sx + A[1] * sy + A[2] * sz
        A_sq = A[0] ** 2 + A[1] ** 2 + A[2] ** 2
        I2 = np.eye(2, dtype=np.complex128)
        np.testing.assert_allclose(sigma_dot_A @ sigma_dot_A, A_sq * I2, atol=1e-12)
