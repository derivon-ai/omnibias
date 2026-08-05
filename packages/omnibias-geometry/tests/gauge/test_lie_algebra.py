# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Lie-algebra structure constants: normalization, antisymmetry, Jacobi identity."""

from __future__ import annotations

import numpy as np
import omnibias.geometry.gauge._core.lie_algebra as la
import pytest
from omnibias.geometry.gauge._core import forms


def test_su2_structure_constants_are_levi_civita() -> None:
    su2 = la.su(2)
    assert su2.dim == 3
    assert su2.n_fundamental == 2
    np.testing.assert_allclose(
        su2.structure_constants(), forms.levi_civita_symbol(3), atol=1e-12
    )


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_generator_normalization(n: int) -> None:
    """tr(T^a T^b) = 1/2 delta^{ab}."""
    g = la.su(n).generators()
    gram = np.einsum("aij,bji->ab", g, g)
    np.testing.assert_allclose(gram, 0.5 * np.eye(n * n - 1), atol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_structure_constants_totally_antisymmetric(n: int) -> None:
    f = la.su(n).structure_constants()
    np.testing.assert_allclose(f, -np.swapaxes(f, 0, 1), atol=1e-12)
    np.testing.assert_allclose(f, -np.swapaxes(f, 1, 2), atol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_symmetric_constants_symmetric(n: int) -> None:
    d = la.su(n).symmetric_constants()
    np.testing.assert_allclose(d, np.swapaxes(d, 0, 1), atol=1e-12)
    np.testing.assert_allclose(d, np.swapaxes(d, 1, 2), atol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_jacobi_identity(n: int) -> None:
    f = la.su(n).structure_constants()
    jac = (
        np.einsum("ade,bcd->abce", f, f)
        + np.einsum("bde,cad->abce", f, f)
        + np.einsum("cde,abd->abce", f, f)
    )
    np.testing.assert_allclose(jac, 0.0, atol=1e-9)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_casimir_invariant_sum(n: int) -> None:
    """sum_{abc} f^{abc} f^{abc} = N (N^2 - 1) (the adjoint Casimir)."""
    f = la.su(n).structure_constants()
    assert float((f**2).sum()) == pytest.approx(n * (n * n - 1), rel=1e-9)


def test_su3_matches_gell_mann_convention() -> None:
    f = la.su(3).structure_constants()
    assert f[0, 1, 2] == pytest.approx(1.0)
    assert f[3, 4, 7] == pytest.approx(np.sqrt(3) / 2)
    assert f[5, 6, 7] == pytest.approx(np.sqrt(3) / 2)
    d = la.su(3).symmetric_constants()
    assert d[0, 0, 7] == pytest.approx(1.0 / np.sqrt(3))


def test_u1_is_abelian() -> None:
    u1 = la.u1()
    assert u1.dim == 1
    assert u1.is_abelian
    np.testing.assert_allclose(u1.structure_constants(), 0.0, atol=1e-12)


def test_as_lie_algebra_label_resolution() -> None:
    assert la.as_lie_algebra("su(2)").dim == 3
    assert la.as_lie_algebra("SU(3)").dim == 8
    assert la.as_lie_algebra("u(1)").dim == 1
    with pytest.raises(ValueError):
        la.as_lie_algebra("so(3)")
