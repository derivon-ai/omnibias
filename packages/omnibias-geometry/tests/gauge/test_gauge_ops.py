# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gauge ops: field strength, self-duality, EOM, Bianchi, action, charge, gauge."""

from __future__ import annotations

import numpy as np
import omnibias.geometry.gauge._core.lie_algebra as la
import pytest
from _gauge_helpers import instanton_arrays

SIG_E = (1, 1, 1, 1)
SIG_M = (-1, 1, 1, 1)


def test_field_strength_antisymmetry(backend, sample_points) -> None:
    su2 = la.su(2)
    a, da, _ = instanton_arrays(sample_points)
    A, dA = backend.asarray(a), backend.asarray(da)
    F = backend.ops.field_strength_from_arrays(A, dA, algebra=su2, coupling=0.9)
    Fn = backend.tonumpy(F)
    np.testing.assert_allclose(Fn, -np.swapaxes(Fn, 1, 2), atol=1e-11)


def test_abelian_limit_zero_coupling(backend, sample_points) -> None:
    su2 = la.su(2)
    a, da, _ = instanton_arrays(sample_points)
    A, dA = backend.asarray(a), backend.asarray(da)
    F = backend.ops.field_strength_from_arrays(A, dA, algebra=su2, coupling=0.0)
    expect = da - np.swapaxes(da, 1, 2)
    np.testing.assert_allclose(backend.tonumpy(F), expect, atol=1e-11)


def test_u1_is_pure_exterior_derivative(backend) -> None:
    u1 = la.u1()
    rng = np.random.default_rng(3)
    a = rng.standard_normal((6, 4, 1))
    da = rng.standard_normal((6, 4, 4, 1))
    F = backend.ops.field_strength_from_arrays(
        backend.asarray(a), backend.asarray(da), algebra=u1, coupling=2.5
    )
    np.testing.assert_allclose(backend.tonumpy(F), da - np.swapaxes(da, 1, 2), atol=1e-11)


def test_instanton_is_self_dual(backend, sample_points) -> None:
    su2 = la.su(2)
    a, da, _ = instanton_arrays(sample_points)
    F = backend.ops.field_strength_from_arrays(
        backend.asarray(a), backend.asarray(da), algebra=su2, coupling=1.0
    )
    defect = backend.ops.self_duality_defect(F, signature=SIG_E)
    assert np.abs(backend.tonumpy(defect)).max() < 1e-9


def test_instanton_satisfies_equation_of_motion(backend, sample_points) -> None:
    su2 = la.su(2)
    a, da, dda = instanton_arrays(sample_points)
    eom = backend.ops.covariant_divergence_from_arrays(
        backend.asarray(a),
        backend.asarray(da),
        backend.asarray(dda),
        algebra=su2,
        coupling=1.0,
        signature=SIG_E,
    )
    assert np.abs(backend.tonumpy(eom)).max() < 1e-7


def test_bianchi_identity_vanishes(backend, sample_points) -> None:
    su2 = la.su(2)
    a, da, dda = instanton_arrays(sample_points)
    bia = backend.ops.bianchi_from_arrays(
        backend.asarray(a),
        backend.asarray(da),
        backend.asarray(dda),
        algebra=su2,
        coupling=1.0,
        signature=SIG_E,
    )
    assert np.abs(backend.tonumpy(bia)).max() < 1e-7


def test_double_hodge_star_sign(backend, sample_points) -> None:
    """** = (-1)^{k(d-k)} sign(det g): +1 Euclidean, -1 Minkowski (2-form, d=4)."""
    su2 = la.su(2)
    a, da, _ = instanton_arrays(sample_points)
    F = backend.ops.field_strength_from_arrays(
        backend.asarray(a), backend.asarray(da), algebra=su2, coupling=1.0
    )
    dd_e = backend.ops.dual_field_strength(
        backend.ops.dual_field_strength(F, signature=SIG_E), signature=SIG_E
    )
    np.testing.assert_allclose(backend.tonumpy(dd_e), backend.tonumpy(F), atol=1e-9)
    dd_m = backend.ops.dual_field_strength(
        backend.ops.dual_field_strength(F, signature=SIG_M), signature=SIG_M
    )
    np.testing.assert_allclose(backend.tonumpy(dd_m), -backend.tonumpy(F), atol=1e-9)


def test_self_dual_projectors(backend, sample_points) -> None:
    su2 = la.su(2)
    a, da, _ = instanton_arrays(sample_points)
    F = backend.ops.field_strength_from_arrays(
        backend.asarray(a), backend.asarray(da), algebra=su2, coupling=1.0
    )
    sd = backend.ops.self_dual_projector(F, signature=SIG_E)
    asd = backend.ops.anti_self_dual_projector(F, signature=SIG_E)
    # the instanton is purely self-dual
    np.testing.assert_allclose(backend.tonumpy(sd), backend.tonumpy(F), atol=1e-9)
    assert np.abs(backend.tonumpy(asd)).max() < 1e-9


def test_infinitesimal_gauge_invariance(backend, sample_points) -> None:
    su2 = la.su(2)
    a, da, _ = instanton_arrays(sample_points)
    F = backend.ops.field_strength_from_arrays(
        backend.asarray(a), backend.asarray(da), algebra=su2, coupling=1.0
    )
    rng = np.random.default_rng(11)
    omega = backend.asarray(rng.standard_normal((sample_points.shape[0], su2.dim)))
    defect = backend.ops.gauge_invariance_defect(
        F, omega, algebra=su2, coupling=1.0, signature=SIG_E
    )
    assert np.abs(backend.tonumpy(defect)).max() < 1e-9


def test_bracket_antisymmetry_and_definition(backend) -> None:
    su2 = la.su(2)
    rng = np.random.default_rng(5)
    x = backend.asarray(rng.standard_normal((10, 3)))
    y = backend.asarray(rng.standard_normal((10, 3)))
    bxy = backend.tonumpy(backend.ops.bracket(x, y, su2))
    byx = backend.tonumpy(backend.ops.bracket(y, x, su2))
    np.testing.assert_allclose(bxy, -byx, atol=1e-12)
    f = su2.structure_constants()
    expect = np.einsum("abc,Ba,Bb->Bc", f, backend.tonumpy(x), backend.tonumpy(y))
    np.testing.assert_allclose(bxy, expect, atol=1e-12)


# --- heavier grid-integral diagnostics (single backend, not parametrized) ----
def _instanton_grid_F(grid: np.ndarray, rho: float = 1.0) -> np.ndarray:
    from _gauge_helpers import thooft_eta

    eta = np.transpose(thooft_eta(), (1, 2, 0))  # (mu, nu, a)
    d = (grid**2).sum(1) + rho**2
    return -4.0 * rho**2 * eta[None] / (d[:, None, None, None] ** 2)


def test_instanton_topological_charge_is_one() -> None:
    import torch

    torch.set_default_dtype(torch.float64)
    import omnibias.geometry.gauge.torch.ops as ops

    axis = np.linspace(-9.0, 9.0, 49)
    cell = (axis[1] - axis[0]) ** 4
    grid = np.stack(np.meshgrid(*([axis] * 4), indexing="ij"), -1).reshape(-1, 4)
    q = s = 0.0
    for chunk in np.array_split(grid, 64):
        F = torch.as_tensor(_instanton_grid_F(chunk))
        q += float(torch.sum(ops.topological_charge_density(F)) * cell)
        s += float(ops.yang_mills_action(F, signature=SIG_E) * cell)
    assert q == pytest.approx(1.0, abs=0.03)
    assert s == pytest.approx(8.0 * np.pi**2, abs=0.5)
    assert ops.is_quantized(q, tol=0.05)
