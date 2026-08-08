# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for SDF primitives, R-functions, ADF, and SDF sampling."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.pinn.domain import (
    Box,
    Sphere,
    approximate_distance,
    boundary_points_sdf,
    interior_points_sdf,
    intersect,
    normalize_adf,
    union,
)
from omnibias.pinn.solver import Domain


def test_sphere_negative_inside():
    s = Sphere(center=(0.0, 0.0), radius=1.0)
    inside = s(np.array([[0.0, 0.0]]))
    outside = s(np.array([[2.0, 0.0]]))
    assert inside[0] < 0
    assert outside[0] > 0
    assert s(np.array([[1.0, 0.0]]))[0] == pytest.approx(0.0, abs=1e-12)


def test_box_negative_inside():
    b = Box(lo=(-1.0, -1.0), hi=(1.0, 1.0))
    assert b(np.array([[0.0, 0.0]]))[0] < 0
    assert b(np.array([[2.0, 0.0]]))[0] > 0


def test_r_conjunction_vanishes_when_either_factor_is_zero():
    # R0 conjunction is zero whenever either argument is zero *and* the
    # other is non-negative (the classical Rvachev positive-inside regime).
    from omnibias.pinn.domain import r_conjunction

    assert r_conjunction(0.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert r_conjunction(2.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    # Intersection of two positive-inside disks via negated SDFs.
    a = Sphere(center=(0.0, 0.0), radius=1.0)
    b = Box(lo=(-0.5, -0.5), hi=(0.5, 0.5))
    # Negate so "inside" is positive, then R-and, then negate back.
    from omnibias.pinn.domain import Negate

    both = Negate(intersect(Negate(a), Negate(b)))
    face = np.array([[0.5, 0.0]])
    assert both(face)[0] == pytest.approx(0.0, abs=1e-6)


def test_union_contains_either():
    a = Sphere(center=(-0.5, 0.0), radius=0.6)
    b = Sphere(center=(0.5, 0.0), radius=0.6)
    u = union(a, b)
    assert u(np.array([[0.0, 0.0]]))[0] < 0


def test_normalize_adf_unit_gradient_near_boundary():
    s = Sphere(center=(0.0, 0.0), radius=1.0)
    # Point just outside.
    X = np.array([[1.01, 0.0]])
    phi = approximate_distance(s, X, h=1e-5)
    # Finite-diff grad of phi should be near 1 in magnitude.
    from omnibias.pinn.domain._core.adf import fd_gradient

    # Build a tiny wrapper SDF that returns phi.
    class _Phi:
        ndim = 2

        def __call__(self, Y):
            return approximate_distance(s, Y, h=1e-5)

    g = fd_gradient(_Phi(), X, h=1e-5)
    assert float(np.linalg.norm(g[0])) == pytest.approx(1.0, abs=5e-2)


def test_normalize_adf_helper():
    omega = np.array([0.0, 3.0])
    grad = np.array([[1.0, 0.0], [0.0, 4.0]])
    phi = normalize_adf(omega, grad)
    assert phi[0] == pytest.approx(0.0)
    assert phi[1] == pytest.approx(3.0 / 5.0)


def test_interior_sampling_all_inside():
    s = Sphere(center=(0.0, 0.0), radius=1.0)
    pts = interior_points_sdf(s, [(-1.5, 1.5), (-1.5, 1.5)], n=32, seed=0)
    assert pts.shape == (32, 2)
    assert np.all(s(pts) < 0)


def test_boundary_sampling_near_zero():
    s = Sphere(center=(0.0, 0.0), radius=1.0)
    pts = boundary_points_sdf(
        s, [(-1.5, 1.5), (-1.5, 1.5)], n=16, seed=1, tol=1e-5
    )
    assert pts.shape == (16, 2)
    assert np.max(np.abs(s(pts))) < 1e-4


def test_domain_accepts_sdf():
    s = Sphere(center=(0.0, 0.0), radius=1.0)
    d = Domain(("x", "y"), [(-1.5, 1.5), (-1.5, 1.5)], sdf=s)
    assert d.is_sdf
    assert d.spatial_bounds() == ((-1.5, 1.5), (-1.5, 1.5))


def test_domain_rejects_sdf_ndim_mismatch():
    s = Sphere(center=(0.0, 0.0), radius=1.0)
    with pytest.raises(ValueError, match="n_spatial"):
        Domain(("x", "y", "z"), [(-1, 1), (-1, 1), (-1, 1)], sdf=s)
