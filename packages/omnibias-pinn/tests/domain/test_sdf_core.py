# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for SDF primitives, R-functions, ADF, boundary factors, and sampling."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.pinn.domain import (
    Box,
    Halfspace,
    Polygon,
    Sphere,
    approximate_distance,
    boundary_factor_jet,
    boundary_junction_mask,
    boundary_points_sdf,
    complement,
    interior_points_sdf,
    intersect,
    normalize_adf,
    omega_gradient,
    r_conjunction,
    r_disjunction,
    r_intersect_sdf,
    r_union_sdf,
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


def test_halfspace_outward_normal_negative_inside():
    # Outward normal +x: interior x < 0 is negative.
    h = Halfspace(normal=(1.0, 0.0), point=(0.0, 0.0))
    assert h(np.array([[-0.5, 0.0]]))[0] < 0
    assert h(np.array([[0.5, 0.0]]))[0] > 0
    assert h(np.array([[0.0, 0.0]]))[0] == pytest.approx(0.0, abs=1e-12)


def test_r_conjunction_positive_inside_convention():
    assert r_conjunction(0.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert r_conjunction(2.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert r_conjunction(1.0, 1.0) > 0


def test_r_intersect_sdf_negative_inside():
    assert r_intersect_sdf(-1.0, -0.5) < 0
    assert r_intersect_sdf(-0.5, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert r_intersect_sdf(0.5, -0.5) > 0


def test_r_union_sdf_negative_inside():
    assert r_union_sdf(-1.0, 0.5) < 0
    assert r_union_sdf(0.5, 0.5) > 0
    # Inside one shape while the other is on its boundary stays interior to the union.
    assert r_union_sdf(-0.5, 0.0) < 0
    assert r_union_sdf(0.0, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_intersection_zero_on_shared_boundary_without_negate():
    a = Sphere(center=(0.0, 0.0), radius=1.0)
    b = Box(lo=(-0.5, -0.5), hi=(0.5, 0.5))
    both = intersect(a, b)
    face = np.array([[0.5, 0.0]])
    assert both(face)[0] == pytest.approx(0.0, abs=1e-6)
    assert both(np.array([[0.0, 0.0]]))[0] < 0


def test_union_contains_either():
    a = Sphere(center=(-0.5, 0.0), radius=0.6)
    b = Sphere(center=(0.5, 0.0), radius=0.6)
    u = union(a, b)
    assert u(np.array([[0.0, 0.0]]))[0] < 0
    assert u(np.array([[2.0, 0.0]]))[0] > 0


def test_disjoint_union_is_negative_only_in_components():
    a = Sphere(center=(-2.0, 0.0), radius=0.4)
    b = Sphere(center=(2.0, 0.0), radius=0.4)
    u = union(a, b)
    assert u(np.array([[-2.0, 0.0]]))[0] < 0
    assert u(np.array([[0.0, 0.0]]))[0] > 0


def test_complement_flips_sign():
    s = Sphere(center=(0.0, 0.0), radius=1.0)
    c = complement(s)
    assert c(np.array([[0.0, 0.0]]))[0] > 0
    assert c(np.array([[2.0, 0.0]]))[0] < 0


def test_concave_polygon_negative_inside():
    # L-shaped nonconvex polygon.
    poly = Polygon(
        vertices=((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0))
    )
    assert poly(np.array([[0.5, 0.5]]))[0] < 0
    assert poly(np.array([[1.5, 1.5]]))[0] > 0


def test_r_compose_preserves_zero_set_on_box_face():
    disk = Sphere(center=(0.0, 0.0), radius=1.0)
    cut = intersect(disk, Box(lo=(-1.0, -1.0), hi=(1.0, 0.5)))
    assert cut(np.array([[0.0, 0.5]]))[0] == pytest.approx(0.0, abs=1e-5)


def test_boundary_junction_detects_csg_meeting():
    b = Box(lo=(-1.0, -1.0), hi=(1.0, 1.0))
    corner = np.array([[1.0, 1.0]])
    assert bool(boundary_junction_mask(b, corner)[0])
    face = np.array([[1.0, 0.0]])
    assert not bool(boundary_junction_mask(b, face)[0])


def test_normalize_adf_unit_gradient_near_boundary():
    s = Sphere(center=(0.0, 0.0), radius=1.0)
    X = np.array([[1.01, 0.0]])
    phi = approximate_distance(s, X, h=1e-5)
    g = omega_gradient(s, X, h=1e-5)
    assert float(np.linalg.norm(g[0])) == pytest.approx(1.0, abs=5e-2)
    assert phi[0] > 0


def test_normalize_adf_helper():
    omega = np.array([0.0, 3.0])
    grad = np.array([[1.0, 0.0], [0.0, 4.0]])
    phi = normalize_adf(omega, grad)
    assert phi[0] == pytest.approx(0.0)
    assert phi[1] == pytest.approx(3.0 / 5.0)


def test_boundary_factor_jet_sphere_first_order():
    from omnibias.core.multi_index import index_position

    s = Sphere(center=(0.0, 0.0), radius=1.0)
    x0 = np.array([0.3, 0.4])
    jet = boundary_factor_jet(s, x0, order=1, normalize=False)
    pos = index_position(2, 1)
    assert jet[pos[(0, 0)]] == pytest.approx(-0.5, abs=1e-12)
    g = omega_gradient(s, x0.reshape(1, -1))[0]
    assert jet[pos[(1, 0)]] == pytest.approx(g[0], abs=1e-12)
    assert jet[pos[(0, 1)]] == pytest.approx(g[1], abs=1e-12)


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


def test_solver_interior_sampling_respects_sdf():
    from omnibias.pinn.solver import CollocationSpec
    from omnibias.pinn.solver._core.sampling import interior_points

    s = Sphere(center=(0.0, 0.0), radius=1.0)
    dom = Domain(("x", "y"), [(-1.5, 1.5), (-1.5, 1.5)], sdf=s)
    pts = interior_points(dom, CollocationSpec(n_interior=24, method="random", seed=3))
    assert np.all(s(pts) < 0)


def test_solver_boundary_sampling_respects_sdf():
    from omnibias.pinn.solver import CollocationSpec
    from omnibias.pinn.solver._core.sampling import spatial_boundary_points

    s = Sphere(center=(0.0, 0.0), radius=1.0)
    dom = Domain(("x", "y"), [(-1.5, 1.5), (-1.5, 1.5)], sdf=s)
    pts = spatial_boundary_points(dom, CollocationSpec(n_boundary=12, seed=4))
    assert np.max(np.abs(s(pts))) < 1e-3
