# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""de Rham topology: Hodge Laplacian on k-forms, Betti numbers, degree, winding.

Oracles (all closed-form field derivatives + quadrature):

- Flat torus ``T^2`` Betti numbers ``(b0, b1, b2) = (1, 2, 1)`` from the nullity
  of the Hodge Laplacian on a Fourier form basis.
- Identity map ``S^2 -> S^2`` has degree 1; a circle winding ``phi = q theta``
  has winding number ``q``.
- Gauss-Bonnet ``chi(S^2) = 2``.
- Curved ``k>=1`` Hodge Laplacian is honestly a ``NotImplementedError``.
- torch vs jax bit-level parity.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.geometry import DifferentialForm, ManifoldSpec, MetricSpec
from omnibias.geometry.jax import ops as jgeo
from omnibias.geometry.torch import ops as tgeo

TWO_PI = 2.0 * math.pi


def _np(x):  # type: ignore[no-untyped-def]
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _flat_torus(builders, xp, stack):  # type: ignore[no-untyped-def]
    g = builders["flat_metric_factory"](xp, stack, 2)
    return ManifoldSpec("T2_flat", 2, MetricSpec(g, dim=2))


def _sphere(builders, xp, stack):  # type: ignore[no-untyped-def]
    g = builders["sphere_metric_factory"](xp, stack)
    return ManifoldSpec("S2", 2, MetricSpec(g, dim=2))


def _field(builders, ops_module, axes_map):  # type: ignore[no-untyped-def]
    cs = CoordinateSpec(("theta", "phi"), time_axis=None)
    return builders["AnalyticField"](
        cs, ComponentSpec(tuple(axes_map)), axes_map, ops_module,
    )


def _torus_form_bases(builders, xp):  # type: ignore[no-untyped-def]
    """Fourier component axes + 0/1/2-form bases for the flat torus."""
    P, C, S, K = (
        builders["Poly1D"], builders["Cos1D"], builders["Sin1D"], builders["Const1D"],
    )
    one = (K(), K())
    axes = {
        "one": one,
        "ct": (C(xp=xp), K()),
        "st": (S(xp=xp), K()),
        "cp": (K(), C(xp=xp)),
        "sp": (K(), S(xp=xp)),
    }
    _ = P  # (linear ramp available if needed)
    basis0 = [DifferentialForm(0, 2, {(): nm}) for nm in axes]
    basis1 = [
        DifferentialForm(1, 2, {(0,): "one"}),  # d(theta)  -- harmonic
        DifferentialForm(1, 2, {(1,): "one"}),  # d(phi)    -- harmonic
        DifferentialForm(1, 2, {(0,): "ct"}),
        DifferentialForm(1, 2, {(0,): "sp"}),
        DifferentialForm(1, 2, {(1,): "ct"}),
        DifferentialForm(1, 2, {(1,): "sp"}),
    ]
    basis2 = [
        DifferentialForm(2, 2, {(0, 1): "one"}),  # d(theta)^d(phi) -- harmonic
        DifferentialForm(2, 2, {(0, 1): "ct"}),
        DifferentialForm(2, 2, {(0, 1): "sp"}),
    ]
    return axes, basis0, basis1, basis2


# --------------------------------------------------------------------------- #
# Betti numbers of the flat 2-torus: (1, 2, 1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree_expected", [(0, 1), (1, 2), (2, 1)])
def test_torus_betti_numbers(builders, degree_expected):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch as td

    degree, expected = degree_expected
    axes, b0, b1, b2 = _torus_form_bases(builders, torch)
    basis = {0: b0, 1: b1, 2: b2}[degree]
    field = _field(builders, td, axes)
    rule = gauss_legendre([(0.0, TWO_PI), (0.0, TWO_PI)], 16)
    state = field(torch.as_tensor(rule.nodes, dtype=torch.float64))
    manifold = _flat_torus(builders, torch, torch.stack)
    mat = tgeo.hodge_laplacian_matrix(state, basis, manifold, rule=rule)
    assert tgeo.betti_number(mat) == expected


def test_harmonic_projection_selects_kernel(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch as td

    axes, b0, _b1, _b2 = _torus_form_bases(builders, torch)
    field = _field(builders, td, axes)
    rule = gauss_legendre([(0.0, TWO_PI), (0.0, TWO_PI)], 16)
    state = field(torch.as_tensor(rule.nodes, dtype=torch.float64))
    manifold = _flat_torus(builders, torch, torch.stack)
    mat = tgeo.hodge_laplacian_matrix(state, b0, manifold, rule=rule)
    # b0 basis: only the constant form "one" (index 0) is harmonic.
    coeffs = torch.ones(len(b0), dtype=torch.float64)
    proj = tgeo.harmonic_projection(mat, coeffs)
    # projection keeps the constant direction, kills the oscillatory ones.
    assert abs(float(proj[0])) > 0.5
    assert np.allclose(_np(proj[1:]), 0.0, atol=1e-8)


# --------------------------------------------------------------------------- #
# Degree of a map S^2 -> S^2 (identity) is 1
# --------------------------------------------------------------------------- #
def _sphere_embedding_axes(builders, xp):  # type: ignore[no-untyped-def]
    C, S, K = builders["Cos1D"], builders["Sin1D"], builders["Const1D"]
    return {
        "n0": (S(xp=xp), C(xp=xp)),  # sin(theta) cos(phi)
        "n1": (S(xp=xp), S(xp=xp)),  # sin(theta) sin(phi)
        "n2": (C(xp=xp), K()),       # cos(theta)
    }


def test_map_degree_identity_sphere(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch as td

    field = _field(builders, td, _sphere_embedding_axes(builders, torch))
    rule = gauss_legendre([(0.0, math.pi), (0.0, TWO_PI)], 24)
    state = field(torch.as_tensor(rule.nodes, dtype=torch.float64))
    deg = tgeo.map_degree(state, ("n0", "n1", "n2"), rule=rule)
    assert abs(float(deg) - 1.0) < 1e-6


@pytest.mark.parametrize("q", [1.0, 2.0, -3.0])
def test_winding_number(builders, q):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch as td

    cs = CoordinateSpec(("theta",), time_axis=None)
    field = builders["AnalyticField"](
        cs, ComponentSpec(("phi",)), {"phi": (builders["Poly1D"]((0.0, q)),)}, td,
    )
    rule = gauss_legendre([(0.0, TWO_PI)], 8)
    state = field(torch.as_tensor(rule.nodes, dtype=torch.float64))
    w = tgeo.winding_number(state, "phi", axis=0, rule=rule)
    assert abs(float(w) - q) < 1e-9


# --------------------------------------------------------------------------- #
# Gauss-Bonnet: chi(S^2) = 2
# --------------------------------------------------------------------------- #
def test_gauss_bonnet_sphere(builders):  # type: ignore[no-untyped-def]
    manifold = _sphere(builders, torch, torch.stack)
    rule = gauss_legendre([(1e-3, math.pi - 1e-3), (0.0, TWO_PI)], 32)
    coords = torch.as_tensor(rule.nodes, dtype=torch.float64)
    chi = tgeo.gauss_bonnet_euler(coords, manifold, rule=rule)
    assert abs(float(chi) - 2.0) < 1e-5


# --------------------------------------------------------------------------- #
# Hodge Laplacian: k=0 matches the scalar op; curved k>=1 is honest
# --------------------------------------------------------------------------- #
def test_hodge_laplacian_zero_form_matches_scalar(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch as td

    axes = {"f": (builders["Cos1D"](xp=torch), builders["Sin1D"](xp=torch))}
    field = _field(builders, td, axes)
    coords = torch.as_tensor(
        [[0.4, 0.7], [1.1, 2.0], [2.3, 4.1]], dtype=torch.float64,
    )
    state = field(coords)
    manifold = _sphere(builders, torch, torch.stack)
    lap = tgeo.hodge_laplacian(state, DifferentialForm(0, 2, {(): "f"}), manifold)
    scal = tgeo.hodge_laplacian_scalar(state, "f", manifold)
    assert np.allclose(_np(lap[()]), _np(scal), rtol=1e-10, atol=1e-10)


def test_hodge_laplacian_one_form_flat_componentwise(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch as td

    # On a flat metric, (Delta omega)_i = -sum_m d^2_m omega_i (componentwise).
    axes = {"a": (builders["Cos1D"](freq=1.0, xp=torch), builders["Sin1D"](freq=1.0, xp=torch))}
    field = _field(builders, td, axes)
    coords = torch.as_tensor([[0.3, 1.2], [1.7, 2.9]], dtype=torch.float64)
    state = field(coords)
    manifold = _flat_torus(builders, torch, torch.stack)
    form = DifferentialForm(1, 2, {(0,): "a"})
    lap = tgeo.hodge_laplacian(state, form, manifold)
    # a = cos(theta) sin(phi); -(d_thth + d_phph) a = -(-a - a) = 2a.
    a_val = torch.cos(coords[:, 0]) * torch.sin(coords[:, 1])
    assert np.allclose(_np(lap[(0,)]), _np(2.0 * a_val), rtol=1e-9, atol=1e-9)


def test_hodge_laplacian_curved_k_form_raises(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch as td

    axes = {"a": (builders["Cos1D"](xp=torch), builders["Const1D"]())}
    field = _field(builders, td, axes)
    coords = torch.as_tensor([[0.9, 1.0], [1.4, 2.2]], dtype=torch.float64)
    state = field(coords)
    manifold = _sphere(builders, torch, torch.stack)  # curved -> Christoffel != 0
    with pytest.raises(NotImplementedError):
        tgeo.hodge_laplacian(state, DifferentialForm(1, 2, {(0,): "a"}), manifold)


# --------------------------------------------------------------------------- #
# Cross-backend parity
# --------------------------------------------------------------------------- #
def test_degree_and_euler_cross_backend(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.jax import _ops_dispatch as jd
    from omnibias.fields.torch import _ops_dispatch as td

    # degree
    rule = gauss_legendre([(0.0, math.pi), (0.0, TWO_PI)], 20)
    tf = _field(builders, td, _sphere_embedding_axes(builders, torch))
    jf = _field(builders, jd, _sphere_embedding_axes(builders, jnp))
    ts = tf(torch.as_tensor(rule.nodes, dtype=torch.float64))
    js = jf(jnp.asarray(rule.nodes, dtype=jnp.float64))
    dt = tgeo.map_degree(ts, ("n0", "n1", "n2"), rule=rule)
    dj = jgeo.map_degree(js, ("n0", "n1", "n2"), rule=rule)
    assert np.allclose(_np(dt), _np(dj), rtol=1e-11, atol=1e-11)

    # Euler
    rule2 = gauss_legendre([(1e-3, math.pi - 1e-3), (0.0, TWO_PI)], 24)
    mt = _sphere(builders, torch, torch.stack)
    mj = _sphere(builders, jnp, jnp.stack)
    ct = torch.as_tensor(rule2.nodes, dtype=torch.float64)
    cj = jnp.asarray(rule2.nodes, dtype=jnp.float64)
    assert np.allclose(
        _np(tgeo.gauss_bonnet_euler(ct, mt, rule=rule2)),
        _np(jgeo.gauss_bonnet_euler(cj, mj, rule=rule2)),
        rtol=1e-10, atol=1e-10,
    )


def test_betti_cross_backend(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.jax import _ops_dispatch as jd
    from omnibias.fields.torch import _ops_dispatch as td

    axes_t, _b0, b1_t, _b2 = _torus_form_bases(builders, torch)
    axes_j, _b0j, b1_j, _b2j = _torus_form_bases(builders, jnp)
    rule = gauss_legendre([(0.0, TWO_PI), (0.0, TWO_PI)], 16)
    tf = _field(builders, td, axes_t)
    jf = _field(builders, jd, axes_j)
    ts = tf(torch.as_tensor(rule.nodes, dtype=torch.float64))
    js = jf(jnp.asarray(rule.nodes, dtype=jnp.float64))
    mt = tgeo.hodge_laplacian_matrix(ts, b1_t, _flat_torus(builders, torch, torch.stack), rule=rule)
    mj = jgeo.hodge_laplacian_matrix(js, b1_j, _flat_torus(builders, jnp, jnp.stack), rule=rule)
    assert np.allclose(_np(mt), _np(mj), rtol=1e-10, atol=1e-10)
    assert tgeo.betti_number(mt) == jgeo.betti_number(mj) == 2


# --------------------------------------------------------------------------- #
# Out-of-thesis enforcement: no combinatorial-topology overclaim
# --------------------------------------------------------------------------- #
def test_no_combinatorial_topology_claimed() -> None:
    forbidden = {
        "homotopy_group", "pi_n", "fundamental_group",
        "persistent_homology", "persistence_diagram",
        "simplicial_homology", "smith_normal_form", "betti_from_simplicial",
    }
    for surface in (tgeo, jgeo):
        names = set(getattr(surface, "__all__", dir(surface)))
        assert not (names & forbidden), f"unexpected combinatorial-topology op in {surface}"
