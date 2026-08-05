# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exterior calculus: d^2 = 0, Hodge star involution, de Rham reduction.

The canonical identities are validated with closed-form field derivatives plus
the metric, and torch vs jax parity.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.geometry import DifferentialForm, ManifoldSpec, MetricSpec
from omnibias.geometry._core.forms import wedge
from omnibias.geometry.jax import ops as jgeo
from omnibias.geometry.torch import ops as tgeo

R = 1.3
COORDS = np.array([[0.7, 0.3], [1.1, 1.5], [1.9, 2.2], [2.4, 4.0]], dtype=np.float64)


def _np(x):  # type: ignore[no-untyped-def]
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _torch_sphere(builders):  # type: ignore[no-untyped-def]
    g = builders["sphere_metric_factory"](torch, torch.stack)
    return ManifoldSpec("S2", 2, MetricSpec(g, dim=2))


def _jax_sphere(builders):  # type: ignore[no-untyped-def]
    g = builders["sphere_metric_factory"](jnp, jnp.stack)
    return ManifoldSpec("S2", 2, MetricSpec(g, dim=2))


def _poly_field(builders, ops_module):  # type: ignore[no-untyped-def]
    P = builders["Poly1D"]
    comp_axes = {"f": (P((0.2, 1.0, -0.5, 0.3)), P((1.0, 0.4, -0.7)))}
    return builders["AnalyticField"](
        builders["coord_spec_2d"](), builders["ComponentSpec"](("f",)),
        comp_axes, ops_module,
    )


def test_d_squared_is_zero(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch

    state = _poly_field(builders, _ops_dispatch)(torch.as_tensor(COORDS, dtype=torch.float64))
    dd = tgeo.d_squared_scalar(state, "f")
    for _, v in dd.items():
        assert np.allclose(_np(v), 0.0, atol=1e-10)


def test_de_rham_reduces_to_laplace_beltrami(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch

    state = _poly_field(builders, _ops_dispatch)(torch.as_tensor(COORDS, dtype=torch.float64))
    m = _torch_sphere(builders)
    # delta d f == - Laplace-Beltrami f, via two independent code paths.
    hodge = _np(tgeo.codifferential_exact_scalar(state, "f", m))
    lb = _np(tgeo.laplace_beltrami(state, "f", m))
    assert np.allclose(hodge, -lb, rtol=1e-8, atol=1e-8)


def test_hodge_star_involution(builders):  # type: ignore[no-untyped-def]
    # On the 2-sphere (d=2), for a 1-form: ** = (-1)^{k(d-k)} = (-1)^1 = -1.
    m = _torch_sphere(builders)
    coords = torch.as_tensor(COORDS, dtype=torch.float64)
    B = COORDS.shape[0]
    alpha = {(0,): torch.linspace(0.5, 1.5, B, dtype=torch.float64),
             (1,): torch.linspace(-1.0, 1.0, B, dtype=torch.float64)}
    star = tgeo.hodge_star(alpha, 1, coords, m)
    star2 = tgeo.hodge_star(star, 1, coords, m)  # d - k = 1 again
    for idx in alpha:
        assert np.allclose(_np(star2[idx]), -_np(alpha[idx]), rtol=1e-9, atol=1e-9)


def test_hodge_star_zero_form(builders):  # type: ignore[no-untyped-def]
    # *1 = sqrt|g| dx^1 ^ dx^2 ; *(*1) should give back 1 (d=2, k=0: (-1)^0=1).
    m = _torch_sphere(builders)
    coords = torch.as_tensor(COORDS, dtype=torch.float64)
    B = COORDS.shape[0]
    f = {(): torch.ones(B, dtype=torch.float64) * 2.0}
    star = tgeo.hodge_star(f, 0, coords, m)          # 2-form
    back = tgeo.hodge_star(star, 2, coords, m)        # 0-form
    assert np.allclose(_np(back[()]), 2.0, rtol=1e-9, atol=1e-9)


def test_wedge_anticommutative() -> None:
    # 1-form ^ 1-form in d=3: a ^ b = - b ^ a.
    B = 5
    rng = np.random.default_rng(1)
    a = {(i,): torch.as_tensor(rng.normal(size=B)) for i in range(3)}
    b = {(i,): torch.as_tensor(rng.normal(size=B)) for i in range(3)}
    ab = wedge(a, 1, b, 1, 3)
    ba = wedge(b, 1, a, 1, 3)
    for idx in ab:
        assert torch.allclose(ab[idx], -ba[idx], atol=1e-12)


def test_exterior_derivative_cross_backend(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.jax import _ops_dispatch as jd
    from omnibias.fields.torch import _ops_dispatch as td

    ts = _poly_field(builders, td)(torch.as_tensor(COORDS, dtype=torch.float64))
    js = _poly_field(builders, jd)(jnp.asarray(COORDS, dtype=jnp.float64))
    form = DifferentialForm(0, 2, {(): "f"})
    dt = tgeo.exterior_derivative(ts, form)
    dj = jgeo.exterior_derivative(js, form)
    for idx in dt:
        assert np.allclose(_np(dt[idx]), _np(dj[idx]), rtol=1e-11, atol=1e-11)


def test_hodge_laplacian_cross_backend(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.jax import _ops_dispatch as jd
    from omnibias.fields.torch import _ops_dispatch as td

    ts = _poly_field(builders, td)(torch.as_tensor(COORDS, dtype=torch.float64))
    js = _poly_field(builders, jd)(jnp.asarray(COORDS, dtype=jnp.float64))
    t = _np(tgeo.hodge_laplacian_scalar(ts, "f", _torch_sphere(builders)))
    j = _np(jgeo.hodge_laplacian_scalar(js, "f", _jax_sphere(builders)))
    assert np.allclose(t, j, rtol=1e-8, atol=1e-8)
