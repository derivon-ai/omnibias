# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Phase 1 core vector-calculus / conservation / flux / wave ops.

Validated three ways (all float64): analytic / identity references, an
independent recomposition from lower-level ops, and torch<->jax bit-parity at
``rtol=atol=1e-12``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from _analytic import Poly, make_field

# float64 in jax is enabled by the package conftest (needed for 1e-12 parity).

# --- field definitions -------------------------------------------------

_P3 = {
    "u": (Poly([1, 2, 1]), Poly([1, -1, 0.5]), Poly([1, 0, 1])),
    "v": (Poly([0, 1, 0]), Poly([1, 1, 1]), Poly([2, -1, 0])),
    "w": (Poly([1, 1, 0]), Poly([1, 0, 0.5]), Poly([1, 1, 1])),
    "c": (Poly([1, 1, 1]), Poly([1, -1, 1]), Poly([1, 0.5, 0])),
    "D": (Poly([2, 1, 0]), Poly([1, 1, 0]), Poly([1, 0, 0.5])),
}
_GROUPS3 = {"vel": ("u", "v", "w")}

_P2 = {
    "u": (Poly([1, 2, 1]), Poly([1, -1, 0])),
    "v": (Poly([0, 1, 0]), Poly([1, 1, 1])),
}
_GROUPS2 = {"vel": ("u", "v")}

# spacetime: axes (x, y, t)
_PST = {
    "c": (Poly([1, 1, 1]), Poly([1, -1, 0.5]), Poly([1, 1, 0])),
    "u": (Poly([1, 1, 0]), Poly([1, 0, 0]), Poly([1, 0, 0])),
    "v": (Poly([1, 0, 0]), Poly([0, 1, 0]), Poly([1, 0, 0])),
    "rho": (Poly([1, 1, 0]), Poly([1, 1, 0]), Poly([1, 2, 1])),
    "fx": (Poly([0, 1, 0]), Poly([1, 0, 0]), Poly([1, 1, 0])),
    "fy": (Poly([1, 0, 0]), Poly([0, 1, 0]), Poly([1, 1, 0])),
}
_GROUPSST = {"vel": ("u", "v")}


def _np(x):  # type: ignore[no-untyped-def]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _nodes(ndim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(24, ndim)).astype(np.float64)


def _coords(backend: str, arr: np.ndarray):  # type: ignore[no-untyped-def]
    if backend == "torch":
        return torch.as_tensor(arr, dtype=torch.float64)
    return jnp.asarray(arr, dtype=jnp.float64)


def _state(backend: str, which: str, nodes: np.ndarray):  # type: ignore[no-untyped-def]
    if which == "3d":
        f = make_field(backend, ("x", "y", "z"), _P3, groups=_GROUPS3, time_axis=None)
    elif which == "2d":
        f = make_field(backend, ("x", "y"), _P2, groups=_GROUPS2, time_axis=None)
    elif which == "st":
        f = make_field(backend, ("x", "y", "t"), _PST, groups=_GROUPSST, time_axis="t")
    else:  # pragma: no cover
        raise ValueError(which)
    return f(_coords(backend, nodes))


# ======================================================================
# Correctness (analytic / identity / independent recomposition)
# ======================================================================


def test_gradient_of_divergence_matches_mixed_partials():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    got = _np(st.ops.gradient_of_divergence(st, ("u", "v", "w")))
    sa = ("x", "y", "z")
    for i, ai in enumerate(sa):
        exp = np.zeros(nodes.shape[0])
        for n, aj in zip(("u", "v", "w"), sa, strict=True):
            exp = exp + _np(st.ops.mixed_partial(st, n, (ai, aj), (1, 1)))
        assert np.allclose(got[:, i], exp, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("which", ["2d", "3d"])
def test_curl_of_curl_identity(which):
    """curl(curl u) == grad(div u) - vector_laplacian(u) in 2D and 3D."""
    ndim = 2 if which == "2d" else 3
    names = ("u", "v") if which == "2d" else ("u", "v", "w")
    nodes = _nodes(ndim)
    st = _state("torch", which, nodes)
    cc = _np(st.ops.curl_of_curl(st, names))
    gd = _np(st.ops.gradient_of_divergence(st, names))
    vl = _np(st.ops.vector_laplacian(st, names))
    assert np.allclose(cc, gd - vl, rtol=1e-12, atol=1e-12)


def test_rate_of_rotation_plus_strain_equals_jacobian():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    W = _np(st.ops.rate_of_rotation_tensor(st, ("u", "v", "w")))
    S = _np(st.ops.strain_rate(st, ("u", "v", "w")))
    J = _np(st.ops.deformation_gradient(st, ("u", "v", "w")))
    assert np.allclose(W + S, J, rtol=1e-12, atol=1e-12)
    # W is antisymmetric
    assert np.allclose(W, -np.swapaxes(W, -1, -2), rtol=1e-12, atol=1e-12)


def test_grad_squared_norm_matches_sum_of_squared_partials():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    got = _np(st.ops.grad_squared_norm(st, "c"))
    exp = np.zeros(nodes.shape[0])
    for a in ("x", "y", "z"):
        exp = exp + _np(st.ops.derivative(st, "c", axis=a, order=1)) ** 2
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-12)


def test_diffusive_flux_constant_and_field():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    grad = np.stack(
        [_np(st.ops.derivative(st, "c", axis=a, order=1)) for a in ("x", "y", "z")],
        axis=-1,
    )
    got_const = _np(st.ops.diffusive_flux(st, "c", diffusivity=0.7))
    assert np.allclose(got_const, -0.7 * grad, rtol=1e-12, atol=1e-12)
    got_field = _np(st.ops.diffusive_flux(st, "c", diffusivity="D"))
    dval = _np(st.ops.value(st, "D"))[:, None]
    assert np.allclose(got_field, -dval * grad, rtol=1e-12, atol=1e-12)


def test_variable_coefficient_diffusion_constant_and_field():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    lap = _np(st.ops.laplacian(st, "c", axes=("x", "y", "z")))
    got_const = _np(st.ops.variable_coefficient_diffusion(st, "c", diffusivity=2.5))
    assert np.allclose(got_const, 2.5 * lap, rtol=1e-12, atol=1e-12)
    # field D: div(D grad c) = grad(D).grad(c) + D lap(c)
    gd = np.zeros(nodes.shape[0])
    for a in ("x", "y", "z"):
        gd = gd + _np(st.ops.derivative(st, "D", axis=a, order=1)) * _np(
            st.ops.derivative(st, "c", axis=a, order=1)
        )
    exp = gd + _np(st.ops.value(st, "D")) * lap
    got_field = _np(st.ops.variable_coefficient_diffusion(st, "c", diffusivity="D"))
    assert np.allclose(got_field, exp, rtol=1e-12, atol=1e-12)


def test_dalembertian_matches_definition_and_signature_flip():
    nodes = _nodes(3)
    st = _state("torch", "st", nodes)
    lap = _np(st.ops.laplacian(st, "c", axes=("x", "y")))
    dtt = _np(st.ops.derivative(st, "c", axis="t", order=2))
    cw = 1.7
    got = _np(st.ops.dalembertian(st, "c", c=cw))
    assert np.allclose(got, lap - dtt / cw**2, rtol=1e-12, atol=1e-12)
    got_minus = _np(st.ops.dalembertian(st, "c", c=cw, signature="mostly_minus"))
    assert np.allclose(got_minus, -got, rtol=1e-12, atol=1e-12)


def test_conservation_residual_matches_definition():
    nodes = _nodes(3)
    st = _state("torch", "st", nodes)
    got = _np(st.ops.conservation_residual(st, density="rho", flux=("fx", "fy"), source=0.3))
    exp = (
        _np(st.ops.derivative(st, "rho", axis="t", order=1))
        + _np(st.ops.derivative(st, "fx", axis="x", order=1))
        + _np(st.ops.derivative(st, "fy", axis="y", order=1))
        - 0.3
    )
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-12)


def test_advection_diffusion_residual_matches_definition():
    nodes = _nodes(3)
    st = _state("torch", "st", nodes)
    got = _np(
        st.ops.advection_diffusion_residual(
            st, scalar="c", velocity=("u", "v"), diffusivity=0.5, source=0.2
        )
    )
    dt = _np(st.ops.derivative(st, "c", axis="t", order=1))
    adv = _np(st.ops.advection(st, velocity=("u", "v"), scalar="c"))
    diff = 0.5 * _np(st.ops.laplacian(st, "c", axes=("x", "y")))
    assert np.allclose(got, dt + adv - diff - 0.2, rtol=1e-12, atol=1e-12)


def test_skew_symmetric_advection_recomposition():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    got = _np(st.ops.skew_symmetric_advection(st, velocity=("u", "v", "w")))
    adv = _np(st.ops.advection(st, velocity=("u", "v", "w")))
    div = _np(st.ops.divergence(st, ("u", "v", "w")))[:, None]
    u_i = _np(st.ops.stack_components(st, ("u", "v", "w")))
    assert np.allclose(got, adv + 0.5 * div * u_i, rtol=1e-12, atol=1e-12)


def test_laplacian_of_composition_against_squared_field():
    """Delta(u^2) computed two ways: via the chain-rule op and via an
    independent separable field whose component *is* u^2."""
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    # build an independent field for c^2 (separable: prod p_d^2)
    sq_polys = {"csq": tuple(p.squared() for p in _P3["c"])}
    f_sq = make_field("torch", ("x", "y", "z"), sq_polys, time_axis=None)
    st_sq = f_sq(_coords("torch", nodes))
    exact = _np(st_sq.ops.laplacian(st_sq, "csq", axes=("x", "y", "z")))
    got = _np(
        st.ops.laplacian_of_composition(
            st, "c", lambda u: 2.0 * u, lambda u: 2.0 + 0.0 * u
        )
    )
    assert np.allclose(got, exact, rtol=1e-12, atol=1e-12)


def test_gradient_of_composition_against_squared_field():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    sq_polys = {"csq": tuple(p.squared() for p in _P3["c"])}
    f_sq = make_field("torch", ("x", "y", "z"), sq_polys, time_axis=None)
    st_sq = f_sq(_coords("torch", nodes))
    exact = _np(st_sq.ops.gradient(st_sq, "csq", axes=("x", "y", "z")))
    got = _np(st.ops.gradient_of_composition(st, "c", lambda u: 2.0 * u))
    assert np.allclose(got, exact, rtol=1e-12, atol=1e-12)


def test_tensor_double_dot_value_and_dissipation_sign():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    S = st.ops.strain_rate(st, ("u", "v", "w"))
    dd = _np(st.ops.tensor_double_dot(S, S))
    assert np.allclose(dd, (_np(S) ** 2).sum(axis=(-2, -1)), rtol=1e-12, atol=1e-12)
    assert np.all(dd >= 0.0)  # S:S is a sum of squares


def test_div_rot_aliases():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    assert np.allclose(
        _np(st.ops.div(st, ("u", "v", "w"))),
        _np(st.ops.divergence(st, ("u", "v", "w"))),
        rtol=1e-12, atol=1e-12,
    )
    assert np.allclose(
        _np(st.ops.rot(st, ("u", "v", "w"))),
        _np(st.ops.curl(st, ("u", "v", "w"))),
        rtol=1e-12, atol=1e-12,
    )


def test_dsl_views_match_dispatch():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    assert np.allclose(
        _np(st.vel.grad_div), _np(st.ops.gradient_of_divergence(st, ("u", "v", "w")))
    )
    assert np.allclose(
        _np(st.vel.curl_curl), _np(st.ops.curl_of_curl(st, ("u", "v", "w")))
    )
    assert np.allclose(
        _np(st.vel.rate_of_rotation),
        _np(st.ops.rate_of_rotation_tensor(st, ("u", "v", "w"))),
    )
    assert np.allclose(_np(st.c.grad_sq), _np(st.ops.grad_squared_norm(st, "c")))


def test_dsl_dt_dx_use_coordinate_spec_axis_names():
    """``.dt`` / ``.dx`` route through ``coordinate_spec``, not hardcoded ``t``/``x``."""
    axes = ("xi", "eta", "time")
    polys = {
        "u": (Poly((1.0, 1.0)), Poly((1.0, 0.0)), Poly((1.0,))),
        "v": (Poly((0.0, 1.0)), Poly((1.0, -1.0)), Poly((1.0,))),
        "c": (Poly((1.0, 2.0)), Poly((1.0, -1.0)), Poly((0.5, 0.0, 1.0))),
    }
    f = make_field("torch", axes, polys, time_axis="time")
    rng = np.random.default_rng(0)
    nodes = rng.standard_normal((5, 3))
    st = f(torch.as_tensor(nodes, dtype=torch.float64))
    assert np.allclose(
        _np(st.c.dt),
        _np(st.ops.derivative(st, "c", axis="time")),
        rtol=0.0,
        atol=0.0,
    )
    assert np.allclose(
        _np(st.c.dx),
        _np(st.ops.derivative(st, "c", axis="xi")),
        rtol=0.0,
        atol=0.0,
    )
    assert np.allclose(
        _np(st.c.dy),
        _np(st.ops.derivative(st, "c", axis="eta")),
        rtol=0.0,
        atol=0.0,
    )
    md = st.ops.material_derivative(st, velocity=("u", "v"), scalar="c")
    assert md.shape == (5,)


def test_dsl_hess_is_full_spacetime_while_hess_spatial_excludes_time():
    """``.hess`` spans all axes (incl. time); ``.hess_spatial`` excludes time.
    Documents/guards the DSL footgun that ``.grad``/``.lap`` are spatial-only
    but ``.hess`` is not."""
    nodes = _nodes(3)
    st = _state("torch", "st", nodes)  # axes (x, y, t), time_axis="t"
    hess = _np(st.rho.hess)            # full (B, 3, 3)
    hess_sp = _np(st.rho.hess_spatial)  # spatial (B, 2, 2)
    b = nodes.shape[0]
    assert hess.shape == (b, 3, 3)
    assert hess_sp.shape == (b, 2, 2)
    # The (x, y) block of the full Hessian is exactly the spatial Hessian.
    np.testing.assert_allclose(hess[:, :2, :2], hess_sp, rtol=1e-12, atol=1e-12)
    # rho has a degree-2 time factor, so d^2/dt^2 is nonzero -> the full
    # Hessian carries time information that hess_spatial drops.
    assert np.max(np.abs(hess[:, 2, 2])) > 1e-6


def test_gradient_hessian_laplacian_match_finite_difference_oracle():
    """grad / Hessian / Laplacian ops vs a coordinate finite-difference oracle.

    The closed-form ops walk the analytic per-axis *derivative* path; the oracle
    re-evaluates only the field *value* at perturbed coordinates and central-
    differences it. The two share no derivative code, so agreement (to the
    O(h^2) truncation floor of the stencil) is an independent numeric witness
    that grad/Hessian/Laplacian compute what their names claim.
    """
    nodes = _nodes(3, seed=11)
    field = make_field("torch", ("x", "y", "z"), _P3, groups=_GROUPS3, time_axis=None)
    st = field(_coords("torch", nodes))
    g = _np(st.ops.gradient(st, "c", axes=("x", "y", "z")))
    hess = _np(st.ops.hessian(st, "c", axes=("x", "y", "z")))
    lap = _np(st.ops.laplacian(st, "c", axes=("x", "y", "z")))

    def value(arr):  # type: ignore[no-untyped-def]
        s = field(_coords("torch", arr))
        return _np(s.ops.value(s, "c"))

    h = 1e-4
    ndim = 3
    n = nodes.shape[0]
    g_fd = np.zeros((n, ndim))
    hess_fd = np.zeros((n, ndim, ndim))
    f0 = value(nodes)
    for i in range(ndim):
        ei = np.zeros(ndim)
        ei[i] = h
        fp = value(nodes + ei)
        fm = value(nodes - ei)
        g_fd[:, i] = (fp - fm) / (2.0 * h)
        hess_fd[:, i, i] = (fp - 2.0 * f0 + fm) / h**2
        for j in range(i + 1, ndim):
            ej = np.zeros(ndim)
            ej[j] = h
            mixed = (
                value(nodes + ei + ej)
                - value(nodes + ei - ej)
                - value(nodes - ei + ej)
                + value(nodes - ei - ej)
            ) / (4.0 * h**2)
            hess_fd[:, i, j] = mixed
            hess_fd[:, j, i] = mixed

    np.testing.assert_allclose(g, g_fd, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hess, hess_fd, rtol=1e-4, atol=1e-4)
    # Laplacian is the Hessian trace; check it against the independent stencil.
    np.testing.assert_allclose(lap, np.einsum("bii->b", hess_fd), rtol=1e-4, atol=1e-4)


# ======================================================================
# torch <-> jax bit-parity (rtol=atol=1e-12)
# ======================================================================

_PARITY_CASES = [
    ("3d", lambda o, s: o.gradient_of_divergence(s, ("u", "v", "w"))),
    ("3d", lambda o, s: o.curl_of_curl(s, ("u", "v", "w"))),
    ("2d", lambda o, s: o.curl_of_curl(s, ("u", "v"))),
    ("3d", lambda o, s: o.rate_of_rotation_tensor(s, ("u", "v", "w"))),
    ("3d", lambda o, s: o.grad_squared_norm(s, "c")),
    ("3d", lambda o, s: o.diffusive_flux(s, "c", diffusivity="D")),
    ("3d", lambda o, s: o.variable_coefficient_diffusion(s, "c", diffusivity="D")),
    ("3d", lambda o, s: o.skew_symmetric_advection(s, velocity=("u", "v", "w"))),
    ("3d", lambda o, s: o.gradient_of_composition(s, "c", lambda u: 2.0 * u)),
    (
        "3d",
        lambda o, s: o.laplacian_of_composition(
            s, "c", lambda u: 2.0 * u, lambda u: 2.0 + 0.0 * u
        ),
    ),
    ("st", lambda o, s: o.dalembertian(s, "c", c=1.7)),
    ("st", lambda o, s: o.conservation_residual(s, density="rho", flux=("fx", "fy"))),
    (
        "st",
        lambda o, s: o.advection_diffusion_residual(
            s, scalar="c", velocity=("u", "v"), diffusivity=0.5
        ),
    ),
]


@pytest.mark.parametrize("which,opfn", _PARITY_CASES)
def test_cross_backend_bit_parity(which, opfn):
    nodes = _nodes(2 if which == "2d" else 3, seed=7)
    ts = _state("torch", which, nodes)
    js = _state("jax", which, nodes)
    t = _np(opfn(ts.ops, ts))
    j = _np(opfn(js.ops, js))
    assert np.allclose(t, j, rtol=1e-12, atol=1e-12)
