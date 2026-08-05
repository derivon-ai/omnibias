# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Phase 2 continuum-mechanics / fluids ops.

Stream-function relations, Newtonian / linear-elastic stress, viscous
dissipation, and the closed-form Stokes / Navier-Cauchy momentum residuals.
Validated by analytic recomposition, tensor identities, DSL parity, and
torch<->jax bit-parity at ``rtol=atol=1e-12`` (float64 via the package conftest).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from _analytic import Poly, make_field

# --- field definitions -------------------------------------------------

_M2 = {
    "psi": (Poly([1, 2, 1, 0.5]), Poly([1, -1, 0.5, 1])),
    "u": (Poly([1, 2, 1]), Poly([1, -1, 0.5])),
    "v": (Poly([0, 1, 0.5]), Poly([1, 1, 1])),
    "p": (Poly([1, 0.5, 0]), Poly([1, 1, 0.5])),
    "fx": (Poly([0, 1, 0]), Poly([1, 0, 0])),
    "fy": (Poly([1, 0, 0]), Poly([0, 1, 0])),
}
_GM2 = {"vel": ("u", "v"), "disp": ("u", "v")}

_M3 = {
    "u": (Poly([1, 2, 1]), Poly([1, -1, 0.5]), Poly([1, 0, 1])),
    "v": (Poly([0, 1, 0]), Poly([1, 1, 1]), Poly([2, -1, 0])),
    "w": (Poly([1, 1, 0]), Poly([1, 0, 0.5]), Poly([1, 1, 1])),
    "p": (Poly([1, 0.5, 0]), Poly([1, 1, 0]), Poly([1, 0, 0.5])),
    "fx": (Poly([0, 1, 0]), Poly([1, 0, 0]), Poly([1, 0, 0])),
    "fy": (Poly([1, 0, 0]), Poly([0, 1, 0]), Poly([1, 0, 0])),
    "fz": (Poly([1, 0, 0]), Poly([1, 0, 0]), Poly([0, 1, 0])),
}
_GM3 = {"vel": ("u", "v", "w"), "disp": ("u", "v", "w")}

# stored 2x2 stress tensor for stress_divergence
_S2 = {
    "sxx": (Poly([1, 1, 0.5]), Poly([1, 0, 1])),
    "sxy": (Poly([0, 1, 1]), Poly([1, 1, 0])),
    "syx": (Poly([0, 1, 1]), Poly([1, 1, 0])),
    "syy": (Poly([1, 0, 1]), Poly([1, 1, 0.5])),
}


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
    if which == "2d":
        f = make_field(backend, ("x", "y"), _M2, groups=_GM2, time_axis=None)
    elif which == "3d":
        f = make_field(backend, ("x", "y", "z"), _M3, groups=_GM3, time_axis=None)
    elif which == "s2":
        f = make_field(backend, ("x", "y"), _S2, time_axis=None)
    else:  # pragma: no cover
        raise ValueError(which)
    return f(_coords(backend, nodes))


# ======================================================================
# Correctness (analytic / identity / recomposition)
# ======================================================================


def test_velocity_from_streamfunction_components_and_incompressible():
    nodes = _nodes(2)
    st = _state("torch", "2d", nodes)
    vel = _np(st.ops.velocity_from_streamfunction(st, "psi"))
    u = _np(st.ops.derivative(st, "psi", axis="y", order=1))
    v = -_np(st.ops.derivative(st, "psi", axis="x", order=1))
    assert np.allclose(vel[:, 0], u, rtol=1e-12, atol=1e-12)
    assert np.allclose(vel[:, 1], v, rtol=1e-12, atol=1e-12)
    # incompressible by construction: du/dx + dv/dy = psi_xy - psi_yx = 0
    div = _np(st.ops.mixed_partial(st, "psi", ("x", "y"), (1, 1))) - _np(
        st.ops.mixed_partial(st, "psi", ("y", "x"), (1, 1))
    )
    assert np.allclose(div, 0.0, atol=1e-12)


def test_vorticity_from_streamfunction_is_minus_laplacian():
    nodes = _nodes(2)
    st = _state("torch", "2d", nodes)
    got = _np(st.ops.vorticity_from_streamfunction(st, "psi"))
    lap = _np(st.ops.laplacian(st, "psi", axes=("x", "y")))
    assert np.allclose(got, -lap, rtol=1e-12, atol=1e-12)
    # omega = dv/dx - du/dy with (u,v) from psi -> -psi_xx - psi_yy = -lap psi
    vort = -_np(st.ops.mixed_partial(st, "psi", ("x", "x"), (2, 0))) - _np(
        st.ops.mixed_partial(st, "psi", ("y", "y"), (0, 2))
    )
    assert np.allclose(got, vort, rtol=1e-12, atol=1e-12)


def test_newtonian_stress_symmetry_and_pressure():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    names = ("u", "v", "w")
    S = _np(st.ops.strain_rate(st, names))
    sig = _np(st.ops.newtonian_stress(st, names, viscosity=0.7))
    assert np.allclose(sig, 2 * 0.7 * S, rtol=1e-12, atol=1e-12)
    assert np.allclose(sig, np.swapaxes(sig, -1, -2), rtol=1e-12, atol=1e-12)
    sig_p = _np(st.ops.newtonian_stress(st, names, viscosity=0.7, pressure="p"))
    p = _np(st.ops.value(st, "p"))
    eye = np.eye(3)
    assert np.allclose(sig_p, 2 * 0.7 * S - p[:, None, None] * eye, rtol=1e-12, atol=1e-12)


def test_linear_elastic_stress_matches_hooke():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    names = ("u", "v", "w")
    lam, mu = 1.3, 0.8
    sig = _np(st.ops.linear_elastic_stress(st, names, lam=lam, mu=mu))
    S = _np(st.ops.strain_rate(st, names))
    tr = _np(st.ops.divergence(st, names))
    eye = np.eye(3)
    exp = 2 * mu * S + lam * tr[:, None, None] * eye
    assert np.allclose(sig, exp, rtol=1e-12, atol=1e-12)
    assert np.allclose(sig, np.swapaxes(sig, -1, -2), rtol=1e-12, atol=1e-12)


def test_viscous_dissipation_is_nonneg_and_matches_double_dot():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    names = ("u", "v", "w")
    phi = _np(st.ops.viscous_dissipation(st, names, viscosity=0.9))
    S = st.ops.strain_rate(st, names)
    assert np.allclose(phi, 2 * 0.9 * _np(st.ops.tensor_double_dot(S, S)), rtol=1e-12, atol=1e-12)
    assert np.all(phi >= 0.0)


def test_stokes_residual_recomposition_with_body_force():
    nodes = _nodes(2)
    st = _state("torch", "2d", nodes)
    res = _np(
        st.ops.stokes_residual(
            st, velocity=("u", "v"), pressure="p", viscosity=0.5,
            body_force=("fx", "fy"),
        )
    )
    lap = _np(st.ops.vector_laplacian(st, ("u", "v")))
    gradp = _np(st.ops.gradient(st, "p", axes=("x", "y")))
    f = _np(st.ops.stack_components(st, ("fx", "fy")))
    assert np.allclose(res, 0.5 * lap - gradp + f, rtol=1e-12, atol=1e-12)


def test_navier_cauchy_residual_recomposition_and_stress_divergence_identity():
    """The Navier-Cauchy operator is the closed-form divergence of the Hooke stress."""
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    names = ("u", "v", "w")
    lam, mu = 1.1, 0.6
    res = _np(st.ops.navier_cauchy_residual(st, displacement=names, lam=lam, mu=mu))
    gd = _np(st.ops.gradient_of_divergence(st, names))
    vl = _np(st.ops.vector_laplacian(st, names))
    assert np.allclose(res, (lam + mu) * gd + mu * vl, rtol=1e-12, atol=1e-12)
    # div(sigma)_i = (lam+mu) d_i(div u) + mu lap(u_i): same operator, recomposed
    # via per-row mixed partials of the analytic displacement field.
    sa = ("x", "y", "z")
    for i, ai in enumerate(sa):
        row = np.zeros(nodes.shape[0])
        # lam d_i(div u) + mu d_jj u_i + mu d_ij u_j
        for n, aj in zip(names, sa, strict=True):
            row = row + lam * _np(st.ops.mixed_partial(st, n, (ai, aj), (1, 1)))
        row = row + mu * _np(st.ops.laplacian(st, names[i], axes=sa))
        for n, aj in zip(names, sa, strict=True):
            row = row + mu * _np(st.ops.mixed_partial(st, n, (ai, aj), (1, 1)))
        assert np.allclose(res[:, i], row, rtol=1e-12, atol=1e-12)


def test_stress_divergence_alias():
    nodes = _nodes(2)
    st = _state("torch", "s2", nodes)
    names = (("sxx", "sxy"), ("syx", "syy"))
    assert np.allclose(
        _np(st.ops.stress_divergence(st, names)),
        _np(st.ops.tensor_divergence(st, names)),
        rtol=1e-12, atol=1e-12,
    )


def test_dsl_views_match_dispatch():
    nodes = _nodes(3)
    st = _state("torch", "3d", nodes)
    names = ("u", "v", "w")
    assert np.allclose(
        _np(st.vel.newtonian_stress(viscosity=0.7)),
        _np(st.ops.newtonian_stress(st, names, viscosity=0.7)),
    )
    assert np.allclose(
        _np(st.vel.elastic_stress(lam=1.3, mu=0.8)),
        _np(st.ops.linear_elastic_stress(st, names, lam=1.3, mu=0.8)),
    )
    assert np.allclose(
        _np(st.vel.viscous_dissipation(viscosity=0.9)),
        _np(st.ops.viscous_dissipation(st, names, viscosity=0.9)),
    )
    assert np.allclose(
        _np(st.vel.navier_cauchy_residual(lam=1.1, mu=0.6)),
        _np(st.ops.navier_cauchy_residual(st, displacement=names, lam=1.1, mu=0.6)),
    )


# ======================================================================
# torch <-> jax bit-parity (rtol=atol=1e-12)
# ======================================================================

_PARITY_CASES = [
    ("2d", lambda o, s: o.velocity_from_streamfunction(s, "psi")),
    ("2d", lambda o, s: o.vorticity_from_streamfunction(s, "psi")),
    ("3d", lambda o, s: o.newtonian_stress(s, ("u", "v", "w"), viscosity=0.7, pressure="p")),
    ("3d", lambda o, s: o.linear_elastic_stress(s, ("u", "v", "w"), lam=1.3, mu=0.8)),
    ("3d", lambda o, s: o.viscous_dissipation(s, ("u", "v", "w"), viscosity=0.9)),
    (
        "2d",
        lambda o, s: o.stokes_residual(
            s, velocity=("u", "v"), pressure="p", viscosity=0.5, body_force=("fx", "fy")
        ),
    ),
    (
        "3d",
        lambda o, s: o.navier_cauchy_residual(
            s, displacement=("u", "v", "w"), lam=1.1, mu=0.6, body_force=("fx", "fy", "fz")
        ),
    ),
    ("s2", lambda o, s: o.stress_divergence(s, (("sxx", "sxy"), ("syx", "syy")))),
]


@pytest.mark.parametrize("which,opfn", _PARITY_CASES)
def test_cross_backend_bit_parity(which, opfn):
    ndim = 2 if which in ("2d", "s2") else 3
    nodes = _nodes(ndim, seed=11)
    ts = _state("torch", which, nodes)
    js = _state("jax", which, nodes)
    assert np.allclose(_np(opfn(ts.ops, ts)), _np(opfn(js.ops, js)), rtol=1e-12, atol=1e-12)
