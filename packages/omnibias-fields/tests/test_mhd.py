# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Magnetohydrodynamics ops (WS3): Lorentz force, induction, ideal-MHD momentum.

Independent oracles:
1. analytic -- a fully independent *symbolic* (sympy) vector-calculus
   implementation of every residual (direct ``curl(u x B)``, direct ``(u.grad)u``)
   evaluated at random nodes equals the op output (which instead uses the
   ``curl(u x B)`` identity and the ``advection`` primitive);
2. an exact nonlinear solution -- a finite-amplitude Elsasser/Alfven wave
   ``u = 1/2(F+C)``, ``B = 1/2(F-C)`` with ``F=F(z-t)`` divergence-free drives
   both the induction and momentum residuals to zero (checked symbolically and
   numerically);
3. structural reductions -- ``B = 0`` collapses the MHD momentum onto
   Navier-Stokes; ``u = 0`` collapses induction onto resistive diffusion
   ``-eta lap B``; the Lorentz force recomposes ``(curl B) x B``;
4. torch vs jax cross-backend parity to ``rtol=1e-9`` (float64).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
import torch
from _analytic import Poly, make_field
from _sympy_field import axis_symbols, make_sympy_field

_U = ("u", "v", "w")
_B = ("Bx", "By", "Bz")
_AXES = ("x", "y", "z", "t")


def _np(x):  # type: ignore[no-untyped-def]
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _nodes(seed: int = 0, n: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.9, 0.9, size=(n, 4)).astype(np.float64)


# ======================================================================
# Symbolic vector-calculus oracle (independent of the op code path)
# ======================================================================

_SYM = axis_symbols(_AXES)
_x, _y, _z, _t = _SYM["x"], _SYM["y"], _SYM["z"], _SYM["t"]
_SP = (_x, _y, _z)

# Arbitrary smooth (coupled) fields -- the curl(uxB) identity holds for any field.
_EXPR = {
    "u": 0.4 * _x * _y - 0.3 * _z + 0.2 * _t * _z,
    "v": 0.1 * _x**2 - 0.2 * _y * _z + 0.15 * _t * _y,
    "w": 0.3 * _z * _x - 0.25 * _y + 0.1 * _t,
    "Bx": 0.2 * _x - 0.4 * _y * _z + 0.1 * _t * _x,
    "By": 0.3 * _y + 0.2 * _x * _z - 0.15 * _t,
    "Bz": -0.1 * _z + 0.25 * _x * _y + 0.2 * _t * _z,
    "p": 0.5 * _x * _y - 0.2 * _z**2 + 0.1 * _t * _x * _y,
}
_GROUPS = {"U": _U, "B": _B}


def _sdiv(vec):  # type: ignore[no-untyped-def]
    return sum(sp.diff(vec[i], _SP[i]) for i in range(3))


def _scurl(vec):  # type: ignore[no-untyped-def]
    return (
        sp.diff(vec[2], _y) - sp.diff(vec[1], _z),
        sp.diff(vec[0], _z) - sp.diff(vec[2], _x),
        sp.diff(vec[1], _x) - sp.diff(vec[0], _y),
    )


def _sgrad(scalar):  # type: ignore[no-untyped-def]
    return tuple(sp.diff(scalar, s) for s in _SP)


def _slapv(vec):  # type: ignore[no-untyped-def]
    return tuple(sum(sp.diff(vec[i], s, 2) for s in _SP) for i in range(3))


def _sadv(vel, target):  # type: ignore[no-untyped-def]
    return tuple(sum(vel[i] * sp.diff(tc, _SP[i]) for i in range(3)) for tc in target)


def _scross(a, b):  # type: ignore[no-untyped-def]
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _u_sym():  # type: ignore[no-untyped-def]
    return (_EXPR["u"], _EXPR["v"], _EXPR["w"])


def _b_sym():  # type: ignore[no-untyped-def]
    return (_EXPR["Bx"], _EXPR["By"], _EXPR["Bz"])


def _lambdify_vec(vec):  # type: ignore[no-untyped-def]
    return [sp.lambdify((_x, _y, _z, _t), c, modules="numpy") for c in vec]


def _eval_vec(vec, nodes):  # type: ignore[no-untyped-def]
    fns = _lambdify_vec(vec)
    cols = [nodes[:, i] for i in range(4)]
    return np.stack([np.broadcast_to(f(*cols), nodes.shape[0]) for f in fns], axis=-1)


def _sym_field(backend):  # type: ignore[no-untyped-def]
    return make_sympy_field(backend, _AXES, _EXPR, _SYM, groups=_GROUPS, time_axis="t")


def test_lorentz_force_matches_symbolic():
    nodes = _nodes(1)
    st = _sym_field("torch")(torch.as_tensor(nodes))
    got = _np(st.ops.lorentz_force(st, magnetic=_B))
    j = _scurl(_b_sym())
    exp = _eval_vec(_scross(j, _b_sym()), nodes)
    assert np.allclose(got, exp, rtol=1e-9, atol=1e-11)


def test_induction_matches_symbolic():
    nodes = _nodes(2)
    eta = 0.05
    st = _sym_field("torch")(torch.as_tensor(nodes))
    got = _np(st.ops.induction_residual(st, velocity=_U, magnetic=_B, resistivity=eta))
    u, b = _u_sym(), _b_sym()
    dtB = tuple(sp.diff(c, _t) for c in b)
    curl_uxb = _scurl(_scross(u, b))
    lapB = _slapv(b)
    res = tuple(dtB[i] - curl_uxb[i] - eta * lapB[i] for i in range(3))
    assert np.allclose(got, _eval_vec(res, nodes), rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("rho,nu", [(1.0, 0.0), (1.3, 0.07)])
def test_momentum_matches_symbolic(rho, nu):
    nodes = _nodes(3)
    st = _sym_field("torch")(torch.as_tensor(nodes))
    got = _np(
        st.ops.ideal_mhd_momentum_residual(
            st, velocity=_U, magnetic=_B, pressure="p", density=rho, viscosity=nu,
        )
    )
    u, b = _u_sym(), _b_sym()
    dtu = tuple(sp.diff(c, _t) for c in u)
    adv = _sadv(u, u)
    gradp = _sgrad(_EXPR["p"])
    jxb = _scross(_scurl(b), b)
    lapu = _slapv(u)
    res = tuple(
        rho * (dtu[i] + adv[i]) + gradp[i] - jxb[i] - nu * lapu[i] for i in range(3)
    )
    assert np.allclose(got, _eval_vec(res, nodes), rtol=1e-9, atol=1e-11)


def test_maxwell_stress_and_magnetic_pressure():
    nodes = _nodes(4)
    st = _sym_field("torch")(torch.as_tensor(nodes))
    t = _np(st.ops.maxwell_stress_tensor(st, magnetic=_B))
    b = _eval_vec(_b_sym(), nodes)
    b2 = (b * b).sum(-1)
    # symmetric, and trace = -1/2 |B|^2
    assert np.allclose(t, np.transpose(t, (0, 2, 1)), atol=1e-11)
    assert np.allclose(np.trace(t, axis1=1, axis2=2), -0.5 * b2, rtol=1e-9, atol=1e-11)
    pm = _np(st.ops.magnetic_pressure(st, magnetic=_B))
    assert np.allclose(pm, 0.5 * b2, rtol=1e-9, atol=1e-11)


# ======================================================================
# Exact nonlinear Elsasser / Alfven wave: residuals vanish
# ======================================================================


def _alfven_exprs():  # type: ignore[no-untyped-def]
    xi = _z - _t                       # wave phase, speed c = 1 = |C|
    fx, fy, fz = xi**2, xi**3, sp.Integer(0)
    cz = sp.Rational(1, 2)
    u = (fx / 2, fy / 2, (fz + 1) / 2)
    b = (fx / 2, fy / 2, (fz - 1) / 2)
    p = -(b[0] ** 2 + b[1] ** 2 + b[2] ** 2) / 2
    del cz
    return {
        "u": u[0], "v": u[1], "w": u[2],
        "Bx": b[0], "By": b[1], "Bz": b[2], "p": p,
    }


def test_alfven_wave_is_exact_symbolically():
    e = _alfven_exprs()
    u = (e["u"], e["v"], e["w"])
    b = (e["Bx"], e["By"], e["Bz"])
    # induction: d_t B - curl(u x B) == 0
    dtB = tuple(sp.diff(c, _t) for c in b)
    ind = tuple(sp.simplify(dtB[i] - _scurl(_scross(u, b))[i]) for i in range(3))
    assert all(c == 0 for c in ind)
    # momentum (rho=1, nu=0): d_t u + (u.grad)u + grad p - (curl B) x B == 0
    dtu = tuple(sp.diff(c, _t) for c in u)
    adv = _sadv(u, u)
    gradp = _sgrad(e["p"])
    jxb = _scross(_scurl(b), b)
    mom = tuple(sp.simplify(dtu[i] + adv[i] + gradp[i] - jxb[i]) for i in range(3))
    assert all(c == 0 for c in mom)


def test_alfven_wave_residuals_vanish_numerically():
    e = _alfven_exprs()
    nodes = _nodes(7)
    f = make_sympy_field("torch", _AXES, e, _SYM, groups=_GROUPS, time_axis="t")
    st = f(torch.as_tensor(nodes))
    ind = _np(st.ops.induction_residual(st, velocity=_U, magnetic=_B))
    mom = _np(
        st.ops.ideal_mhd_momentum_residual(
            st, velocity=_U, magnetic=_B, pressure="p", density=1.0, viscosity=0.0,
        )
    )
    assert np.allclose(ind, 0.0, atol=1e-9)
    assert np.allclose(mom, 0.0, atol=1e-9)


# ======================================================================
# Structural reductions on polynomial fields (genuine backend arithmetic)
# ======================================================================

_POLY = {
    "u": (Poly([0.1, 0.4]), Poly([1.0, -0.3]), Poly([1.0, 0.2]), Poly([1.0, 0.1])),
    "v": (Poly([1.0, 0.2]), Poly([0.2, 0.5]), Poly([1.0, -0.1]), Poly([1.0, 0.2])),
    "w": (Poly([1.0, -0.1]), Poly([1.0, 0.3]), Poly([0.3, 0.4]), Poly([1.0, 0.1])),
    "Bx": (Poly([0.3, 0.2]), Poly([1.0, 0.4]), Poly([1.0, -0.2]), Poly([1.0, 0.1])),
    "By": (Poly([1.0, -0.2]), Poly([0.1, 0.3]), Poly([1.0, 0.5]), Poly([1.0, 0.2])),
    "Bz": (Poly([1.0, 0.1]), Poly([1.0, -0.4]), Poly([0.2, 0.3]), Poly([1.0, -0.1])),
    "p": (Poly([0.5, 0.3]), Poly([1.0, 0.2]), Poly([1.0, -0.3]), Poly([1.0, 0.15])),
}
_ZERO = (Poly([0.0]), Poly([0.0]), Poly([0.0]), Poly([0.0]))


def _poly_field(backend, comps):  # type: ignore[no-untyped-def]
    return make_field(backend, _AXES, comps, groups=_GROUPS, time_axis="t")


def _coords(backend, nodes):  # type: ignore[no-untyped-def]
    return torch.as_tensor(nodes) if backend == "torch" else jnp.asarray(nodes)


def test_momentum_reduces_to_navier_stokes_when_B_zero():
    nodes = _nodes(5)
    comps = dict(_POLY)
    comps["Bx"], comps["By"], comps["Bz"] = _ZERO, _ZERO, _ZERO
    st = _poly_field("torch", comps)(_coords("torch", nodes))
    rho, nu = 1.2, 0.05
    got = _np(
        st.ops.ideal_mhd_momentum_residual(
            st, velocity=_U, magnetic=_B, pressure="p", density=rho, viscosity=nu,
        )
    )
    ns = _np(
        rho * (st.ops.vector_derivative(st, _U, axis="t", order=1)
               + st.ops.advection(st, velocity=_U))
        + st.ops.gradient(st, "p")
        - nu * st.ops.vector_laplacian(st, _U)
    )
    assert np.allclose(got, ns, rtol=1e-9, atol=1e-11)


def test_induction_reduces_to_resistive_diffusion_when_u_zero():
    nodes = _nodes(6)
    comps = dict(_POLY)
    comps["u"], comps["v"], comps["w"] = _ZERO, _ZERO, _ZERO
    eta = 0.3
    st = _poly_field("torch", comps)(_coords("torch", nodes))
    got = _np(st.ops.induction_residual(st, velocity=_U, magnetic=_B, resistivity=eta))
    diff = _np(
        st.ops.vector_derivative(st, _B, axis="t", order=1)
        - eta * st.ops.vector_laplacian(st, _B)
    )
    assert np.allclose(got, diff, rtol=1e-9, atol=1e-11)


def test_lorentz_force_recomposition_with_explicit_current():
    nodes = _nodes(8)
    comps = dict(_POLY)
    comps["Jx"] = (Poly([0.2, 0.3]), Poly([1.0, 0.1]), Poly([1.0, 0.2]), Poly([1.0, 0.0]))
    comps["Jy"] = (Poly([1.0, 0.1]), Poly([0.3, 0.2]), Poly([1.0, 0.1]), Poly([1.0, 0.0]))
    comps["Jz"] = (Poly([1.0, 0.2]), Poly([1.0, 0.3]), Poly([0.1, 0.2]), Poly([1.0, 0.0]))
    groups = dict(_GROUPS)
    groups["J"] = ("Jx", "Jy", "Jz")
    st = make_field("torch", _AXES, comps, groups=groups, time_axis="t")(
        _coords("torch", nodes)
    )
    got = _np(st.ops.lorentz_force(st, magnetic=_B, current=("Jx", "Jy", "Jz")))
    j = _np(st.ops.stack_components(st, ("Jx", "Jy", "Jz")))
    b = _np(st.ops.stack_components(st, _B))
    exp = np.cross(j, b)
    assert np.allclose(got, exp, rtol=1e-9, atol=1e-11)


def test_magnetic_divergence_is_divergence():
    nodes = _nodes(9)
    st = _poly_field("torch", _POLY)(_coords("torch", nodes))
    assert np.allclose(
        _np(st.ops.magnetic_divergence(st, magnetic=_B)),
        _np(st.ops.divergence(st, _B)),
        rtol=1e-12, atol=1e-12,
    )


# ======================================================================
# torch <-> jax bit-parity (float64)
# ======================================================================

_PARITY = [
    lambda o, s: o.lorentz_force(s, magnetic=_B),
    lambda o, s: o.current_density(s, magnetic=_B),
    lambda o, s: o.magnetic_pressure(s, magnetic=_B),
    lambda o, s: o.maxwell_stress_tensor(s, magnetic=_B),
    lambda o, s: o.magnetic_divergence(s, magnetic=_B),
    lambda o, s: o.induction_residual(s, velocity=_U, magnetic=_B, resistivity=0.05),
    lambda o, s: o.ideal_mhd_momentum_residual(
        s, velocity=_U, magnetic=_B, pressure="p", density=1.1, viscosity=0.03,
    ),
]


@pytest.mark.parametrize("opfn", _PARITY)
def test_cross_backend_parity(opfn):
    nodes = _nodes(11)
    ts = _poly_field("torch", _POLY)(_coords("torch", nodes))
    js = _poly_field("jax", _POLY)(_coords("jax", nodes))
    assert np.allclose(_np(opfn(ts.ops, ts)), _np(opfn(js.ops, js)), rtol=1e-9, atol=1e-11)
