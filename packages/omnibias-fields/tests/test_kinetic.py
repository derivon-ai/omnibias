# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Kinetic-theory ops (WS3): Vlasov transport, BGK relaxation, moments.

Independent oracles:
1. exact solution -- force-free transport ``f = g(x - v t)`` has identically zero
   Vlasov residual (1D-1V and 2D-2V);
2. analytic (sympy) -- with a supplied force field the op residual equals a fully
   independent symbolic ``d_t f + v.grad_x f + (F/m).grad_v f``;
3. analytic moments -- the velocity moments of a closed-form Maxwellian recover
   ``n``, ``n u`` and ``n(1/2|u|^2 + d T / 2m)`` via Gauss-Legendre quadrature;
   BGK relaxation is moment-linear (mass/momentum conservation structure);
4. honesty enforcement -- the full Boltzmann collision integral is flagged
   numerical and no closed-form Boltzmann-collision op is exported;
5. torch vs jax cross-backend parity to ``rtol=1e-9`` (float64).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
import torch
from _analytic import Poly, make_field
from _sympy_field import axis_symbols, make_sympy_field
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.fields.torch.ops import kinetic as torch_kinetic

torch.set_default_dtype(torch.float64)


def _np(x):  # type: ignore[no-untyped-def]
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _coords(backend, nodes):  # type: ignore[no-untyped-def]
    return torch.as_tensor(nodes) if backend == "torch" else jnp.asarray(nodes)


# ======================================================================
# 1. Force-free transport: f = g(x - v t) has zero Vlasov residual
# ======================================================================


def test_free_streaming_1d1v_is_exact():
    axes = ("x", "vx", "t")
    syms = axis_symbols(axes)
    xi = syms["x"] - syms["vx"] * syms["t"]
    exprs = {"f": xi**3 - 0.5 * xi + sp.exp(-(xi**2))}
    rng = np.random.default_rng(0)
    nodes = rng.uniform(-0.8, 0.8, size=(20, 3))
    st = make_sympy_field("torch", axes, exprs, syms, time_axis="t")(torch.as_tensor(nodes))
    res = _np(
        st.ops.vlasov_residual(st, "f", position_axes=("x",), velocity_axes=("vx",))
    )
    assert np.allclose(res, 0.0, atol=1e-8)


def test_free_streaming_2d2v_is_exact():
    axes = ("x", "y", "vx", "vy", "t")
    syms = axis_symbols(axes)
    xix = syms["x"] - syms["vx"] * syms["t"]
    xiy = syms["y"] - syms["vy"] * syms["t"]
    exprs = {"f": sp.exp(-(xix**2) - xiy**2) * (1 + 0.3 * xix - 0.2 * xiy)}
    rng = np.random.default_rng(1)
    nodes = rng.uniform(-0.7, 0.7, size=(24, 5))
    st = make_sympy_field("torch", axes, exprs, syms, time_axis="t")(torch.as_tensor(nodes))
    res = _np(
        st.ops.vlasov_residual(
            st, "f", position_axes=("x", "y"), velocity_axes=("vx", "vy"),
        )
    )
    assert np.allclose(res, 0.0, atol=1e-8)


# ======================================================================
# 2. Vlasov with a force field vs an independent symbolic residual
# ======================================================================


def test_vlasov_with_force_matches_symbolic():
    axes = ("x", "vx", "t")
    syms = axis_symbols(axes)
    x, vx, t = syms["x"], syms["vx"], syms["t"]
    exprs = {
        "f": (x - vx * t) ** 2 + 0.5 * x * vx - 0.3 * t + sp.sin(0.4 * x),
        "Fx": -0.7 * x + 0.2 * vx,
    }
    mass = 1.3
    rng = np.random.default_rng(2)
    nodes = rng.uniform(-0.8, 0.8, size=(20, 3))
    st = make_sympy_field("torch", axes, exprs, syms, time_axis="t")(torch.as_tensor(nodes))
    got = _np(
        st.ops.vlasov_residual(
            st, "f", position_axes=("x",), velocity_axes=("vx",),
            force=("Fx",), mass=mass,
        )
    )
    f = exprs["f"]
    res = sp.diff(f, t) + vx * sp.diff(f, x) + (exprs["Fx"] / mass) * sp.diff(f, vx)
    fn = sp.lambdify((x, vx, t), res, modules="numpy")
    exp = fn(nodes[:, 0], nodes[:, 1], nodes[:, 2])
    assert np.allclose(got, exp, rtol=1e-9, atol=1e-10)


# ======================================================================
# 3. BGK relaxation recomposition
# ======================================================================


def test_bgk_collision_recomposition():
    axes = ("vx",)
    syms = axis_symbols(axes)
    exprs = {"f": 1.0 + 0.5 * syms["vx"] ** 2, "feq": 0.8 + 0.1 * syms["vx"] ** 2}
    rng = np.random.default_rng(3)
    nodes = rng.uniform(-1.0, 1.0, size=(15, 1))
    st = make_sympy_field("torch", axes, exprs, syms, time_axis=None)(torch.as_tensor(nodes))
    tau = 0.4
    got = _np(st.ops.bgk_collision(st, "f", equilibrium="feq", tau=tau))
    f = _np(st.ops.value(st, "f"))
    feq = _np(st.ops.value(st, "feq"))
    assert np.allclose(got, -(f - feq) / tau, rtol=1e-12, atol=1e-12)


def test_bgk_vlasov_residual_is_transport_minus_collision():
    axes = ("x", "vx", "t")
    syms = axis_symbols(axes)
    x, vx, t = syms["x"], syms["vx"], syms["t"]
    exprs = {"f": (x - vx * t) ** 2 + 0.3 * x, "feq": 0.5 + 0.1 * x**2}
    rng = np.random.default_rng(4)
    nodes = rng.uniform(-0.8, 0.8, size=(18, 3))
    st = make_sympy_field("torch", axes, exprs, syms, time_axis="t")(torch.as_tensor(nodes))
    tau = 0.6
    got = _np(
        st.ops.bgk_vlasov_residual(
            st, "f", position_axes=("x",), velocity_axes=("vx",),
            equilibrium="feq", tau=tau,
        )
    )
    lhs = _np(st.ops.vlasov_residual(st, "f", position_axes=("x",), velocity_axes=("vx",)))
    coll = _np(st.ops.bgk_collision(st, "f", equilibrium="feq", tau=tau))
    assert np.allclose(got, lhs - coll, rtol=1e-12, atol=1e-12)


# ======================================================================
# 4. Maxwellian op and its analytic velocity moments
# ======================================================================


def _maxwellian_np(v, *, n, u, temp, mass):  # type: ignore[no-untyped-def]
    d = v.shape[-1]
    dv = v - np.asarray(u)
    pref = n * (mass / (2.0 * math.pi * temp)) ** (d / 2.0)
    return pref * np.exp(-mass * (dv * dv).sum(-1) / (2.0 * temp))


def test_maxwellian_op_matches_formula():
    axes = ("vx", "vy")
    rng = np.random.default_rng(5)
    nodes = rng.uniform(-2.0, 2.0, size=(30, 2))
    # any field will do -- maxwellian reads the coordinate columns, not the value
    comps = {"f": (Poly([1.0]), Poly([1.0]))}
    st = make_field("torch", axes, comps, time_axis=None)(torch.as_tensor(nodes))
    got = _np(
        st.ops.maxwellian(
            st, velocity_axes=axes, density=1.3, bulk_velocity=(0.2, -0.1),
            temperature=1.1, mass=0.9,
        )
    )
    exp = _maxwellian_np(nodes, n=1.3, u=(0.2, -0.1), temp=1.1, mass=0.9)
    assert np.allclose(got, exp, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("d", [1, 2])
def test_maxwellian_moments_recover_fluid_variables(d):
    n, temp, mass = 1.4, 1.2, 0.8
    u = (0.3, -0.2)[:d]
    vaxes = ("vx", "vy")[:d]
    syms = axis_symbols(vaxes)
    dv_sq = sum((syms[a] - u[i]) ** 2 for i, a in enumerate(vaxes))
    pref = n * (mass / (2 * sp.pi * temp)) ** sp.Rational(d, 2)
    expr = pref * sp.exp(-mass * dv_sq / (2 * temp))
    rule = gauss_legendre([(-10.0, 10.0)] * d, 48)
    field = make_sympy_field("torch", vaxes, {"f": expr}, syms, time_axis=None)
    st = field(torch.as_tensor(rule.nodes))
    nd = float(_np(st.ops.number_density(st, "f", rule=rule)))
    md = _np(st.ops.momentum_density(st, "f", rule=rule, velocity_axes=vaxes))
    ke = float(_np(st.ops.kinetic_energy_density(st, "f", rule=rule, velocity_axes=vaxes)))
    assert nd == pytest.approx(n, rel=1e-6)
    assert np.allclose(md, n * np.asarray(u), rtol=1e-6, atol=1e-7)
    exp_ke = n * (0.5 * float(np.dot(u, u)) + d * temp / (2.0 * mass))
    assert ke == pytest.approx(exp_ke, rel=1e-6)


def test_bgk_is_moment_linear_mass_and_momentum():
    """BGK conserves the moments the equilibrium is built to match (linearity)."""
    vaxes = ("vx",)
    syms = axis_symbols(vaxes)
    vx = syms["vx"]
    ma = 1.2 * sp.exp(-((vx - 0.3) ** 2) / (2 * 1.1)) / sp.sqrt(2 * sp.pi * 1.1)
    mb = 0.9 * sp.exp(-((vx - 0.1) ** 2) / (2 * 1.0)) / sp.sqrt(2 * sp.pi * 1.0)
    rule = gauss_legendre([(-10.0, 10.0)], 48)
    st = make_sympy_field("torch", vaxes, {"fa": ma, "fb": mb}, syms, time_axis=None)(
        torch.as_tensor(rule.nodes)
    )
    tau = 0.5
    coll = st.ops.bgk_collision(st, "fa", equilibrium="fb", tau=tau)
    w = torch.as_tensor(rule.weights)
    v = torch.as_tensor(rule.nodes[:, 0])
    int_c = float(_np((w * coll).sum()))
    int_vc = float(_np((w * v * coll).sum()))
    nd_a = float(_np(st.ops.number_density(st, "fa", rule=rule)))
    nd_b = float(_np(st.ops.number_density(st, "fb", rule=rule)))
    mom_a = float(_np(st.ops.momentum_density(st, "fa", rule=rule, velocity_axes=vaxes))[0])
    mom_b = float(_np(st.ops.momentum_density(st, "fb", rule=rule, velocity_axes=vaxes))[0])
    assert int_c == pytest.approx(-(nd_a - nd_b) / tau, rel=1e-9, abs=1e-10)
    assert int_vc == pytest.approx(-(mom_a - mom_b) / tau, rel=1e-9, abs=1e-10)


# ======================================================================
# 5. Honesty enforcement: Boltzmann collision integral stays numerical
# ======================================================================


def test_boltzmann_collision_is_labelled_numerical():
    assert torch_kinetic.BOLTZMANN_COLLISION_IS_NUMERICAL is True
    # No closed-form Boltzmann-collision op may be exported from the module.
    offenders = [
        n for n in torch_kinetic.__all__
        if "boltzmann" in n.lower() and callable(getattr(torch_kinetic, n, None))
    ]
    assert offenders == []


# ======================================================================
# 6. torch <-> jax cross-backend parity (float64)
# ======================================================================

_PHASE_AXES = ("x", "vx", "t")
_PHASE = {
    "f": (Poly([0.5, 0.3]), Poly([1.0, -0.2, 0.1]), Poly([1.0, 0.2])),
    "Fx": (Poly([0.1, -0.4]), Poly([1.0, 0.1]), Poly([1.0, 0.0])),
}


def _phase_field(backend):  # type: ignore[no-untyped-def]
    return make_field(backend, _PHASE_AXES, _PHASE, time_axis="t")


def test_parity_vlasov_and_bgk():
    rng = np.random.default_rng(6)
    nodes = rng.uniform(-0.9, 0.9, size=(16, 3))
    ts = _phase_field("torch")(_coords("torch", nodes))
    js = _phase_field("jax")(_coords("jax", nodes))

    def vlasov(o, s):  # type: ignore[no-untyped-def]
        return o.vlasov_residual(
            s, "f", position_axes=("x",), velocity_axes=("vx",), force=("Fx",), mass=1.1,
        )

    def maxw(o, s):  # type: ignore[no-untyped-def]
        return o.maxwellian(s, velocity_axes=("vx",), density=1.2, temperature=1.3)

    assert np.allclose(_np(vlasov(ts.ops, ts)), _np(vlasov(js.ops, js)), rtol=1e-9, atol=1e-11)
    assert np.allclose(_np(maxw(ts.ops, ts)), _np(maxw(js.ops, js)), rtol=1e-9, atol=1e-11)


def test_parity_moments():
    rule = gauss_legendre([(-6.0, 6.0)], 24)
    comps = {"f": (Poly([1.0, 0.0, -0.05]),)}
    ts = make_field("torch", ("vx",), comps, time_axis=None)(_coords("torch", rule.nodes))
    js = make_field("jax", ("vx",), comps, time_axis=None)(_coords("jax", rule.nodes))
    for op in ("number_density",):
        gt = _np(getattr(ts.ops, op)(ts, "f", rule=rule))
        gj = _np(getattr(js.ops, op)(js, "f", rule=rule))
        assert np.allclose(gt, gj, rtol=1e-9, atol=1e-11)
    gt = _np(ts.ops.momentum_density(ts, "f", rule=rule, velocity_axes=("vx",)))
    gj = _np(js.ops.momentum_density(js, "f", rule=rule, velocity_axes=("vx",)))
    assert np.allclose(gt, gj, rtol=1e-9, atol=1e-11)
