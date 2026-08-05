# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form field fractional partial + fractional-diffusion residual.

These lift the analytic (jet) fractional operator onto a ``FieldState``. A tiny
polynomial field double (``P(x) * R(t)`` with an exact per-axis derivative tower)
makes the closed-form path exact, so the field op can be checked against the
hand-computed power law, the standalone jet op, the ordinary derivative (integer
order), the jax twin, and a method-of-manufactured-solutions residual. float64
throughout (jax x64 enabled in ``conftest``).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState
from omnibias.fractional.jax import field as jfield
from omnibias.fractional.torch import field as tfield
from omnibias.fractional.torch.ops import analytic as ta

F = torch.float64


def _poly_eval(coeffs, x, xp, deriv: int = 0):
    """Evaluate the ``deriv``-th derivative of ``sum_k coeffs[k] x**k`` at ``x``."""
    out = xp.zeros_like(x)
    for k in range(len(coeffs)):
        if k < deriv:
            continue
        fac = 1.0
        for j in range(deriv):
            fac *= k - j
        out = out + coeffs[k] * fac * x ** (k - deriv)
    return out


class _PolyField:
    """``u(x, t) = P(x) * R(t)`` with an exact per-axis tower (spectral dispatch).

    ``rt=None`` makes a steady field with a single spatial axis and no time axis.
    """

    _omnibias_dispatch = "spectral"

    def __init__(self, xp, ops_module, px, rt=None):  # type: ignore[no-untyped-def]
        self.xp = xp
        self._ops = ops_module
        self.px = px
        self.rt = rt
        if rt is None:
            self.coordinate_spec = CoordinateSpec(("x",), time_axis=None)
        else:
            self.coordinate_spec = CoordinateSpec(("x", "t"), time_axis="t")
        self.components = ComponentSpec(("u",))

    def evaluate(self, coords):  # type: ignore[no-untyped-def]
        return FieldState(
            coords=coords, field=self, components=self.components,
            coordinate_spec=self.coordinate_spec, ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    __call__ = evaluate

    def value_component(self, state, name):  # type: ignore[no-untyped-def]
        v = _poly_eval(self.px, state.coords[:, 0], self.xp)
        if self.rt is not None:
            v = v * _poly_eval(self.rt, state.coords[:, 1], self.xp)
        return v

    def derivative(self, state, name, *, axis, order):  # type: ignore[no-untyped-def]
        x = state.coords[:, 0]
        if axis == 0:
            dv = _poly_eval(self.px, x, self.xp, deriv=order)
            if self.rt is not None:
                dv = dv * _poly_eval(self.rt, state.coords[:, 1], self.xp)
            return dv
        if axis == 1 and self.rt is not None:
            return _poly_eval(self.px, x, self.xp) * _poly_eval(
                self.rt, state.coords[:, 1], self.xp, deriv=order
            )
        raise NotImplementedError

    def mixed_partial(self, state, name, axes, orders):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def _torch_state(px, rt, x, t=None):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch

    cols = [torch.as_tensor(x, dtype=F)]
    if t is not None:
        cols.append(torch.as_tensor(t, dtype=F))
    coords = torch.stack(cols, dim=1)
    return _PolyField(torch, _ops_dispatch, px, rt)(coords)


def _jax_state(px, rt, x, t=None):  # type: ignore[no-untyped-def]
    from omnibias.fields.jax import _ops_dispatch

    cols = [jnp.asarray(x, dtype=jnp.float64)]
    if t is not None:
        cols.append(jnp.asarray(t, dtype=jnp.float64))
    coords = jnp.stack(cols, axis=1)
    return _PolyField(jnp, _ops_dispatch, px, rt)(coords)


# ----- exact on the analytic (polynomial) class -----


def test_field_fractional_partial_polynomial_power_law() -> None:
    px = [2.0, 3.0, 4.0, -1.0]  # P(x); R = 1
    x = np.linspace(0.2, 1.8, 21)
    st = _torch_state(px, [1.0], x, np.zeros_like(x))
    got = tfield.field_fractional_partial(st, "u", axis="x", alpha=0.6, order=3, a=0.0)
    xt = torch.as_tensor(x, dtype=F)
    expected = torch.zeros_like(xt)
    for k in range(4):
        coef = math.gamma(k + 1) / math.gamma(k + 1 - 0.6)
        expected = expected + px[k] * coef * xt ** (k - 0.6)
    assert torch.allclose(got, expected, rtol=1e-11, atol=1e-11)


def test_field_fractional_partial_matches_standalone_jet_op() -> None:
    px = [1.0, -2.0, 0.5, 0.0, 0.25]  # R = 1 => field jet is exactly px
    x = np.linspace(0.3, 1.5, 17)
    st = _torch_state(px, [1.0], x, np.zeros_like(x))
    got = tfield.field_fractional_partial(
        st, "u", axis="x", alpha=0.7, order=4, a=0.0, kind="caputo"
    )
    exp = ta.fractional_derivative(
        torch.tensor(px, dtype=F), torch.as_tensor(x, dtype=F),
        alpha=0.7, a=0.0, kind="caputo",
    )
    assert torch.allclose(got, exp, rtol=1e-11, atol=1e-11)


def test_field_fractional_partial_carries_transverse_axis() -> None:
    px = [0.0, 1.0, 0.5]
    rt = [1.0, 2.0]  # R(t) = 1 + 2 t
    x = np.linspace(0.2, 1.6, 13)
    t = np.linspace(0.0, 1.0, 13)
    st = _torch_state(px, rt, x, t)
    got = tfield.field_fractional_partial(
        st, "u", axis="x", alpha=0.5, order=2, a=0.0, kind="caputo"
    )
    xt = torch.as_tensor(x, dtype=F)
    R = 1.0 + 2.0 * torch.as_tensor(t, dtype=F)
    dp = torch.zeros_like(xt)
    for k in (1, 2):  # caputo drops k = 0 for 0 < alpha < 1
        coef = math.gamma(k + 1) / math.gamma(k + 1 - 0.5)
        dp = dp + px[k] * coef * xt ** (k - 0.5)
    assert torch.allclose(got, R * dp, rtol=1e-11, atol=1e-11)


def test_field_fractional_partial_integer_reduces_to_derivative() -> None:
    px = [2.0, 3.0, 4.0, -1.0]  # P' = 3 + 8x - 3x^2 ; P'' = 8 - 6x
    rt = [1.0, 0.5, 1.0]
    x = np.linspace(0.2, 1.8, 15)
    t = np.linspace(0.1, 0.9, 15)
    st = _torch_state(px, rt, x, t)
    xt = torch.as_tensor(x, dtype=F)
    R = 1.0 + 0.5 * torch.as_tensor(t, dtype=F) + torch.as_tensor(t, dtype=F) ** 2
    d1 = tfield.field_fractional_partial(st, "u", axis="x", alpha=1.0, order=3)
    d2 = tfield.field_fractional_partial(st, "u", axis="x", alpha=2.0, order=3)
    assert torch.allclose(d1, (3.0 + 8.0 * xt - 3.0 * xt**2) * R, rtol=1e-11, atol=1e-11)
    assert torch.allclose(d2, (8.0 - 6.0 * xt) * R, rtol=1e-11, atol=1e-11)


# ----- differentiability in the order -----


def test_field_fractional_partial_order_is_differentiable() -> None:
    px = [1.0, 0.0, 2.0, -0.5]
    x = np.linspace(0.4, 1.4, 9)
    st = _torch_state(px, [1.0], x, np.zeros_like(x))
    alpha = torch.tensor(0.6, dtype=F, requires_grad=True)
    tfield.field_fractional_partial(st, "u", axis="x", alpha=alpha, order=3).pow(
        2
    ).sum().backward()
    assert alpha.grad is not None and torch.isfinite(alpha.grad)


# ----- cross-backend parity -----


def test_field_fractional_partial_torch_jax_parity() -> None:
    px = [2.0, 3.0, 4.0, -1.0]
    rt = [1.0, 0.5, 1.0]
    x = np.linspace(0.2, 1.8, 19)
    t = np.linspace(0.1, 0.9, 19)
    vt = (
        tfield.field_fractional_partial(
            _torch_state(px, rt, x, t), "u", axis="x", alpha=0.5, order=3, kind="caputo"
        )
        .detach()
        .numpy()
    )
    vj = np.asarray(
        jfield.field_fractional_partial(
            _jax_state(px, rt, x, t), "u", axis="x", alpha=0.5, order=3, kind="caputo"
        )
    )
    assert np.allclose(vt, vj, rtol=1e-9, atol=1e-11)


# ----- fractional-diffusion residual: manufactured solution -----


def _caputo_hand_frac(px, xv, alpha, xp):  # type: ignore[no-untyped-def]
    out = xp.zeros_like(xv)
    ceil_a = math.ceil(alpha)
    for k in range(len(px)):
        if k < ceil_a:
            continue
        coef = math.gamma(k + 1) / math.gamma(k + 1 - alpha)
        out = out + px[k] * coef * xv ** (k - alpha)
    return out


def test_fractional_diffusion_residual_mms_zero_torch() -> None:
    px = [2.0, 3.0, 4.0, -1.0]
    rt = [1.0, 0.5, 1.0]
    alpha, order = 0.5, 3
    x = np.linspace(0.2, 1.8, 25)
    t = np.linspace(0.1, 0.9, 25)
    st = _torch_state(px, rt, x, t)

    def source_fn(state):  # type: ignore[no-untyped-def]
        xv = state.coords[:, 0]
        tv = state.coords[:, 1]
        p = _poly_eval(px, xv, torch)
        r = _poly_eval(rt, tv, torch)
        r_t = _poly_eval(rt, tv, torch, deriv=1)
        u_t = p * r_t
        d_alpha = r * _caputo_hand_frac(px, xv, alpha, torch)
        return u_t - d_alpha

    out = tfield.fractional_diffusion_residual(
        st, alphas=(alpha,), order=order, component="u", kind="caputo", a=0.0,
        source=source_fn,
    )
    assert torch.allclose(out.residual, torch.zeros_like(out.residual), atol=1e-10)
    assert out.diag["mean_sq_residual"] < 1e-16


def test_fractional_diffusion_residual_torch_jax_parity() -> None:
    px = [1.0, 2.0, 0.5, -0.25]
    rt = [1.0, 1.0]
    x = np.linspace(0.2, 1.6, 17)
    t = np.linspace(0.0, 0.8, 17)
    rt_t = tfield.fractional_diffusion_residual(
        _torch_state(px, rt, x, t), alphas=(0.5,), order=3, kind="caputo"
    ).residual
    rj = jfield.fractional_diffusion_residual(
        _jax_state(px, rt, x, t), alphas=(0.5,), order=3, kind="caputo"
    ).residual
    assert np.allclose(rt_t.detach().numpy(), np.asarray(rj), rtol=1e-9, atol=1e-11)


# ----- error paths -----


def test_field_fractional_partial_bad_kind_raises() -> None:
    st = _torch_state([1.0, 2.0], [1.0], np.linspace(0.2, 1.0, 5), np.zeros(5))
    with pytest.raises(ValueError, match="kind must be"):
        tfield.field_fractional_partial(st, "u", axis="x", alpha=0.5, order=2, kind="gl")


def test_field_fractional_partial_negative_order_raises() -> None:
    st = _torch_state([1.0, 2.0], [1.0], np.linspace(0.2, 1.0, 5), np.zeros(5))
    with pytest.raises(ValueError, match="order must be"):
        tfield.field_fractional_partial(st, "u", axis="x", alpha=0.5, order=-1)


def test_fractional_diffusion_residual_requires_time_axis() -> None:
    st = _torch_state([1.0, 2.0, 0.5], None, np.linspace(0.2, 1.0, 6))  # steady
    with pytest.raises(ValueError, match="time axis"):
        tfield.fractional_diffusion_residual(st, alphas=(0.5,), order=2)


def test_fractional_diffusion_residual_alpha_count_mismatch() -> None:
    st = _torch_state([1.0, 2.0, 0.5], [1.0, 0.5], np.linspace(0.2, 1.0, 6), np.zeros(6))
    with pytest.raises(ValueError, match="one order per spatial axis"):
        tfield.fractional_diffusion_residual(st, alphas=(0.5, 0.5), order=2)
