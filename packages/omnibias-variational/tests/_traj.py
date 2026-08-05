# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared analytic fields for the variational tests.

A closed-form ("spectral"-dispatch) field whose components are user-supplied
analytic functions of the coordinates, so the omnibias field ops return exact
values / derivatives and the variational ops can be checked against hand
computations. Torch and jax builders share the same spec, so the cross-backend
parity tests are apples-to-apples.
"""

from __future__ import annotations

import numpy as np
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState

try:  # float64 parity needs x64 on jax
    import jax

    jax.config.update("jax_enable_x64", True)
except ModuleNotFoundError:  # pragma: no cover
    pass


class AnalyticTrajField:
    """Analytic trajectory field ``t -> (component values)``.

    ``specs`` maps each component name to a tuple of callables
    ``(value(t), d1(t), d2(t), ...)`` of the (single) time coordinate, indexed
    by derivative order. A 3-tuple supports orders 0..2; pass a longer tuple for
    the higher-order (Euler-Poisson) operators.
    """

    _omnibias_dispatch = "spectral"

    def __init__(self, xp, ops_module, specs, *, axes=("t",), time_axis="t"):  # type: ignore[no-untyped-def]
        self.xp = xp
        self.coordinate_spec = CoordinateSpec(axes, time_axis=time_axis)
        self.components = ComponentSpec(tuple(specs.keys()))
        self._ops = ops_module
        self._specs = specs

    def evaluate(self, coords):  # type: ignore[no-untyped-def]
        return FieldState(
            coords=coords, field=self, components=self.components,
            coordinate_spec=self.coordinate_spec, ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    __call__ = evaluate

    def _t(self, state):  # type: ignore[no-untyped-def]
        return state.coords[:, self.coordinate_spec.axis_index("t")]

    def value_component(self, state, name):  # type: ignore[no-untyped-def]
        return self._specs[name][0](self._t(state))

    def derivative(self, state, name, *, axis, order):  # type: ignore[no-untyped-def]
        derivs = self._specs[name]
        if 0 <= order < len(derivs):
            return derivs[order](self._t(state))
        raise NotImplementedError(f"order {order} not provided for {name!r}")

    def mixed_partial(self, state, name, axes, orders):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def sho_specs(xp, omega):  # type: ignore[no-untyped-def]
    """Harmonic solutions ``cos(w t)`` / ``sin(w t)`` and the ramp ``lin = t``.

    ``(cos, sin)`` together trace circular motion of the 2-D isotropic
    oscillator, used for the rotational-symmetry (angular-momentum) test.
    """
    return {
        "cos": (
            lambda t: xp.cos(omega * t),
            lambda t: -omega * xp.sin(omega * t),
            lambda t: -(omega**2) * xp.cos(omega * t),
        ),
        "sin": (
            lambda t: xp.sin(omega * t),
            lambda t: omega * xp.cos(omega * t),
            lambda t: -(omega**2) * xp.sin(omega * t),
        ),
        "lin": (
            lambda t: t,
            lambda t: xp.ones_like(t),
            lambda t: xp.zeros_like(t),
        ),
    }


class AnalyticPlaneWaveField:
    """Analytic scalar field ``phi(x, t) = cos(k x - w t)`` on axes ``(x, t)``.

    Every partial derivative is exact: ``phi`` is linear in the phase
    ``arg = k x - w t``, so an order-``n`` derivative along an axis just brings
    down its phase coefficient (``k`` for ``x``, ``-w`` for ``t``) and shifts the
    cosine. Supports single / mixed partials, so the field ops build the exact
    gradient and Hessian used by the field Euler-Lagrange operator.
    """

    _omnibias_dispatch = "spectral"

    def __init__(self, xp, ops_module, k, omega):  # type: ignore[no-untyped-def]
        self.xp = xp
        self.coordinate_spec = CoordinateSpec(("x", "t"), time_axis="t")
        self.components = ComponentSpec(("phi",))
        self._ops = ops_module
        self._coeff = (k, -omega)  # phase coefficient per axis (x, t)

    def evaluate(self, coords):  # type: ignore[no-untyped-def]
        return FieldState(
            coords=coords, field=self, components=self.components,
            coordinate_spec=self.coordinate_spec, ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    __call__ = evaluate

    def _arg(self, state):  # type: ignore[no-untyped-def]
        # arg = k x - w t, whose per-axis phase coefficients are self._coeff.
        cx, ct = self._coeff
        return cx * state.coords[:, 0] + ct * state.coords[:, 1]

    def _dcos(self, arg, n):  # type: ignore[no-untyped-def]
        m = n % 4
        if m == 0:
            return self.xp.cos(arg)
        if m == 1:
            return -self.xp.sin(arg)
        if m == 2:
            return -self.xp.cos(arg)
        return self.xp.sin(arg)

    def value_component(self, state, name):  # type: ignore[no-untyped-def]
        return self.xp.cos(self._arg(state))

    def derivative(self, state, name, *, axis, order):  # type: ignore[no-untyped-def]
        if order == 0:
            return self.value_component(state, name)
        c = self._coeff[axis]
        return (c**order) * self._dcos(self._arg(state), order)

    def mixed_partial(self, state, name, axes, orders):  # type: ignore[no-untyped-def]
        factor = 1.0
        total = 0
        for a, o in zip(axes, orders, strict=False):
            factor = factor * self._coeff[a] ** o
            total += o
        return factor * self._dcos(self._arg(state), total)


class AnalyticSeparableField:
    """Analytic scalar field ``phi(x, t) = fx(x) * ft(t)`` on axes ``(x, t)``.

    ``fx`` / ``ft`` are 2-tuples ``(f, f')`` of callables of one coordinate. Only
    the value and first derivatives are supplied -- enough to serve as a
    first-variation perturbation ``eta`` (value + gradient). A product of sines
    vanishing on a box boundary makes ``eta`` an admissible variation.
    """

    _omnibias_dispatch = "spectral"

    def __init__(self, xp, ops_module, fx, ft):  # type: ignore[no-untyped-def]
        self.xp = xp
        self.coordinate_spec = CoordinateSpec(("x", "t"), time_axis="t")
        self.components = ComponentSpec(("phi",))
        self._ops = ops_module
        self._fx = fx
        self._ft = ft

    def evaluate(self, coords):  # type: ignore[no-untyped-def]
        return FieldState(
            coords=coords, field=self, components=self.components,
            coordinate_spec=self.coordinate_spec, ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    __call__ = evaluate

    def _xt(self, state):  # type: ignore[no-untyped-def]
        return state.coords[:, 0], state.coords[:, 1]

    def value_component(self, state, name):  # type: ignore[no-untyped-def]
        x, t = self._xt(state)
        return self._fx[0](x) * self._ft[0](t)

    def derivative(self, state, name, *, axis, order):  # type: ignore[no-untyped-def]
        x, t = self._xt(state)
        if order == 0:
            return self._fx[0](x) * self._ft[0](t)
        if order == 1:
            if axis == 0:
                return self._fx[1](x) * self._ft[0](t)
            return self._fx[0](x) * self._ft[1](t)
        raise NotImplementedError(f"order {order} not provided")

    def mixed_partial(self, state, name, axes, orders):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def torch_separable_state(fx_fac, ft_fac, xt_values):  # type: ignore[no-untyped-def]
    import torch
    from omnibias.fields.torch import _ops_dispatch

    field = AnalyticSeparableField(torch, _ops_dispatch, fx_fac(torch), ft_fac(torch))
    coords = torch.as_tensor(np.asarray(xt_values), dtype=torch.float64)
    return field(coords)


def jax_separable_state(fx_fac, ft_fac, xt_values):  # type: ignore[no-untyped-def]
    import jax.numpy as jnp
    from omnibias.fields.jax import _ops_dispatch

    field = AnalyticSeparableField(jnp, _ops_dispatch, fx_fac(jnp), ft_fac(jnp))
    coords = jnp.asarray(np.asarray(xt_values), dtype=jnp.float64)
    return field(coords)


def torch_state(specs_fn, omega, t_values):  # type: ignore[no-untyped-def]
    import torch
    from omnibias.fields.torch import _ops_dispatch

    field = AnalyticTrajField(torch, _ops_dispatch, specs_fn(torch, omega))
    coords = torch.as_tensor(np.asarray(t_values)[:, None], dtype=torch.float64)
    return field(coords)


def jax_state(specs_fn, omega, t_values):  # type: ignore[no-untyped-def]
    import jax.numpy as jnp
    from omnibias.fields.jax import _ops_dispatch

    field = AnalyticTrajField(jnp, _ops_dispatch, specs_fn(jnp, omega))
    coords = jnp.asarray(np.asarray(t_values)[:, None], dtype=jnp.float64)
    return field(coords)


def torch_planewave_state(k, omega, xt_values):  # type: ignore[no-untyped-def]
    import torch
    from omnibias.fields.torch import _ops_dispatch

    field = AnalyticPlaneWaveField(torch, _ops_dispatch, k, omega)
    coords = torch.as_tensor(np.asarray(xt_values), dtype=torch.float64)
    return field(coords)


def jax_planewave_state(k, omega, xt_values):  # type: ignore[no-untyped-def]
    import jax.numpy as jnp
    from omnibias.fields.jax import _ops_dispatch

    field = AnalyticPlaneWaveField(jnp, _ops_dispatch, k, omega)
    coords = jnp.asarray(np.asarray(xt_values), dtype=jnp.float64)
    return field(coords)


def to_np(v):  # type: ignore[no-untyped-def]
    import torch

    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
