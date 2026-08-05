# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared geometry test fixtures: analytic separable fields + model manifolds.

Each field component is a product of per-axis 1D functions (polynomial / trig),
so value, gradient and Hessian are exact closed forms; the geometry ops consume
them through the state-method dispatch path (no omnibias-pinn dependency). Model
manifolds (sphere, flat) carry analytic per-point metrics for both backends.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ModuleNotFoundError:  # pragma: no cover
    pass

from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState


# --- separable 1D axis functions (backend-agnostic arithmetic) ---------
class Poly1D:
    def __init__(self, coeffs):  # type: ignore[no-untyped-def]
        self.coeffs = tuple(float(c) for c in coeffs)

    def _eval(self, coeffs, x):  # type: ignore[no-untyped-def]
        acc = None
        for k, c in enumerate(coeffs):
            term = c * x**k
            acc = term if acc is None else acc + term
        return acc

    def value(self, x):  # type: ignore[no-untyped-def]
        return self._eval(self.coeffs, x)

    def deriv(self, x, order):  # type: ignore[no-untyped-def]
        c = list(self.coeffs)
        for _ in range(order):
            c = [k * c[k] for k in range(1, len(c))] if len(c) > 1 else [0.0]
        if not c:
            c = [0.0]
        return self._eval(c, x)


class Cos1D:
    """``amp * cos(freq * x)`` with exact derivatives."""

    def __init__(self, amp=1.0, freq=1.0, xp=None):  # type: ignore[no-untyped-def]
        self.amp = float(amp)
        self.freq = float(freq)
        self.xp = xp  # module exposing sin/cos (torch or jnp)

    def value(self, x):  # type: ignore[no-untyped-def]
        return self.amp * self.xp.cos(self.freq * x)

    def deriv(self, x, order):  # type: ignore[no-untyped-def]
        f = self.freq
        a = self.amp * (f**order)
        m = order % 4
        if m == 0:
            return a * self.xp.cos(f * x)
        if m == 1:
            return -a * self.xp.sin(f * x)
        if m == 2:
            return -a * self.xp.cos(f * x)
        return a * self.xp.sin(f * x)


class Sin1D:
    """``amp * sin(freq * x)`` with exact derivatives."""

    def __init__(self, amp=1.0, freq=1.0, xp=None):  # type: ignore[no-untyped-def]
        self.amp = float(amp)
        self.freq = float(freq)
        self.xp = xp  # module exposing sin/cos (torch or jnp)

    def value(self, x):  # type: ignore[no-untyped-def]
        return self.amp * self.xp.sin(self.freq * x)

    def deriv(self, x, order):  # type: ignore[no-untyped-def]
        f = self.freq
        a = self.amp * (f**order)
        m = order % 4
        if m == 0:
            return a * self.xp.sin(f * x)
        if m == 1:
            return a * self.xp.cos(f * x)
        if m == 2:
            return -a * self.xp.sin(f * x)
        return -a * self.xp.cos(f * x)


class Const1D:
    def __init__(self, c=1.0):  # type: ignore[no-untyped-def]
        self.c = float(c)

    def value(self, x):  # type: ignore[no-untyped-def]
        return self.c + 0.0 * x

    def deriv(self, x, order):  # type: ignore[no-untyped-def]
        return 0.0 * x


class AnalyticField:
    _omnibias_dispatch = "spectral"

    def __init__(self, coordinate_spec, components, comp_axes, ops_module):  # type: ignore[no-untyped-def]
        self.coordinate_spec = coordinate_spec
        self.components = components
        self._axes = comp_axes  # dict[name -> tuple[axis1d, ...]]
        self._ops = ops_module

    def evaluate(self, coords):  # type: ignore[no-untyped-def]
        return FieldState(
            coords=coords, field=self, components=self.components,
            coordinate_spec=self.coordinate_spec, ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    __call__ = evaluate

    def value_component(self, state, name):  # type: ignore[no-untyped-def]
        x = state.coords
        acc = None
        for d, ax in enumerate(self._axes[name]):
            term = ax.value(x[:, d])
            acc = term if acc is None else acc * term
        return acc

    def derivative(self, state, name, *, axis, order):  # type: ignore[no-untyped-def]
        x = state.coords
        acc = None
        for d, ax in enumerate(self._axes[name]):
            term = ax.deriv(x[:, d], order) if d == axis else ax.value(x[:, d])
            acc = term if acc is None else acc * term
        return acc

    def mixed_partial(self, state, name, axes, orders):  # type: ignore[no-untyped-def]
        ob = dict(zip(axes, orders, strict=False))
        x = state.coords
        acc = None
        for d, ax in enumerate(self._axes[name]):
            o = ob.get(d, 0)
            term = ax.deriv(x[:, d], o) if o > 0 else ax.value(x[:, d])
            acc = term if acc is None else acc * term
        return acc


# --- model manifolds ----------------------------------------------------
SPHERE_R = 1.3
# Torus of revolution (major radius R, minor radius r); R - r > 0 keeps the
# metric non-degenerate over the whole (theta, phi) chart.
TORUS_R = 2.0
TORUS_r = 0.7
# Conformally-flat 2-metric  g = e^{2 lambda} I  with
#   lambda(theta, phi) = CONF_A sin(theta) + CONF_B cos(phi).
CONF_A = 0.3
CONF_B = 0.2


def _sphere_metric_factory(xp, stack):  # type: ignore[no-untyped-def]
    r2 = SPHERE_R**2

    def g_point(x):  # x: (2,) = (theta, phi)
        theta = x[0]
        g00 = r2 * (1.0 + 0.0 * theta)
        g11 = r2 * xp.sin(theta) ** 2
        z = 0.0 * theta
        return stack([stack([g00, z]), stack([z, g11])])

    return g_point


def _torus_metric_factory(xp, stack):  # type: ignore[no-untyped-def]
    rmin2 = TORUS_r**2

    def g_point(x):  # x: (2,) = (theta, phi)
        theta = x[0]
        g00 = rmin2 * (1.0 + 0.0 * theta)
        g11 = (TORUS_R + TORUS_r * xp.cos(theta)) ** 2
        z = 0.0 * theta
        return stack([stack([g00, z]), stack([z, g11])])

    return g_point


def _conformal_metric_factory(xp, stack):  # type: ignore[no-untyped-def]
    def g_point(x):  # x: (2,) = (theta, phi)
        theta = x[0]
        phi = x[1]
        lam = CONF_A * xp.sin(theta) + CONF_B * xp.cos(phi)
        e2 = xp.exp(2.0 * lam)
        z = 0.0 * theta
        return stack([stack([e2, z]), stack([z, e2])])

    return g_point


def _flat_metric_factory(xp, stack, dim):  # type: ignore[no-untyped-def]
    def g_point(x):
        rows = []
        for i in range(dim):
            rows.append(stack([(1.0 if i == j else 0.0) + 0.0 * x[0] for j in range(dim)]))
        return stack(rows)

    return g_point


@pytest.fixture
def torch_mod():  # type: ignore[no-untyped-def]
    import torch

    return torch


@pytest.fixture
def jax_mod():  # type: ignore[no-untyped-def]
    import jax.numpy as jnp

    return jnp


def _coord_spec_2d():  # type: ignore[no-untyped-def]
    return CoordinateSpec(("theta", "phi"), time_axis=None)


# expose builders to tests
@pytest.fixture
def builders():  # type: ignore[no-untyped-def]
    return {
        "Poly1D": Poly1D,
        "Cos1D": Cos1D,
        "Sin1D": Sin1D,
        "Const1D": Const1D,
        "AnalyticField": AnalyticField,
        "coord_spec_2d": _coord_spec_2d,
        "ComponentSpec": ComponentSpec,
        "sphere_metric_factory": _sphere_metric_factory,
        "flat_metric_factory": _flat_metric_factory,
        "torus_metric_factory": _torus_metric_factory,
        "conformal_metric_factory": _conformal_metric_factory,
        "SPHERE_R": SPHERE_R,
        "TORUS_R": TORUS_R,
        "TORUS_r": TORUS_r,
        "CONF_A": CONF_A,
        "CONF_B": CONF_B,
        "math": math,
        "np": np,
    }
