# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tiny analytic closed-form fields for the notebook gallery.

The field-level ops in ``omnibias-fields`` / ``omnibias-geometry`` /
``omnibias-score`` consume a :class:`~omnibias.fields._core.state.FieldState`
whose ``value`` / ``derivative`` / ``mixed_partial`` are exact. In production you
get one from ``omnibias-pinn``'s field constructors (a trained network); for a
tutorial we instead build a *separable analytic* field whose every derivative is
a known closed form, so each op can be checked against an exact reference.

Each component is a product of per-axis 1D functions::

    field = make_field(
        axes=("theta", "phi"),
        comp_axes={"f": (Cos(xp=torch), Const())},   # f(theta, phi) = cos(theta)
        ops_module=torch_dispatch,
    )
    state = field(coords)        # coords: (B, n_axes) backend tensor

The same pure-Python arithmetic runs on torch and jax tensors, so the two
backends are bit-identical by construction.
"""

from __future__ import annotations

from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState


class Poly:
    """A 1D polynomial ``sum_k c[k] x^k`` with exact derivatives."""

    def __init__(self, coeffs):
        self.coeffs = tuple(float(c) for c in coeffs)

    def _eval(self, coeffs, x):
        acc = None
        for k, c in enumerate(coeffs):
            term = c * x**k
            acc = term if acc is None else acc + term
        return acc

    def value(self, x):
        return self._eval(self.coeffs, x)

    def deriv(self, x, order):
        c = list(self.coeffs)
        for _ in range(order):
            c = [k * c[k] for k in range(1, len(c))] if len(c) > 1 else [0.0]
        return self._eval(c or [0.0], x)


class Cos:
    """``amp * cos(freq * x)`` with exact derivatives (needs a backend module)."""

    def __init__(self, amp=1.0, freq=1.0, xp=None):
        self.amp = float(amp)
        self.freq = float(freq)
        self.xp = xp

    def value(self, x):
        return self.amp * self.xp.cos(self.freq * x)

    def deriv(self, x, order):
        a = self.amp * (self.freq**order)
        m = order % 4
        if m == 0:
            return a * self.xp.cos(self.freq * x)
        if m == 1:
            return -a * self.xp.sin(self.freq * x)
        if m == 2:
            return -a * self.xp.cos(self.freq * x)
        return a * self.xp.sin(self.freq * x)


class Gauss:
    """``exp(-c x^2)`` with exact derivatives up to order 2 (needs a backend)."""

    def __init__(self, c=1.0, xp=None):
        self.c = float(c)
        self.xp = xp

    def value(self, x):
        return self.xp.exp(-self.c * x**2)

    def deriv(self, x, order):
        g = self.value(x)
        if order == 0:
            return g
        if order == 1:
            return -2.0 * self.c * x * g
        if order == 2:
            return (4.0 * self.c**2 * x**2 - 2.0 * self.c) * g
        raise NotImplementedError("Gauss.deriv supports order in {0, 1, 2}")


class Const:
    """The constant ``c`` (zero derivative)."""

    def __init__(self, c=1.0):
        self.c = float(c)

    def value(self, x):
        return self.c + 0.0 * x

    def deriv(self, x, order):
        return 0.0 * x


class _AnalyticField:
    _omnibias_dispatch = "spectral"  # use the state-method op path

    def __init__(self, coordinate_spec, components, comp_axes, ops_module):
        self.coordinate_spec = coordinate_spec
        self.components = components
        self._axes = comp_axes
        self._ops = ops_module

    def evaluate(self, coords):
        return FieldState(
            coords=coords, field=self, components=self.components,
            coordinate_spec=self.coordinate_spec, ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    __call__ = evaluate

    def value_component(self, state, name):
        return self._product(state, name, {})

    def derivative(self, state, name, *, axis, order):
        return self._product(state, name, {axis: order})

    def mixed_partial(self, state, name, axes, orders):
        return self._product(state, name, dict(zip(axes, orders, strict=False)))

    def _product(self, state, name, order_by_axis):
        x = state.coords
        acc = None
        for d, fn in enumerate(self._axes[name]):
            o = order_by_axis.get(d, 0)
            term = fn.deriv(x[:, d], o) if o > 0 else fn.value(x[:, d])
            acc = term if acc is None else acc * term
        return acc


def make_field(axes, comp_axes, ops_module, *, groups=None):
    """Build an analytic separable field over the named ``axes``.

    Parameters
    ----------
    axes
        Tuple of coordinate-axis names, e.g. ``("theta", "phi")``.
    comp_axes
        Mapping ``component_name -> tuple(per_axis_fn, ...)`` (one 1D function
        per axis). The component value is the product across axes.
    ops_module
        The backend dispatch module, e.g. ``omnibias.fields.torch._ops_dispatch``.
    groups
        Optional named component groups for vector views.
    """
    coord_spec = CoordinateSpec(tuple(axes), time_axis=None)
    comp_spec = ComponentSpec(tuple(comp_axes), groups=groups or {})
    return _AnalyticField(coord_spec, comp_spec, dict(comp_axes), ops_module)
