# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tiny analytic omnibias fields for the variational examples.

The `omnibias-variational` ops take an `omnibias.fields.FieldState` and read the
trajectory ``q(t)`` (or field ``phi(x, t)``) and its *closed-form* derivatives
off it. For a **known** analytic path we attach it with a minimal
``"spectral"``-dispatch field: the field just answers ``value`` / ``derivative``
/ ``mixed_partial`` queries from user-supplied callables, so the omnibias field
ops return exact values and the variational ops (action, Euler-Lagrange, energy,
...) are exercised against a ground truth.

These are the same helpers the package test-suite uses, trimmed for the docs.
Torch only, float64. (Real trainable trajectories use a neural field such as
``omnibias.pinn.torch.fields.OneLayerVectorField``; see the direct-method note
in ``docs/api/variational.md``.)
"""

from __future__ import annotations

import torch
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState
from omnibias.fields.torch import _ops_dispatch


class TrajectoryField:
    """Analytic trajectory ``t -> (component values)`` on a 1-D time axis.

    ``specs`` maps each component name to ``(value(t), d1(t), d2(t), ...)`` --
    callables of the (single) time coordinate returning ``(B,)`` tensors, indexed
    by derivative order. A 3-tuple covers orders 0..2; pass a longer tuple for the
    higher-order (Euler-Poisson) examples. Callables may also ignore ``t`` and
    return precomputed per-node arrays (used for the cycloid).
    """

    _omnibias_dispatch = "spectral"

    def __init__(self, specs, *, axes=("t",), time_axis="t"):
        self.coordinate_spec = CoordinateSpec(axes, time_axis=time_axis)
        self.components = ComponentSpec(tuple(specs.keys()))
        self._specs = specs

    def evaluate(self, coords):
        return FieldState(
            coords=coords, field=self, components=self.components,
            coordinate_spec=self.coordinate_spec, ops=_ops_dispatch,
            sigma_cache=SigmaCache(z=coords),
        )

    __call__ = evaluate

    def _t(self, state):
        return state.coords[:, self.coordinate_spec.axis_index(self.coordinate_spec.time_axis)]

    def value_component(self, state, name):
        return self._specs[name][0](self._t(state))

    def derivative(self, state, name, *, axis, order):
        derivs = self._specs[name]
        return derivs[order](self._t(state)) if 0 <= order < len(derivs) else _no(order, name)

    def mixed_partial(self, state, name, axes, orders):
        raise NotImplementedError


class PlaneWaveField:
    """Analytic scalar field ``phi(x, t) = cos(k x - w t)`` on axes ``(x, t)``.

    ``phi`` is a cosine of the linear phase ``k x - w t``, so every (mixed)
    partial is exact: differentiating once along an axis multiplies by that axis'
    phase coefficient (``k`` for ``x``, ``-w`` for ``t``) and rotates the cosine.
    """

    _omnibias_dispatch = "spectral"

    def __init__(self, k, omega):
        self.coordinate_spec = CoordinateSpec(("x", "t"), time_axis="t")
        self.components = ComponentSpec(("phi",))
        self._coeff = (k, -omega)

    def evaluate(self, coords):
        return FieldState(
            coords=coords, field=self, components=self.components,
            coordinate_spec=self.coordinate_spec, ops=_ops_dispatch,
            sigma_cache=SigmaCache(z=coords),
        )

    __call__ = evaluate

    def _arg(self, state):
        cx, ct = self._coeff
        return cx * state.coords[:, 0] + ct * state.coords[:, 1]

    def _dcos(self, arg, n):
        return (torch.cos(arg), -torch.sin(arg), -torch.cos(arg), torch.sin(arg))[n % 4]

    def value_component(self, state, name):
        return torch.cos(self._arg(state))

    def derivative(self, state, name, *, axis, order):
        if order == 0:
            return self.value_component(state, name)
        return (self._coeff[axis] ** order) * self._dcos(self._arg(state), order)

    def mixed_partial(self, state, name, axes, orders):
        factor, total = 1.0, 0
        for a, o in zip(axes, orders, strict=False):
            factor = factor * self._coeff[a] ** o
            total += o
        return factor * self._dcos(self._arg(state), total)


def _no(order, name):
    raise NotImplementedError(f"order {order} not provided for {name!r}")


def column(values):
    """Shape a 1-D sequence of coordinate samples into a ``(B, 1)`` float64 tensor."""
    return torch.as_tensor(values, dtype=torch.float64).reshape(-1, 1)
