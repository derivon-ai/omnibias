# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wirtinger calculus: Cauchy-Riemann (dzbar = 0) for holomorphic fields.

The complex field is ``f(z) = exp(z) = e^x cos y + i e^x sin y`` (holomorphic),
carried as two real components. Checks: ``dzbar f == 0``, ``dz f == f`` (since
d/dz e^z = e^z), and torch/jax bit-parity. All in float64.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState

COORDS = np.array([[0.2, 0.5], [-0.4, 1.1], [0.7, -0.3], [0.1, 2.0]], dtype=np.float64)


class _ExpZField:
    """f = exp(z): fR = e^x cos y, fI = e^x sin y (state-method dispatch)."""

    _omnibias_dispatch = "spectral"

    def __init__(self, xp, ops_module):  # type: ignore[no-untyped-def]
        self.xp = xp
        self.coordinate_spec = CoordinateSpec(("x", "y"), time_axis=None)
        self.components = ComponentSpec(("fR", "fI"))
        self._ops = ops_module

    def evaluate(self, coords):  # type: ignore[no-untyped-def]
        return FieldState(
            coords=coords, field=self, components=self.components,
            coordinate_spec=self.coordinate_spec, ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    __call__ = evaluate

    def value_component(self, state, name):  # type: ignore[no-untyped-def]
        x, y = state.coords[:, 0], state.coords[:, 1]
        ex = self.xp.exp(x)
        return ex * self.xp.cos(y) if name == "fR" else ex * self.xp.sin(y)

    def derivative(self, state, name, *, axis, order):  # type: ignore[no-untyped-def]
        if order != 1:
            raise NotImplementedError("test field only needs first derivatives")
        x, y = state.coords[:, 0], state.coords[:, 1]
        ex = self.xp.exp(x)
        c, s = self.xp.cos(y), self.xp.sin(y)
        # fR = e^x cos y: d/dx = e^x cos y, d/dy = -e^x sin y
        # fI = e^x sin y: d/dx = e^x sin y, d/dy =  e^x cos y
        if name == "fR":
            return ex * c if axis == 0 else -ex * s
        return ex * s if axis == 0 else ex * c

    def mixed_partial(self, state, name, axes, orders):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def _np(v):  # type: ignore[no-untyped-def]
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def test_cauchy_riemann_holomorphic():  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch
    from omnibias.fields.torch.ops.complex import dzbar

    state = _ExpZField(torch, _ops_dispatch)(torch.as_tensor(COORDS, dtype=torch.float64))
    re, im = dzbar(state, "fR", "fI")
    assert np.allclose(_np(re), 0.0, atol=1e-12)
    assert np.allclose(_np(im), 0.0, atol=1e-12)


def test_dz_of_exp_is_exp():  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch
    from omnibias.fields.torch.ops.complex import dz

    field = _ExpZField(torch, _ops_dispatch)
    state = field(torch.as_tensor(COORDS, dtype=torch.float64))
    re, im = dz(state, "fR", "fI")
    # d/dz e^z = e^z -> (fR, fI)
    assert np.allclose(_np(re), _np(field.value_component(state, "fR")), atol=1e-12)
    assert np.allclose(_np(im), _np(field.value_component(state, "fI")), atol=1e-12)


def test_wirtinger_cross_backend():  # type: ignore[no-untyped-def]
    from omnibias.fields.jax import _ops_dispatch as jd
    from omnibias.fields.jax.ops.complex import dz as jdz
    from omnibias.fields.torch import _ops_dispatch as td
    from omnibias.fields.torch.ops.complex import dz as tdz

    ts = _ExpZField(torch, td)(torch.as_tensor(COORDS, dtype=torch.float64))
    js = _ExpZField(jnp, jd)(jnp.asarray(COORDS, dtype=jnp.float64))
    tre, tim = tdz(ts, "fR", "fI")
    jre, jim = jdz(js, "fR", "fI")
    assert np.allclose(_np(tre), _np(jre), rtol=1e-12, atol=1e-12)
    assert np.allclose(_np(tim), _np(jim), rtol=1e-12, atol=1e-12)
