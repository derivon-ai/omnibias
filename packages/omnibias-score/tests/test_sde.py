# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""SDE / score operators validated on the Ornstein-Uhlenbeck process.

OU: ``dX = -theta X dt + sigma dW`` has stationary density
``N(0, sigma^2 / (2 theta))``, for which the Fokker-Planck adjoint vanishes:
``L* p_inf = 0``. The generator on monomials and the score of the Gaussian have
closed forms too. Cross-backend parity is checked. All float64.
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
from omnibias.score.jax import ops as jsde
from omnibias.score.torch import ops as tsde

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ModuleNotFoundError:  # pragma: no cover
    pass

THETA = 0.8
A = 0.5  # sigma^2
V = A / (2.0 * THETA)  # stationary variance
X = np.array([-1.2, -0.4, 0.3, 0.9, 1.7], dtype=np.float64)


class _OUField:
    """1D field with components: gaussian stationary density + monomials."""

    _omnibias_dispatch = "spectral"

    def __init__(self, xp, ops_module):  # type: ignore[no-untyped-def]
        self.xp = xp
        self.coordinate_spec = CoordinateSpec(("x",), time_axis=None)
        self.components = ComponentSpec(("gauss", "lin", "quad"))
        self._ops = ops_module

    def evaluate(self, coords):  # type: ignore[no-untyped-def]
        return FieldState(
            coords=coords, field=self, components=self.components,
            coordinate_spec=self.coordinate_spec, ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    __call__ = evaluate

    def _g(self, x):  # type: ignore[no-untyped-def]
        return self.xp.exp(-x * x / (2.0 * V))

    def value_component(self, state, name):  # type: ignore[no-untyped-def]
        x = state.coords[:, 0]
        if name == "gauss":
            return self._g(x)
        if name == "lin":
            return x
        return x * x

    def derivative(self, state, name, *, axis, order):  # type: ignore[no-untyped-def]
        x = state.coords[:, 0]
        if name == "gauss":
            g = self._g(x)
            if order == 1:
                return -x / V * g
            if order == 2:
                return (x * x / V**2 - 1.0 / V) * g
            raise NotImplementedError
        if name == "lin":
            return self.xp.ones_like(x) if order == 1 else 0.0 * x
        if name == "quad":
            if order == 1:
                return 2.0 * x
            if order == 2:
                return 2.0 * self.xp.ones_like(x)
            return 0.0 * x
        raise NotImplementedError

    def mixed_partial(self, state, name, axes, orders):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def _np(v):  # type: ignore[no-untyped-def]
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def _torch_state():  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch

    return _OUField(torch, _ops_dispatch)(torch.as_tensor(X[:, None], dtype=torch.float64))


def _jax_state():  # type: ignore[no-untyped-def]
    from omnibias.fields.jax import _ops_dispatch

    return _OUField(jnp, _ops_dispatch)(jnp.asarray(X[:, None], dtype=jnp.float64))


def _drift(xp, x):  # type: ignore[no-untyped-def]
    return -THETA * x  # (B, 1)


def test_fokker_planck_stationary_is_zero() -> None:
    state = _torch_state()
    b = torch.as_tensor(-THETA * X[:, None], dtype=torch.float64)
    a = torch.as_tensor([[A]], dtype=torch.float64)
    div = torch.as_tensor(np.full_like(X, -THETA), dtype=torch.float64)
    res = _np(tsde.fokker_planck(state, "gauss", drift=b, diffusion=a, drift_divergence=div))
    assert np.allclose(res, 0.0, atol=1e-10)


def test_generator_on_monomials() -> None:
    state = _torch_state()
    b = torch.as_tensor(-THETA * X[:, None], dtype=torch.float64)
    a = torch.as_tensor([[A]], dtype=torch.float64)
    # L x = -theta x
    lin = _np(tsde.ito_generator(state, "lin", drift=b, diffusion=a))
    assert np.allclose(lin, -THETA * X, atol=1e-12)
    # L x^2 = -2 theta x^2 + a
    quad = _np(tsde.ito_generator(state, "quad", drift=b, diffusion=a))
    assert np.allclose(quad, -2.0 * THETA * X**2 + A, atol=1e-12)


def test_score_of_gaussian() -> None:
    state = _torch_state()
    sc = _np(tsde.score(state, "gauss"))  # grad log p = -x / V
    assert np.allclose(sc[:, 0], -X / V, atol=1e-10)


def test_sde_cross_backend() -> None:
    ts, js = _torch_state(), _jax_state()
    tb = torch.as_tensor(-THETA * X[:, None], dtype=torch.float64)
    jb = jnp.asarray(-THETA * X[:, None], dtype=jnp.float64)
    ta = torch.as_tensor([[A]], dtype=torch.float64)
    ja = jnp.asarray([[A]], dtype=jnp.float64)
    t = _np(tsde.ito_generator(ts, "quad", drift=tb, diffusion=ta))
    j = _np(jsde.ito_generator(js, "quad", drift=jb, diffusion=ja))
    assert np.allclose(t, j, rtol=1e-12, atol=1e-12)
    ts_sc = _np(tsde.score(ts, "gauss"))
    js_sc = _np(jsde.score(js, "gauss"))
    assert np.allclose(ts_sc, js_sc, rtol=1e-12, atol=1e-12)


def test_ou_closed_forms_dense_grid_and_random() -> None:
    """Dense-grid + random-sample soundness: the OU generator, stationary
    Fokker-Planck adjoint and Gaussian score match their closed forms -- and torch
    matches jax -- at a dense grid AND a random sample of spatial points (the
    tests above check only a fixed 5-point sample)."""
    from omnibias.fields.jax import _ops_dispatch as j_ops
    from omnibias.fields.torch import _ops_dispatch as t_ops

    rng = np.random.default_rng(41)
    dense = np.linspace(-3.0, 3.0, 61, dtype=np.float64)
    rand = rng.uniform(-3.0, 3.0, size=40).astype(np.float64)
    xs = np.concatenate([dense, rand])

    ts = _OUField(torch, t_ops)(torch.as_tensor(xs[:, None], dtype=torch.float64))
    js = _OUField(jnp, j_ops)(jnp.asarray(xs[:, None], dtype=jnp.float64))

    tb = torch.as_tensor(-THETA * xs[:, None], dtype=torch.float64)
    jb = jnp.asarray(-THETA * xs[:, None], dtype=jnp.float64)
    ta = torch.as_tensor([[A]], dtype=torch.float64)
    ja = jnp.asarray([[A]], dtype=jnp.float64)
    div = torch.as_tensor(np.full_like(xs, -THETA), dtype=torch.float64)

    # L* p_inf = 0 everywhere on the stationary Gaussian.
    fp = _np(tsde.fokker_planck(ts, "gauss", drift=tb, diffusion=ta, drift_divergence=div))
    assert np.allclose(fp, 0.0, atol=1e-9)

    # Generator closed forms at every sample point: L x = -theta x, L x^2 = -2 theta x^2 + a.
    lin = _np(tsde.ito_generator(ts, "lin", drift=tb, diffusion=ta))
    quad = _np(tsde.ito_generator(ts, "quad", drift=tb, diffusion=ta))
    assert np.allclose(lin, -THETA * xs, atol=1e-11)
    assert np.allclose(quad, -2.0 * THETA * xs**2 + A, atol=1e-11)

    # Gaussian score closed form + cross-backend parity on the whole sample.
    sc_t = _np(tsde.score(ts, "gauss"))[:, 0]
    sc_j = _np(jsde.score(js, "gauss"))[:, 0]
    assert np.allclose(sc_t, -xs / V, atol=1e-9)
    assert np.allclose(sc_t, sc_j, rtol=1e-12, atol=1e-12)
    quad_j = _np(jsde.ito_generator(js, "quad", drift=jb, diffusion=ja))
    assert np.allclose(quad, quad_j, rtol=1e-12, atol=1e-12)
