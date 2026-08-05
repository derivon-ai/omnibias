# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Riemannian optimisation in a learned-chart pullback metric (torch + jax).

The metric-pluggable optimisers :class:`omnibias.torch.optim.NaturalGradient` and
:func:`omnibias.jax.optim.natural_gradient_step` accept *any* SPD metric provider, so the
closed-form geometry pullback ``g = J^T h J``
(:func:`omnibias.geometry.{torch,jax}.ops.pullback_metric`) drops straight in. This locks
down the seam the roadmap targets ("JetSubspaceTensor respecting g = J^T h J"):

1. torch drop-in descends ``f`` in the pullback geometry down to the minimiser;
2. the jax functional step does the same;
3. the two backends produce a bit-identical natural-gradient direction on a shared point.

The immersion is a paraboloid ``phi(x) = (x0, x1, 1/2 (x0^2 + x1^2))``, whose pullback metric
``g(x) = [[1 + x0^2, x0 x1], [x0 x1, 1 + x1^2]]`` is SPD everywhere and genuinely
coordinate-dependent (not the identity), so the natural gradient differs from the Euclidean
one. All in float64.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import torch.nn as nn  # noqa: E402
from omnibias.geometry import ChartSpec  # noqa: E402
from omnibias.geometry.jax import ops as jgeo  # noqa: E402
from omnibias.geometry.torch import ops as tgeo  # noqa: E402
from omnibias.jax.optim import natural_gradient_direction as jax_ngd  # noqa: E402
from omnibias.jax.optim import natural_gradient_step  # noqa: E402
from omnibias.torch.optim import NaturalGradient  # noqa: E402
from omnibias.torch.optim import natural_gradient_direction as torch_ngd  # noqa: E402

torch.set_default_dtype(torch.float64)

TARGET = (0.3, -0.2)
START = (1.0, 0.8)


def _paraboloid_phi(xp):  # type: ignore[no-untyped-def]
    def phi(x):  # (2,) -> (3,)
        return xp.stack([x[0], x[1], 0.5 * (x[0] ** 2 + x[1] ** 2)])

    return phi


def _chart(xp):  # type: ignore[no-untyped-def]
    return ChartSpec(phi=_paraboloid_phi(xp), domain_dim=2, ambient_dim=3, name="paraboloid")


def _np(x):  # type: ignore[no-untyped-def]
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def test_pullback_natural_gradient_direction_cross_backend() -> None:
    """Same point + gradient: pullback metric and its natural-gradient step agree torch <-> jax."""
    x = np.array([[0.7, -0.4]])  # a single (batched) domain point
    grad = np.array([0.5, 1.3])
    g_t = tgeo.pullback_metric(torch.as_tensor(x), _chart(torch))[0]
    g_j = jgeo.pullback_metric(jnp.asarray(x), _chart(jnp))[0]
    # the raw pullback metric is itself bit-identical across backends
    assert np.allclose(_np(g_t), _np(g_j), atol=1e-12)
    d_t = _np(torch_ngd(g_t, torch.as_tensor(grad), damping=0.0))
    d_j = _np(jax_ngd(g_j, jnp.asarray(grad), damping=0.0))
    assert np.allclose(d_t, d_j, atol=1e-10)


def test_torch_natural_gradient_descends_in_pullback_metric() -> None:
    """The torch drop-in, preconditioned by the pullback metric, reaches the minimiser."""
    chart = _chart(torch)
    target = torch.as_tensor(TARGET, dtype=torch.float64)

    class _Domain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.theta = nn.Parameter(torch.as_tensor(START, dtype=torch.float64))

        def loss(self) -> torch.Tensor:
            return 0.5 * ((self.theta - target) ** 2).sum()

    model = _Domain()

    def metric(flat: torch.Tensor) -> torch.Tensor:
        return tgeo.pullback_metric(flat.reshape(1, -1), chart)[0]

    opt = NaturalGradient(model.parameters(), metric=metric, lr=1.0, damping=1e-10)
    f_start = float(model.loss().detach())
    for _ in range(120):
        opt.step(model.loss)
    assert float(model.loss().detach()) < f_start * 1e-8
    assert torch.allclose(model.theta.detach(), target, atol=1e-5)


def test_jax_natural_gradient_step_descends_in_pullback_metric() -> None:
    """The jax functional Fisher-scoring step reaches the minimiser in the pullback geometry."""
    chart = _chart(jnp)
    target = jnp.asarray(TARGET)
    theta = jnp.asarray(START)
    f_start = 0.5 * float(jnp.sum((theta - target) ** 2))
    for _ in range(200):
        grad = theta - target
        metric = jgeo.pullback_metric(theta.reshape(1, -1), chart)[0]
        theta = natural_gradient_step(theta, grad, metric, learning_rate=1.0, damping=1e-10)
    f_final = 0.5 * float(jnp.sum((theta - target) ** 2))
    assert f_final < f_start * 1e-8
    assert jnp.allclose(theta, target, atol=1e-5)
