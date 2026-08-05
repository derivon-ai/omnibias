# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""cbf_filter: feasibility, torch<->jax parity, differentiability."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.control import CBFSpec, FilterSchedule


def _corridor_states(n=48, seed=1):
    """Recoverable approach states below a unit disc at the origin (small speed)."""
    rng = np.random.default_rng(seed)
    px = rng.uniform(-1.0, 1.0, n)
    py = rng.uniform(-3.0, -2.6, n)
    v = 0.1 * rng.standard_normal((n, 2))
    return np.stack([px, py, v[:, 0], v[:, 1]], axis=1)


def _rows_jax(X):
    import jax.numpy as jnp
    import omnibias.control.jax as cj

    def f(x):
        return jnp.array([x[2], x[3], 0.0, 0.0])

    def g(x):
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    def bar(x):
        return x[0] ** 2 + x[1] ** 2 - 1.0

    spec = CBFSpec(gains=(2.0, 2.0), a_max=2.5)
    return cj.control_affine_cbf_rows(f, g, bar, jnp.asarray(X), spec)


def test_feasible_input_stays_feasible():
    """If ``a_nom`` is already feasible (0 on the corridor), the projection returns it."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import omnibias.control.jax as cj

    X = _corridor_states()
    G, h = _rows_jax(X)
    a0 = jnp.zeros((X.shape[0], 2))
    a = cj.cbf_filter(a0, G, h, FilterSchedule())
    # 0 is strictly interior here -> residual well below 0 and the point does not move.
    assert float(cj.cbf_residual(G, h, a).max()) < 0.0
    assert float(jnp.max(jnp.abs(a))) < 1e-6


def test_projection_near_feasible():
    """A large (infeasible) ``a_nom`` is projected to a near-feasible action."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import omnibias.control.jax as cj

    X = _corridor_states()
    G, h = _rows_jax(X)
    rng = np.random.default_rng(3)
    a_nom = jnp.asarray(rng.standard_normal((X.shape[0], 2)) * 4.0)
    before = float(cj.cbf_residual(G, h, a_nom).max())
    a = cj.cbf_filter(a_nom, G, h, FilterSchedule())
    after = float(cj.cbf_residual(G, h, a).max())
    assert before > 0.5                      # the raw action is infeasible
    assert after < 5e-2                       # the exterior penalty drives it ~feasible
    assert after < 0.1 * before               # a >10x reduction in violation


def test_torch_jax_parity():
    import jax

    jax.config.update("jax_enable_x64", True)
    import numpy as _np
    import torch

    torch.set_default_dtype(torch.float64)
    import jax.numpy as jnp
    import omnibias.control.jax as cj
    import omnibias.control.torch as ct

    X = _corridor_states()
    G, h = _rows_jax(X)
    rng = _np.random.default_rng(5)
    a_np = rng.standard_normal((X.shape[0], 2)) * 3.0
    aJ = cj.cbf_filter(jnp.asarray(a_np), G, h, FilterSchedule())
    aT = ct.cbf_filter(
        torch.tensor(a_np), torch.tensor(_np.asarray(G)), torch.tensor(_np.asarray(h)),
        FilterSchedule(),
    )
    assert _np.max(_np.abs(_np.asarray(aJ) - aT.detach().numpy())) < 1e-10


def test_differentiable_through_filter():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import omnibias.control.jax as cj

    X = _corridor_states(n=8)
    G, h = _rows_jax(X)

    def loss(a_nom):
        a = cj.cbf_filter(a_nom, G, h, FilterSchedule.fast())
        return jnp.sum(a ** 2)

    a_nom = jnp.asarray(np.random.default_rng(7).standard_normal((8, 2)))
    grad = jax.grad(loss)(a_nom)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


def test_schedule_validation():
    with pytest.raises(ValueError):
        FilterSchedule(mu0=0.0)
    with pytest.raises(ValueError):
        FilterSchedule(safety=1.5)
    with pytest.raises(ValueError):
        CBFSpec(gains=(1.0, 2.0, 3.0))         # relative degree 3 unsupported
    with pytest.raises(ValueError):
        CBFSpec(gains=(2.0,), a_max=-1.0)
