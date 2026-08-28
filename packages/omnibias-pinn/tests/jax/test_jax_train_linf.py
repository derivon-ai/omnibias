# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""L-infinity (minimax) discovery trainer (jax): pytree wrapper.

Generic comparison-utility test: a small synthetic linear residual with a
known-exact minimax solution, and a pytree (dict-of-arrays) params round-trip
check. Not a campaign benchmark -- no CCF/IPM residual, champion network, or
committed artifact is touched.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import Array  # noqa: E402
from omnibias.pinn.jax.discovery.train_linf import (  # noqa: E402
    LinfConfig,
    linf_minimax_train,
)


def test_linf_minimax_train_matches_known_exact_solution() -> None:
    """Flat-array params: same known-exact minimax point as the underlying
    ``omnibias.jax.optim.linf_minimax_minimize`` unit test."""
    c = jnp.asarray([1.0, -0.2, 0.4])

    def residual_fn(theta: Array) -> Array:
        return theta[0] + c

    theta0 = jnp.asarray([0.0])
    theta1, history = linf_minimax_train(
        residual_fn, theta0, config=LinfConfig(steps=5, box0=1.0)
    )
    assert float(theta1[0]) == pytest.approx(-0.4, abs=1e-8)
    assert float(history[-1]) == pytest.approx(0.6, abs=1e-8)


def test_linf_minimax_train_accepts_dict_pytree_params() -> None:
    """The ravel/unravel wrapper must round-trip a nested dict of arrays,
    exactly like :func:`omnibias.pinn.jax.discovery.train_gn.gauss_newton_minimize`."""
    targets = jnp.asarray([0.1, 0.1, 0.1, 0.1, 0.9])

    def residual_fn(params: dict[str, Array]) -> Array:
        return jax.nn.sigmoid(params["theta"][0]) - targets

    params0 = {"theta": jnp.asarray([0.0])}
    params1, history = linf_minimax_train(
        residual_fn, params0, config=LinfConfig(steps=50, box0=1.0)
    )
    assert set(params1.keys()) == {"theta"}
    assert params1["theta"].shape == (1,)
    maxr = float(history[-1])
    # Known closed-form Linf optimum for this 4-inlier/1-outlier toy fit is
    # max|r| = 0.40 (sigma(theta*) = 0.5, the midrange of {0.1, 0.9}).
    assert maxr <= 0.45
    # History is non-increasing by the safeguard's construction.
    assert all(float(history[i + 1]) <= float(history[i]) + 1e-12 for i in range(len(history) - 1))


def test_linf_config_defaults_match_dataclass_fields() -> None:
    cfg = LinfConfig()
    assert cfg.steps == 50
    assert cfg.box0 == pytest.approx(1.0)
    assert cfg.accept_tol == pytest.approx(0.0)
