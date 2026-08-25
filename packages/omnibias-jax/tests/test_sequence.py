# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX causal-transverse unit checks (theory 05-02)."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.core.sequence import causal_transverse_taps
from omnibias.jax.sequence import (
    apply_causal_transverse_filter,
    causal_transverse_filter,
    init_leaky_integrator,
)
from omnibias.jax.sequence import (
    causal_transverse_taps as jax_taps,
)


def test_init_leaky_taps_match_core() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    params = init_leaky_integrator(0.95, width=8)
    taps = np.asarray(
        jax_taps(
            params["coeff"],
            params["alpha"],
            params["tau"],
            order=0,
            width=8,
        )
    )
    expected = np.asarray(
        causal_transverse_taps(
            width=8,
            coeff=float(params["coeff"]),
            alpha=float(params["alpha"]),
            tau=float(params["tau"]),
            order=0,
        )
    )
    np.testing.assert_allclose(taps, expected, rtol=0.0, atol=1e-12)


def test_impulse_recovers_taps() -> None:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    taps = jax_taps(jnp.asarray(0.4), jnp.asarray(0.2), jnp.asarray(0.0), order=0, width=5)
    impulse = jnp.zeros((1, 7))
    impulse = impulse.at[0, 0].set(1.0)
    out = np.asarray(apply_causal_transverse_filter(impulse, taps, jnp.asarray(0.0)))
    np.testing.assert_allclose(out[0, :5], np.asarray(taps), rtol=0.0, atol=1e-12)


def test_filter_wrapper_matches_apply() -> None:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    params = init_leaky_integrator(0.9, width=6)
    x = jnp.linspace(-0.3, 0.4, 6)[None, :]
    wrapped = np.asarray(causal_transverse_filter(x, params))
    taps = jax_taps(
        params["coeff"],
        params["alpha"],
        params["tau"],
        order=0,
        width=6,
    )
    direct = np.asarray(apply_causal_transverse_filter(x, taps, params["bias"]))
    np.testing.assert_allclose(wrapped, direct, rtol=0.0, atol=0.0)


def test_rejects_bad_order_width() -> None:
    import jax.numpy as jnp

    with pytest.raises(ValueError, match="order"):
        jax_taps(jnp.asarray(1.0), jnp.asarray(0.2), jnp.asarray(0.0), order=-1, width=4)
    with pytest.raises(ValueError, match="width"):
        jax_taps(jnp.asarray(1.0), jnp.asarray(0.2), jnp.asarray(0.0), order=0, width=0)
