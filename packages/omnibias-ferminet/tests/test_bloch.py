# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bloch/twist mixed partials via multivariate jets."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.ferminet.bloch import bloch_twist_mixed_partials  # noqa: E402


def test_twist_mixed_partial_is_finite() -> None:
    partials = bloch_twist_mixed_partials(
        jnp.array(0.3),
        jnp.array(0.2),
        weight_x=1.2,
        weight_twist=-0.4,
        order=2,
    )
    mixed = partials[(1, 1)]
    assert jnp.isfinite(mixed)
    # Mixed partial of sigma(ax+b theta) is a*b*sigma''.
    assert mixed != 0.0
