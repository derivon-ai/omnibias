# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX score twin for theory 09-21."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from omnibias.score.jax.score_matching import exact_div_neg_id, worked_example

jax.config.update("jax_enable_x64", True)


def test_g1_jax() -> None:
    ex = worked_example()
    assert abs(ex["div"] + 1.0) < 1e-12
    assert exact_div_neg_id(jnp.asarray(0.0)) == -1.0
