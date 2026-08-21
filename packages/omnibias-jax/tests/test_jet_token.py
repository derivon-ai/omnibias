# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX jet-token / jet-distill twins (09-02 / 09-19)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from omnibias.core.jet_token import worked_example
from omnibias.jax.architectures.jet_token import worked_compose_jet
from omnibias.jax.jet_distill import jet_distill_loss, recover_tanh_scale

jax.config.update("jax_enable_x64", True)


def test_g1_matches_compose_jet() -> None:
    out = worked_compose_jet()
    ex = worked_example()
    assert abs(float(out[0]) - ex["value"]) < 1e-12
    assert abs(float(out[1]) - ex["deriv"]) < 1e-12
    rec = recover_tanh_scale(0.5)
    assert rec["loss"] < 1e-12
    loss = jet_distill_loss(jnp.asarray([0.0, 0.5]), jnp.asarray([0.0, 1.0]))
    assert abs(float(loss) - 0.25) < 1e-12
