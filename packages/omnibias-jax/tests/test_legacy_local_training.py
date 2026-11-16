# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
from __future__ import annotations

import jax.numpy as jnp

from omnibias.core.local_jet import LocalJetConfig
from omnibias.jax.train_local import local_jet_step


def test_legacy_local_training_module_runs_one_small_sweep() -> None:
    layers = [(jnp.asarray([[0.2]]), jnp.asarray([0.1]), "tanh")]
    updated, report = local_jet_step(
        layers,
        jnp.asarray([0.3]),
        target=jnp.asarray([0.0]),
        config=LocalJetConfig(n_directions=1, jet_order=1, damping=1e-2),
    )

    assert len(updated) == 1
    assert report.n_params == 2
    assert report.residual_norms[0] >= 0.0
    assert report.greedy_only_claimed_optimal is False
