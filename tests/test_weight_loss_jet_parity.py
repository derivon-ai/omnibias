# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch / JAX twins of the one-layer weight-space loss jet are bit-identical."""

from __future__ import annotations

import pytest
from omnibias.core.weight_loss_jet import one_layer_loss_jet as core_jet
from omnibias.core.weight_loss_jet import pack_one_layer_params

_XS = ((0.5, -0.25), (0.0, 0.75))
_YS = (0.15, -0.05)
_THETA = pack_one_layer_params(
    0.05,
    (0.8, -0.4),
    (0.1, -0.2),
    ((0.3, -0.1), (0.2, 0.5)),
)
_DIR = pack_one_layer_params(
    0.0,
    (0.2, 0.0),
    (0.0, -0.1),
    ((1.0, 0.0), (0.0, 0.5)),
)


def test_torch_jax_core_loss_jet_parity() -> None:
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.jax.weight_loss_jet import one_layer_loss_jet as jax_jet
    from omnibias.torch.weight_loss_jet import one_layer_loss_jet as torch_jet

    torch.set_default_dtype(torch.float64)
    expected = core_jet(
        _XS,
        _YS,
        _THETA,
        _DIR,
        order=3,
        hidden=2,
        dim=2,
        activation="tanh",
    )
    t_out = torch_jet(
        torch.tensor(_XS, dtype=torch.float64),
        torch.tensor(_YS, dtype=torch.float64),
        torch.tensor(_THETA, dtype=torch.float64),
        torch.tensor(_DIR, dtype=torch.float64),
        order=3,
        hidden=2,
        dim=2,
        activation="tanh",
    )
    j_out = jax_jet(
        jnp.asarray(_XS, dtype=jnp.float64),
        jnp.asarray(_YS, dtype=jnp.float64),
        jnp.asarray(_THETA, dtype=jnp.float64),
        jnp.asarray(_DIR, dtype=jnp.float64),
        order=3,
        hidden=2,
        dim=2,
        activation="tanh",
    )
    assert t_out == expected
    assert j_out == expected
