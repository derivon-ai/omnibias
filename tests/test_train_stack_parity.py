# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch / JAX twins of the 08-01 stack are bit-identical on the same floats."""

from __future__ import annotations

import pytest
from omnibias.core.train_stack import TrainStackConfig, recommended_stack_step
from omnibias.core.weight_loss_jet import one_layer_forward

_HIDDEN = 1
_DIM = 1
_XS = ((0.4,), (-0.3,), (0.8,), (-0.6,))
_TEACHER = [0.0, 1.2, 0.1, 0.7]
_YS = tuple(one_layer_forward(x, _TEACHER, _HIDDEN, _DIM, "tanh") for x in _XS)
_STUDENT = [0.05, 0.9, 0.0, 0.4]


def test_stack_step_parity() -> None:
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.jax.train_stack import recommended_stack_step as jax_step
    from omnibias.torch.train_stack import recommended_stack_step as torch_step

    torch.set_default_dtype(torch.float64)
    cfg = TrainStackConfig(use_kantorovich=False)
    core_theta, core_report = recommended_stack_step(
        _XS, _YS, _STUDENT, _HIDDEN, _DIM, config=cfg
    )
    t_theta, t_report = torch_step(
        torch.tensor(_XS, dtype=torch.float64),
        torch.tensor(_YS, dtype=torch.float64),
        torch.tensor(_STUDENT, dtype=torch.float64),
        _HIDDEN,
        _DIM,
        config=cfg,
    )
    j_theta, j_report = jax_step(
        jnp.asarray(_XS, dtype=jnp.float64),
        jnp.asarray(_YS, dtype=jnp.float64),
        jnp.asarray(_STUDENT, dtype=jnp.float64),
        _HIDDEN,
        _DIM,
        config=cfg,
    )
    assert t_theta == core_theta
    assert j_theta == core_theta
    assert t_report.step == core_report.step
    assert j_report.step == core_report.step
    assert t_report.loss1 == core_report.loss1
    assert j_report.loss1 == core_report.loss1
