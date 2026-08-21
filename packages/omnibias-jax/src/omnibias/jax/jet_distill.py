# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet distillation (jax; theory 09-19).

A student matches a teacher's N-jet. The founding bias collapse
(``delta -> 0``) supplies the derivatives. Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the
two. Not ImageNet KD.
"""

from __future__ import annotations

from omnibias.core import jet_token as core

import jax.numpy as jnp
from jax import Array

DISCLAIMER = core.DISCLAIMER
JetDistillConfig = core.JetDistillConfig
honesty_payload = core.honesty_payload


def jet_distill_loss(
    student: Array,
    teacher: Array,
    *,
    config: JetDistillConfig | None = None,
) -> Array:
    cfg = JetDistillConfig() if config is None else config
    weights = jnp.asarray(cfg.weights, dtype=student.dtype)
    return jnp.sum(weights * (student - teacher) ** 2)


def recover_tanh_scale(a0: float = 0.5) -> dict[str, float]:
    return core.recover_tanh_scale(a0)


def ssl_flip_residual(x: float) -> float:
    return core.ssl_flip_residual(x)


__all__ = [
    "DISCLAIMER",
    "JetDistillConfig",
    "honesty_payload",
    "jet_distill_loss",
    "recover_tanh_scale",
    "ssl_flip_residual",
]
