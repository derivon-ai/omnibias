# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet distillation (torch; theory 09-19).

A student matches a teacher's N-jet. The founding bias collapse
(``delta -> 0``) supplies the derivatives. Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the
two. Not ImageNet KD.
"""

from __future__ import annotations

from omnibias.core import jet_token as core

import torch
from torch import Tensor

DISCLAIMER = core.DISCLAIMER
JetDistillConfig = core.JetDistillConfig
honesty_payload = core.honesty_payload


def jet_distill_loss(
    student: Tensor,
    teacher: Tensor,
    *,
    config: JetDistillConfig | None = None,
) -> Tensor:
    cfg = JetDistillConfig() if config is None else config
    weights = torch.tensor(cfg.weights, dtype=student.dtype, device=student.device)
    return torch.sum(weights * (student - teacher) ** 2)


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
