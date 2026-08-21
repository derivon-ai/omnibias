# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""q-OMBU hybrid (torch; theory 09-15).

Named ``q -> 1`` is not founding bias collapse (``delta -> 0``) and
not temperature collapse (``beta -> inf``, feasibility). Ordinary
``sigma`` cells still use founding bias collapse. do not conflate
the two.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.qcalculus._core import hybrid as core
from torch import Tensor

DISCLAIMER = core.DISCLAIMER
QOMBUConfig = core.QOMBUConfig
honesty_payload = core.honesty_payload


def q_ombu_forward(
    x: Tensor,
    params: Sequence[float] | None = None,
    *,
    config: QOMBUConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Returns ``(y, limit_residual)``."""
    y, residual = core.q_ombu_forward(float(x.reshape(-1)[0].detach()), params, config=config)
    return x.new_tensor(y), x.new_tensor(residual)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "QOMBUConfig",
    "honesty_payload",
    "q_ombu_forward",
    "worked_example",
]
