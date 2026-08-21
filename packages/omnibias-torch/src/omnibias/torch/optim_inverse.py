# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Newton-on-input inverse design (torch; theory 09-22).

Exact ``sigma'`` from founding bias collapse (``delta -> 0``).
Temperature collapse (``beta -> inf``, feasibility) does not
appear. do not conflate the two. Not 08-03. Not a global inverse.
"""

from __future__ import annotations

from omnibias.core import inverse_design as core

from torch import Tensor

DISCLAIMER = core.DISCLAIMER
InverseDesignConfig = core.InverseDesignConfig
InverseDesignReport = core.InverseDesignReport
honesty_payload = core.honesty_payload


def invert_input(
    f: object | None,
    y: Tensor | float,
    x0: Tensor | float,
    *,
    config: InverseDesignConfig | None = None,
    family: str = "tanh_scale",
) -> InverseDesignReport:
    """Newton-on-``x``. Raises if ``|sigma'|`` is tiny or ``y`` is saturated."""
    target = float(y.reshape(-1)[0].detach()) if isinstance(y, Tensor) else float(y)
    start = float(x0.reshape(-1)[0].detach()) if isinstance(x0, Tensor) else float(x0)
    return core.invert_input(f, target, start, config=config, family=family)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "InverseDesignConfig",
    "InverseDesignReport",
    "honesty_payload",
    "invert_input",
    "worked_example",
]
