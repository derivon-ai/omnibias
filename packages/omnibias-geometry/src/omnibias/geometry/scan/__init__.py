# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Chart-coordinate bias scan (theory 02-08, gated).

Arc-length spacing uses the pullback metric ``g = J^T h J``. Discrete
``C_L`` orbit, not SO(2) / SO(3). Exact steering is gaussian-family only.
"""

from __future__ import annotations

__all__ = ["chart_scan"]


def __getattr__(name: str) -> object:
    if name == "chart_scan":
        from omnibias.geometry.scan.torch import chart_scan

        return chart_scan
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
