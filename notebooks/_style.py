# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared plotting style + tiny helpers for the omnibias notebook gallery.

Import at the top of every notebook:

    from _style import set_style, PRIMARY, ACCENT, GOOD
    set_style()
"""

from __future__ import annotations

import matplotlib.pyplot as plt

PRIMARY = "#3b6fb6"
ACCENT = "#d1495b"
GOOD = "#2a9d8f"
WARM = "#e09f3e"
INK = "#22223b"


def set_style() -> None:
    """Apply the consistent omnibias notebook look."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.4),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#dfe3e8",
            "grid.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "font.size": 12,
            "legend.frameon": False,
            "lines.linewidth": 2.2,
        }
    )
