# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""JAX backend for omnibias-tab: the bit-identical functional forward twin."""

from __future__ import annotations

from omnibias.tab.jax.arrangement import arrangement_forward, boosted_forward
from omnibias.tab.jax.jet import (
    extract_arrangement_jet,
    extract_tree_jet,
    extract_tree_jet_directional,
)
from omnibias.tab.jax.model import (
    fit_natural_gradient,
    forward,
    forward_arrays,
    natural_gradient_step,
)

__all__ = [
    "arrangement_forward",
    "boosted_forward",
    "extract_arrangement_jet",
    "extract_tree_jet",
    "extract_tree_jet_directional",
    "fit_natural_gradient",
    "forward",
    "forward_arrays",
    "natural_gradient_step",
]
