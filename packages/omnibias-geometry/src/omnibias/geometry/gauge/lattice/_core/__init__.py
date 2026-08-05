# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic SU(2) lattice kernels and statistics.

:mod:`kernels` holds the ``xp``-generic deterministic array math (quaternions,
staples, plaquette / Wilson / Polyakov loops, APE smearing, correlators, GEVP)
shared bit-identically by the torch and jax lattice backends; :mod:`stats` holds
the pure-Python jackknife / Creutz-ratio / string-tension helpers.
"""

from __future__ import annotations

from omnibias.geometry.gauge.lattice._core import kernels, stats

__all__ = ["kernels", "stats"]
