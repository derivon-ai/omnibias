# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Shared (backend-agnostic) constants for the annealed relaxation.

The relaxation parametrises ``x = sigmoid(beta theta)``; at ``theta = 0`` the soft
assignment sits at the ``x = 1/2`` saddle where the energy gradient can vanish (e.g.
max-cut). A tiny **deterministic** perturbation of ``theta`` breaks that symmetry
identically for both backends, so the torch and jax twins stay bit-identical.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: Amplitude of the deterministic symmetry-breaking perturbation of ``theta_0``.
INIT_THETA_SCALE = 1e-2


def initial_theta(n: int) -> NDArray[np.float64]:
    """Deterministic initial ``theta_0`` of length ``n`` (fixed seed, backend-shared)."""
    rng = np.random.default_rng(0)
    return (INIT_THETA_SCALE * rng.standard_normal(n)).astype(float)


__all__ = ["INIT_THETA_SCALE", "initial_theta"]
