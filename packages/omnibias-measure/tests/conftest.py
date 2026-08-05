# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Enable float64 in jax / torch for the measure cross-backend parity tests."""

from __future__ import annotations

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ModuleNotFoundError:  # pragma: no cover
    pass

try:
    import torch

    torch.set_default_dtype(torch.float64)
except ModuleNotFoundError:  # pragma: no cover
    pass
