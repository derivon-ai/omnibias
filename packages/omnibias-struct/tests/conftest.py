# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Shared pytest config: enable float64 JAX so the torch <-> jax twins are bit-identical."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

try:  # float64 parity for the torch <-> jax soft-DP twins
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:  # pragma: no cover - jax optional
    pass


@pytest.fixture(autouse=True)
def _float64_torch() -> Iterator[None]:
    """Restore float64 per test even when another package changes the global dtype."""
    try:
        import torch
    except ImportError:
        yield
        return
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)
