# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Torch backend for the sparse-recovery front-end: the differentiable ``l_p`` relaxation."""

from __future__ import annotations

from omnibias.discrete.sparse.torch.relaxation import sparse_relaxation

__all__ = ["sparse_relaxation"]
