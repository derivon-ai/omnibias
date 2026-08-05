# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""PyTorch backend for the MaxSAT front-end: the differentiable relaxation."""

from __future__ import annotations

from omnibias.discrete.maxsat.torch.relaxation import maxsat_relaxation

__all__ = ["maxsat_relaxation"]
