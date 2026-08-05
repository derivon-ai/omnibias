# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""PyTorch backend for omnibias-qubo."""

from __future__ import annotations

from omnibias.qubo.torch.relaxation import qubo_relaxation

__all__ = ["qubo_relaxation"]
