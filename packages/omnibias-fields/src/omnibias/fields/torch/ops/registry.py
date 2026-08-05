# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Re-export of the backend-agnostic ops registry decorator."""

from __future__ import annotations

from omnibias.fields._core.ops_registry import register

__all__ = ["register"]
