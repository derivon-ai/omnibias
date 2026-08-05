# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch backend for the omnibias-fields operator surface.

Exposes the functional op surface (:mod:`omnibias.fields.torch.ops`) and the
thin dispatch router (:mod:`omnibias.fields.torch._ops_dispatch`) that the
attribute-DSL views forward into via ``state.ops.<fn>(...)``.
"""

from __future__ import annotations

from omnibias.fields.torch import ops

__all__ = ["ops"]
