# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX backend for the omnibias-fields operator surface.

Exposes the functional op surface (:mod:`omnibias.fields.jax.ops`) and the
thin dispatch router (:mod:`omnibias.fields.jax._ops_dispatch`) that the
attribute-DSL views forward into via ``state.ops.<fn>(...)``.
"""

from __future__ import annotations

from omnibias.fields.jax import ops

__all__ = ["ops"]
