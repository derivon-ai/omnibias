# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Reproducible limitation study: factoring as a soft-gate Boolean system.

A deliberately *negative* result. Encoding ``N = p * q`` as a differentiable
soft-gate :class:`~omnibias.boolean.torch.ops.solver.BooleanSystem` and annealing
``beta -> inf`` does **not** shrink the factoring search space: the success rate
collapses as the factor bit-length grows (the "accuracy cliff"). This is an
honest demonstration of the method's boundary -- *not* an attack on RSA. See
``docs/cookbook/rsa-limitation.md``.
"""

from __future__ import annotations

from examples.boolean_limits.multiplier import (
    bits_to_factors,
    factor_system,
    semiprimes_for_width,
)

__all__ = ["bits_to_factors", "factor_system", "semiprimes_for_width"]
