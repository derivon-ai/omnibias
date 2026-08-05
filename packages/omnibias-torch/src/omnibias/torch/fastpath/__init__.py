# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form fast-path kernels for the bias-collapse limit.

Per-activation kernel modules (:mod:`eulerian` for sigmoid,
:mod:`hermite` for Gaussian, :mod:`legendre` for tanh) export functions
of the form ``sigma_nth_derivative(z, n) -> sigma^(n)(z)``, evaluated in
closed form via the appropriate polynomial family.

The :mod:`dispatch` module exposes the literal multi-bias forward and
the collapsed-form forward used by :class:`OperatorMultiBiasUnit` and
:class:`OperatorBlock`.
"""

from omnibias.torch.fastpath.dispatch import (
    bias_spread,
    is_collapsed,
    multibias_collapsed_forward,
    multibias_forward,
    multibias_literal_forward,
)

__all__ = [
    "bias_spread",
    "is_collapsed",
    "multibias_collapsed_forward",
    "multibias_forward",
    "multibias_literal_forward",
]
