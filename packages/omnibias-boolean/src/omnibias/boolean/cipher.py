# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""S-box cryptanalysis (public surface).

Thin re-export of the exact :mod:`omnibias.boolean._core.cipher` primitives:
the difference-distribution and linear-approximation tables, differential
uniformity, linearity / nonlinearity, algebraic degree, and the exact
higher-order Boolean derivative used in higher-order differential attacks.

Each S-box component (:meth:`SBox.component`) is an ordinary Boolean function, so
the rigorous interval bias bounds in :mod:`omnibias.boolean._core.verified`
(``linear_bias_iv``, ``differential_bias_iv``, ...) apply directly to a
differentiable or noisy S-box relaxation.
"""

from __future__ import annotations

from omnibias.boolean._core.cipher import (
    SBox,
    difference_distribution_table,
    differential_uniformity,
    directional_derivative,
    higher_order_derivative,
    linear_approximation_table,
    linearity,
    nonlinearity,
    sbox_algebraic_degree,
    sbox_directional_derivative,
    sbox_from_table,
    sbox_higher_order_derivative,
)

__all__ = [
    "SBox",
    "difference_distribution_table",
    "differential_uniformity",
    "directional_derivative",
    "higher_order_derivative",
    "linear_approximation_table",
    "linearity",
    "nonlinearity",
    "sbox_algebraic_degree",
    "sbox_directional_derivative",
    "sbox_from_table",
    "sbox_higher_order_derivative",
]
