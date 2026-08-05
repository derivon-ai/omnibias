# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch fractional-derivative operator surface.

The ``grunwald_letnikov`` / ``riemann_liouville`` / ``caputo`` /
``spectral_fractional`` ops are grid-based numerical approximations (non-local,
**not** closed form). ``fractional_derivative`` / ``mlp_fractional_derivative``
(from :mod:`~omnibias.fractional.torch.ops.analytic`) are the closed-form,
jet-based operators on the analytic-function class.
"""

from __future__ import annotations

from omnibias.fractional.torch.ops.activation import (
    ACTIVATION_FRACTIONAL,
    activation_fractional_derivative,
    cosh_fractional,
    exp_fractional,
    sinh_fractional,
)
from omnibias.fractional.torch.ops.analytic import (
    fractional_derivative,
    mlp_fractional_derivative,
    piecewise_fractional_derivative,
)
from omnibias.fractional.torch.ops.fractional import (
    caputo,
    grunwald_letnikov,
    riemann_liouville,
    spectral_fractional,
)
from omnibias.fractional.torch.ops.special import (
    lerch,
    lower_incomplete_gamma,
    mittag_leffler,
    polylog,
)
from omnibias.fractional.torch.ops.spectral import (
    spectral_fractional_laplacian,
    tukey_window,
    windowed_spectral_fractional,
)

__all__ = [
    "ACTIVATION_FRACTIONAL",
    "activation_fractional_derivative",
    "caputo",
    "cosh_fractional",
    "exp_fractional",
    "fractional_derivative",
    "grunwald_letnikov",
    "lerch",
    "lower_incomplete_gamma",
    "mittag_leffler",
    "mlp_fractional_derivative",
    "piecewise_fractional_derivative",
    "polylog",
    "riemann_liouville",
    "sinh_fractional",
    "spectral_fractional",
    "spectral_fractional_laplacian",
    "tukey_window",
    "windowed_spectral_fractional",
]
