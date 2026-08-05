# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Complex (Wirtinger) calculus on a split-real field (torch).

A complex field ``f = f_R + i f_I`` is carried as two real components (the
convention used by ``omnibias-qpinn``). The Wirtinger derivatives

.. math::

    \\partial_z = \\tfrac12(\\partial_x - i\\,\\partial_y), \\qquad
    \\partial_{\\bar z} = \\tfrac12(\\partial_x + i\\,\\partial_y),

are returned as ``(real, imag)`` tensor pairs, built from the closed-form first
derivatives of ``f_R`` and ``f_I``. A holomorphic field satisfies
``dzbar(...) == (0, 0)`` (the Cauchy-Riemann equations).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibias.fields.torch.ops.basic import derivative
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def _partials(
    state: FieldState, re_name: str, im_name: str, real_axis: int | str, imag_axis: int | str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    fr_x = derivative(state, re_name, axis=real_axis, order=1)
    fr_y = derivative(state, re_name, axis=imag_axis, order=1)
    fi_x = derivative(state, im_name, axis=real_axis, order=1)
    fi_y = derivative(state, im_name, axis=imag_axis, order=1)
    return fr_x, fr_y, fi_x, fi_y


def dz(
    state: FieldState,
    re_name: str,
    im_name: str,
    *,
    real_axis: int | str = "x",
    imag_axis: int | str = "y",
) -> tuple[Tensor, Tensor]:
    r""":math:`\partial_z f` as a ``(real, imag)`` pair."""
    fr_x, fr_y, fi_x, fi_y = _partials(state, re_name, im_name, real_axis, imag_axis)
    return 0.5 * (fr_x + fi_y), 0.5 * (fi_x - fr_y)


def dzbar(
    state: FieldState,
    re_name: str,
    im_name: str,
    *,
    real_axis: int | str = "x",
    imag_axis: int | str = "y",
) -> tuple[Tensor, Tensor]:
    r""":math:`\partial_{\bar z} f` as a ``(real, imag)`` pair (zero if holomorphic)."""
    fr_x, fr_y, fi_x, fi_y = _partials(state, re_name, im_name, real_axis, imag_axis)
    return 0.5 * (fr_x - fi_y), 0.5 * (fi_x + fr_y)


__all__ = ["dz", "dzbar"]
