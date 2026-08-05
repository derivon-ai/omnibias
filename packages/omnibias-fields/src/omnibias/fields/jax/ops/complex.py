# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Complex (Wirtinger) calculus on a split-real field (jax).

Bit-identical twin of :mod:`omnibias.fields.torch.ops.complex`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax import Array
from omnibias.fields.jax.ops.basic import derivative

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def _partials(
    state: FieldState, re_name: str, im_name: str, real_axis: int | str, imag_axis: int | str,
) -> tuple[Array, Array, Array, Array]:
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
) -> tuple[Array, Array]:
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
) -> tuple[Array, Array]:
    r""":math:`\partial_{\bar z} f` as a ``(real, imag)`` pair (zero if holomorphic)."""
    fr_x, fr_y, fi_x, fi_y = _partials(state, re_name, im_name, real_axis, imag_axis)
    return 0.5 * (fr_x - fi_y), 0.5 * (fi_x + fr_y)


__all__ = ["dz", "dzbar"]
