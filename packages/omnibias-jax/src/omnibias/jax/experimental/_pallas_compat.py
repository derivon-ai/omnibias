# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pallas API differences between JAX releases that the fused kernels must tolerate.

``jax.experimental.pallas.dot`` was moved to ``jax.experimental.pallas.triton.dot`` and the old name deprecated; releases
before the move have no ``triton.dot``. :func:`triton_dot` uses whichever exists, so one kernel source serves both and the
deprecated alias is never touched on a release that has the new name.
"""

from __future__ import annotations

from typing import Any

__all__ = ["pallas_dot_available", "triton_dot"]


def triton_dot(
    a: Any, b: Any, *, trans_a: bool = False, trans_b: bool = False, allow_tf32: bool | None = None
) -> Any:
    """``a @ b`` of two kernel tiles on the Triton lowering (with optional transposes), float64 accumulation preserved.

    ``allow_tf32=False`` keeps float32 operands at full precision (the default of the lowering is hardware dependent);
    float64 dots on the MMAv2 layout need ``M >= 16``, ``N >= 8`` and ``K >= 16`` per tile.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    fn = getattr(plt, "dot", None) or getattr(pl, "dot", None)
    if fn is None:
        raise NotImplementedError("this Pallas build has neither pallas.triton.dot nor pallas.dot")
    if allow_tf32 is None:
        return fn(a, b, trans_a=trans_a, trans_b=trans_b)
    return fn(a, b, trans_a=trans_a, trans_b=trans_b, allow_tf32=allow_tf32)


def pallas_dot_available() -> bool:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    return hasattr(plt, "dot") or hasattr(pl, "dot")
