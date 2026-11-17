# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The numerical-precision policy of planned derivative programs, independent of any tensor framework.

Three things decide how accurate a planned derivative is: the dtype of the inputs, the precision the hardware gives
matrix products of that dtype, and the derivative order (each order divides by another power of the input scale and
amplifies rounding). The policy fixes the second and checks the first against the third:

* ``matmul="highest"`` (the default) computes float32 products in float32 on every device. Matrix engines on recent GPUs
  otherwise give float32 products a 10-bit mantissa (TF32), which is an error of about ``1e-3`` per product before any
  derivative amplifies it. ``"tf32"`` opts in to that speed; ``"default"`` leaves the framework's own setting.
* ``float16`` and ``bfloat16`` inputs are refused for derivative order two and above (the Laplacian and everything
  built on it), because their 8- and 11-bit mantissas leave no digits after two derivatives, unless
  ``allow_low_precision=True``. First-order quantities (divergence, ``log |det(I + J)|``, the first-order Gram rows) are
  allowed and reported.

The policy is applied by the JAX and PyTorch front ends and reported by ``explain()``; the measured float32 error tables are
in ``docs/precision.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["MATMUL_MODES", "PrecisionPolicy", "request_order", "resolve_precision"]

MATMUL_MODES = ("highest", "tf32", "default")
_LOW = {"float16", "bfloat16", "half"}


@dataclass(frozen=True)
class PrecisionPolicy:
    """``matmul`` is ``"highest"``, ``"tf32"`` or ``"default"``; ``allow_low_precision`` admits half-precision inputs."""

    matmul: str = "highest"
    allow_low_precision: bool = False

    def __post_init__(self) -> None:
        if self.matmul not in MATMUL_MODES:
            raise ValueError(f"matmul precision must be one of {MATMUL_MODES}, got {self.matmul!r}")

    @property
    def jax_name(self) -> str | None:
        """The ``jax.default_matmul_precision`` value (``None`` leaves JAX's default)."""
        return {"highest": "highest", "tf32": "tensorfloat32", "default": None}[self.matmul]

    def check(self, dtype: Any, order: int, *, what: str = "this request") -> list[str]:
        """Raise for a half-precision ``dtype`` at ``order >= 2`` unless allowed; return the notes to report."""
        name = _dtype_name(dtype)
        notes: list[str] = []
        if name in _LOW:
            if order >= 2 and not self.allow_low_precision:
                raise ValueError(
                    f"{name} inputs are refused for derivative order {order} ({what}): the mantissa leaves no digits "
                    "after two derivatives; pass allow_low_precision=True to run it anyway, or compute_dtype='float64'"
                )
            notes.append(
                f"{name} inputs at derivative order {order}: accuracy is not bounded by this library"
            )
        if name == "float32" and self.matmul == "tf32":
            notes.append(
                "float32 products run with TF32 matrix engines where the device has them (about 1e-3 each)"
            )
        return notes

    def to_config(self) -> dict[str, Any]:
        return {"matmul": self.matmul, "allow_low_precision": self.allow_low_precision}


def _dtype_name(dtype: Any) -> str:
    text = str(getattr(dtype, "name", dtype))
    return text.replace("torch.", "")


def request_order(kind: str, power: int = 1, demand_order: int | None = None) -> int:
    """The highest derivative order a request differentiates the model to."""
    if demand_order is not None:
        return int(demand_order)
    if kind == "laplacian":
        return 2 * int(power)
    return 1  # divergence, log-determinant and solves with the Jacobian


def resolve_precision(
    precision: str | PrecisionPolicy | None, allow_low_precision: bool = False
) -> PrecisionPolicy:
    """A policy from ``None`` (the default), a matmul mode name or a policy."""
    if isinstance(precision, PrecisionPolicy):
        if allow_low_precision and not precision.allow_low_precision:
            return PrecisionPolicy(precision.matmul, True)
        return precision
    return PrecisionPolicy(
        "highest" if precision is None else str(precision), bool(allow_low_precision)
    )
