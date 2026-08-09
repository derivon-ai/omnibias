# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Empirical lambda-instability laws (initialization only).

DeepMind (arXiv:2509.14185, Fig. 2e) reported asymptotic fits

* IPM: ``lam_n ~ 1 / (1.1459 n + 0.9723)``
* Boussinesq: ``lam_n ~ 1 / (1.4187 n + 1.0863) + 1``

These are **initialization hints**, never reference values. The anti-circularity
guard :func:`assert_not_reference_value` exists so formula outputs cannot be
written into benchmark gates or certificates as validated measurements.
"""

from __future__ import annotations

from typing import Literal

EquationFamily = Literal["ipm", "boussinesq", "ccf"]


def predict_lambda_init(
    order: int,
    *,
    family: EquationFamily,
) -> float:
    """Return an initialization ``lambda`` for instability order ``n``.

    ``order=0`` is the stable solution; ``order>=1`` is the n-th unstable.
    For CCF no published asymptotic law exists yet; raise for ``family='ccf'``.
    """
    n = int(order)
    if n < 0:
        raise ValueError(f"order must be >= 0, got {n}")
    if family == "ipm":
        return 1.0 / (1.1459 * n + 0.9723)
    if family == "boussinesq":
        return 1.0 / (1.4187 * n + 1.0863) + 1.0
    if family == "ccf":
        raise ValueError(
            "CCF has no published asymptotic lambda law yet; pass lam_init explicitly"
        )
    raise ValueError(f"unknown family {family!r}")


def assert_not_reference_value(tag: str) -> None:
    """Anti-circularity: formula outputs must never be stored as references.

    Call sites that write benchmark / certificate payloads should invoke this
    whenever a value originated from :func:`predict_lambda_init`.
    """
    raise RuntimeError(
        f"anti-circularity: refusing to treat empirical-law output as a "
        f"reference/validated value (tag={tag!r}). Use measured residuals / "
        f"funnel-validated lambda instead."
    )


__all__ = [
    "EquationFamily",
    "assert_not_reference_value",
    "predict_lambda_init",
]
