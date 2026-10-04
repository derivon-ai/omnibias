# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Hand a sealed SOS certificate to the Lean kernel.

An SOS proof reduces to a *finite, rational* fact: the interval ``LDL^T`` pivots
of the Gram matrix are all positive.  That is exactly the obligation the repo's
Mathlib-free kernel already proves -- so this module writes **no Lean**.  It
forwards the sealed ``positive_definite`` certificate to
:func:`omnibias.core.proof.lean_check.check_certificate`, which emits the
``allPivotsPos [...] = true`` obligation (``matrix_positive_definite_certified``)
and runs ``lake build``.

``theorem_prover_verified`` is earned **only** by a genuine ``lake`` pass; with no
toolchain present the check degrades gracefully (``available=False``,
``verified=False``) and never raises.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omnibias.core.proof.lean_check import (
    LeanCheckResult,
    check_certificate,
    lean_check_available,
)


def lean_check_sos(
    sealed_certificate: Mapping[str, Any], *, timeout: float = 600.0
) -> LeanCheckResult:
    r"""Kernel-check a sealed SOS certificate's ``allPivotsPos`` obligation.

    Returns a :class:`~omnibias.core.proof.lean_check.LeanCheckResult`;
    ``verified`` is ``True`` only on a genuine ``lake`` pass, ``available`` is
    ``False`` when no Lean toolchain / kernel checkout is present.
    """
    return check_certificate(sealed_certificate, timeout=timeout)


def is_theorem_prover_verified(
    sealed_certificate: Mapping[str, Any], *, timeout: float = 600.0
) -> bool:
    """``True`` iff the Lean kernel accepts the certificate's PD obligation."""
    return bool(lean_check_sos(sealed_certificate, timeout=timeout).verified)


def lean_available() -> bool:
    """``True`` iff a ``lake`` executable and the verified-kernel checkout are present."""
    return bool(lean_check_available())




__all__ = [
    "is_theorem_prover_verified",
    "lean_available",
    "lean_check_sos",
]
