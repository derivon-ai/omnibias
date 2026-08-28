# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified PSD of an NPA moment matrix via interval ``LDL^T``."""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.verified.eig_operator import is_positive_definite
from omnibias.sos.npa.moments import MomentMatrix


def moment_matrix_is_certified_psd(matrix: MomentMatrix) -> bool:
    """``True`` iff the moment matrix is certified positive definite.

    Each float entry is promoted through ``Fraction(float(x))`` (lossless for
    finite ``float64``) and handed to
    :func:`~omnibias.core.verified.eig_operator.is_positive_definite`.  A
    merely semidefinite (singular) moment matrix is **not** certified here
    -- that is an honest refuse, not a silent pass.
    """
    interval_matrix = [
        [Fraction(float(entry)) for entry in row] for row in matrix.entries
    ]
    return is_positive_definite(interval_matrix)


__all__ = ["moment_matrix_is_certified_psd"]
