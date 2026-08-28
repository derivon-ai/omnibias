# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""1-D Cohn-Elkies bound on the Hermite self-dual basis."""

from __future__ import annotations

from omnibias.core.verified.cohn_elkies import (
    PUBLISHED_1D_PACKING_DENSITY,
    cohn_elkies_hermite_bound,
)


def test_cohn_elkies_1d_hermite_bound_is_certified_and_loose() -> None:
    result = cohn_elkies_hermite_bound()
    assert result.dimension == 1
    assert result.certified
    assert result.high_dimension_limit_claim is False
    assert result.published_comparison == PUBLISHED_1D_PACKING_DENSITY
    # A two-coefficient test function is a valid but typically loose bound.
    assert result.density_upper >= PUBLISHED_1D_PACKING_DENSITY
    assert result.f_at_zero[0] > 0.0
    assert result.hat_f_at_zero[0] > 0.0
