# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.dynamics.uniform_remainder import (
    certify_uniform_flat_remainder,
    verify_uniform_flat_remainder,
)


def test_uniform_flat_remainder_certificate_round_trip() -> None:
    certificate = certify_uniform_flat_remainder(
        parameter_lo=Fraction(1, 1000),
        parameter_hi=Fraction(1, 10),
        flatness_exponent=3,
        derivative_order=1,
        majorant_constant=Fraction(2),
        truncation_order=2,
        claim="test uniform flat remainder",
    )
    assert verify_uniform_flat_remainder(certificate)
    assert certificate.seal["honesty"]["uniform_remainder_proved"] is True


def test_uniform_flat_remainder_rejects_nonpositive_majorant() -> None:
    with pytest.raises(ValueError, match="majorant_constant"):
        certify_uniform_flat_remainder(
            parameter_lo=Fraction(1, 100),
            parameter_hi=Fraction(1, 10),
            flatness_exponent=2,
            derivative_order=0,
            majorant_constant=Fraction(0),
            truncation_order=1,
            claim="bad",
        )
