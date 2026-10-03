# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.dulac import DulacExpansion
from omnibias.dynamics.hyperbolic_dulac import (
    certify_hyperbolic_dulac_expansion,
    verify_hyperbolic_dulac_expansion,
)


def test_hyperbolic_dulac_certificate_replays() -> None:
    expansion = DulacExpansion.create(
        ((Fraction(1), 0, 1), (Fraction(2), 1, Fraction(-1, 2))),
        truncation_order=2,
        remainder_bound=Fraction(0),
    )
    certificate = certify_hyperbolic_dulac_expansion(
        expansion,
        parameter_lo=Fraction(1, 100),
        parameter_hi=Fraction(1),
        flatness_exponent=2,
        majorant_constant=Fraction(1),
    )
    assert verify_hyperbolic_dulac_expansion(certificate)
