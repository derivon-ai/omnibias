# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Two-sided finite-lattice Hubbard ground-state certificate."""

from __future__ import annotations

from omnibias.core.verified.lattice_ground_state import (
    hubbard_half_filled_ground_state,
    two_site_hubbard_exact_energy,
)


def test_two_site_hubbard_sandwich_contains_exact() -> None:
    hopping, u = 1.0, 4.0
    exact = two_site_hubbard_exact_energy(hopping=hopping, u=u)
    cert = hubbard_half_filled_ground_state(
        2, hopping=hopping, u=u, boundary="open", tolerance=0.25
    )
    assert cert.n_sites == 2
    assert cert.sector == (1, 1)
    assert cert.boundary == "open"
    assert cert.continuum_claim is False
    assert cert.lower <= exact <= cert.upper
    assert cert.certified
    assert cert.numerical_oracle_energy is not None
    assert abs(cert.numerical_oracle_energy - exact) < 1e-8
