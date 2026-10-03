# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from omnibias.geometry.patchwork import SignDistribution, staircase_triangulation
from omnibias.geometry.patchwork_symmetry import canonical_sign_orbit, sign_orbit_representatives


def test_canonical_sign_orbit_is_invariant_under_reflection() -> None:
    triangulation = staircase_triangulation(4)
    bits = tuple(0 for _ in range(len(triangulation.vertices)))
    signs = SignDistribution(4, bits)
    assert canonical_sign_orbit(signs) == canonical_sign_orbit(signs)


def test_sign_orbit_representatives_prune_duplicates() -> None:
    triangulation = staircase_triangulation(4)
    bits = tuple(index % 2 for index in range(len(triangulation.vertices)))
    signs = SignDistribution(4, bits)
    reps = sign_orbit_representatives((signs, signs))
    assert len(reps) == 1
