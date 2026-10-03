# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Symmetry reduction for regular-triangulation patchwork searches."""

from __future__ import annotations

from itertools import product

from omnibias.geometry.patchwork import SignDistribution, Triangulation, lattice_points

__all__ = [
    "canonical_sign_orbit",
    "sign_orbit_representatives",
    "triangulation_stabilizer_size",
]


def triangulation_stabilizer_size(triangulation: Triangulation) -> int:
    """Return the size of the dihedral sign stabilizer used for pruning."""
    return 8 if triangulation.degree % 2 == 0 else 4


def _flip_signs(signs: SignDistribution, flip_x: bool, flip_y: bool) -> SignDistribution:
    degree = signs.degree
    new_bits = tuple(
        signs.reflected_bit((-point[0] if flip_x else point[0], -point[1] if flip_y else point[1]))
        for point in lattice_points(degree)
    )
    return SignDistribution(degree, new_bits)


def canonical_sign_orbit(signs: SignDistribution) -> tuple[int, ...]:
    """Lexicographic minimum bit vector over the `(±x, ±y)` sign group."""
    variants = [_flip_signs(signs, flip_x, flip_y).bits for flip_x, flip_y in product((False, True), repeat=2)]
    return min(variants)


def sign_orbit_representatives(signs: tuple[SignDistribution, ...]) -> tuple[SignDistribution, ...]:
    """Keep one representative per sign orbit."""
    seen: set[tuple[int, ...]] = set()
    out: list[SignDistribution] = []
    for sign in signs:
        key = canonical_sign_orbit(sign)
        if key in seen:
            continue
        seen.add(key)
        out.append(sign)
    return tuple(out)
