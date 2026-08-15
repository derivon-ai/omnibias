# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Dimension census for coordinate vs covariant vs singlet search spaces.

A raw 2-jet of ``A_mu^a`` in 4D SU(3) has 480 real components. Symbolic
regression must never build that library. This module *counts* the explosion
and refuses the coordinate / flattened-adjoint constructions.

Honesty: these are finite index counts on a smooth connection, not a
Hilbert-series completeness theorem and not a mass-gap claim.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from math import comb

# Named SU(3), d=4 gates (absolute, not isfinite).
SU3_4D_COORDINATE_2JET = 480
SU3_4D_F = 48
SU3_4D_DF_RAW = 192
SU3_4D_DF_BIANCHI_REDUCED = 160

_FLAT_EXACT = frozenset({"|A|^2", "u_x", "u_t", "u_y", "u_z", "dA", "raw dA"})
_FLAT_RE = re.compile(r"^(A_\d+_\d+|F_\d+_\d+|DF_\d+_\d+_\d+|D2F_.*)$")


def _positive_int(value: int, name: str) -> int:
    n = int(value)
    if n < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return n


def raw_connection_jet_dimension(
    spacetime_dim: int, algebra_dim: int, order: int
) -> int:
    r"""Independent real components of ``{A, dA, ..., d^{order} A}``.

    The ``k``-th partials of each of the ``d n`` connection components are
    symmetrized: ``C(d + k - 1, k) * d * n``. For ``d=4``, ``n=8``,
    ``order=2`` this is ``32 + 128 + 320 = 480``.
    """
    d = _positive_int(spacetime_dim, "spacetime_dim")
    n = _positive_int(algebra_dim, "algebra_dim")
    k_max = int(order)
    if k_max < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    n_comp = d * n
    return sum(comb(d + k - 1, k) * n_comp for k in range(k_max + 1))


def raw_covariant_fiber_dimension(
    spacetime_dim: int, algebra_dim: int, cov_order: int
) -> int:
    r"""Unreduced ``(D^k F)_{mu nu}^a`` count: ``d^k * C(d, 2) * n``.

    ``cov_order=0`` is ``F`` (``48`` for SU(3) in 4D). ``cov_order=1`` is
    raw ``D F`` (``192``). Covariant derivatives are *not* symmetrized.
    """
    d = _positive_int(spacetime_dim, "spacetime_dim")
    n = _positive_int(algebra_dim, "algebra_dim")
    k = int(cov_order)
    if k < 0:
        raise ValueError(f"cov_order must be >= 0, got {cov_order}")
    n_f = comb(d, 2) * n
    return (d**k) * n_f


def bianchi_reduced_df_dimension(spacetime_dim: int, algebra_dim: int) -> int:
    """Raw ``D F`` minus the independent Bianchi 1-form (``d n`` components)."""
    d = _positive_int(spacetime_dim, "spacetime_dim")
    n = _positive_int(algebra_dim, "algebra_dim")
    return raw_covariant_fiber_dimension(d, n, 1) - d * n


def refuse_coordinate_jet_library(*_args: object, **_kwargs: object) -> None:
    """Hard refuse: never build a coordinate jet of ``A`` as an SR library."""
    raise ValueError(
        "never search the coordinate 2-jet of A (coordinate trap); "
        "use GaugeInvariantDictionary of contracted Weyl singlets"
    )


def refuse_component_fiber_library(*, cov_order: int) -> None:
    """Hard refuse: uncontracted ``D^k F`` components are not a search space."""
    if int(cov_order) >= 2:
        raise ValueError(
            "uncontracted D^k F component fibers with k>=2 are not a search "
            "space; use GaugeInvariantDictionary (contracted singlets, "
            "cov_order<=1)"
        )
    raise ValueError(
        "flattened adjoint component fibers are not a search space; "
        "use GaugeInvariantDictionary of contracted Weyl singlets"
    )


def refuse_flattened_adjoint_library(names: Iterable[str]) -> None:
    """Raise if any name is a color-basis / coordinate component."""
    illegal = [name for name in names if name in _FLAT_EXACT or _FLAT_RE.match(name)]
    if illegal:
        raise ValueError(
            "gauge library admits only allowlisted covariant atoms; "
            "flattened adjoint components are not a search space "
            f"(coordinate trap); rejected {illegal}"
        )


__all__ = [
    "SU3_4D_COORDINATE_2JET",
    "SU3_4D_DF_BIANCHI_REDUCED",
    "SU3_4D_DF_RAW",
    "SU3_4D_F",
    "bianchi_reduced_df_dimension",
    "raw_connection_jet_dimension",
    "raw_covariant_fiber_dimension",
    "refuse_component_fiber_library",
    "refuse_coordinate_jet_library",
    "refuse_flattened_adjoint_library",
]
