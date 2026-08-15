# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Mass-dimension-graded Weyl-singlet dictionary for Yang-Mills SR.

Group-theoretic reduction happens *before* STLSQ. The searchable library is a
closed list of Lorentz-scalar color singlets of ``F`` and ``D F``, truncated
by mass dimension. Bianchi is an identity, not a feature. ``|F-*F|^2`` is a
Euclidean dim-4 syzygy of ``tr(F^2)`` and ``tr(F*Ftilde)``. The cyclic
``d^{abc} tr(F^3)`` contraction is a 4D syzygy (antisymmetric 2-forms).

This is not a Hilbert-series completeness claim, not Wilson / Polyakov
language, and not a continuum mass-gap claim. First ship: spacetime dim 4,
``max_cov_order <= 1``, mass dimension ``<= 6``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge._core.covariant_jet import (
    LEGAL_SINGLET_ATOMS,
    SINGLET_BIANCHI_SQ,
    SINGLET_SELF_DUAL_SQ,
    SINGLET_TR_F2,
    SINGLET_TR_F_FTILDE,
    SINGLET_YM_SQ,
    GaugeCovariantJet,
)
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra

AtomRole = Literal["search", "identity", "syzygy"]

SINGLET_DF_SQ = "|DF|^2"
SINGLET_TR_F3 = "tr(F^3)"

IMPLEMENTED_INVARIANT_NAMES: frozenset[str] = frozenset(
    set(LEGAL_SINGLET_ATOMS) | {SINGLET_DF_SQ, SINGLET_TR_F3}
)

# Named upper bound on searchable dim-6 / k<=1 / SU(3) atoms.
MAX_SEARCHABLE_DIM6_SU3 = 8


@dataclass(frozen=True)
class InvariantAtom:
    """One contracted Weyl singlet (or identity / syzygy diagnostic)."""

    name: str
    mass_dimension: int
    cov_order: int
    n_traces: int
    role: AtomRole
    lorentz_irrep: str = "scalar"
    color_irrep: str = "singlet"
    algebra_min_n: int = 1

    @property
    def complexity(self) -> int:
        """Representation-theoretic cost: mass dimension + number of traces."""
        return int(self.mass_dimension) + int(self.n_traces)


def representation_complexity(
    names: Iterable[str], atoms: Mapping[str, InvariantAtom]
) -> int:
    """Sum of :attr:`InvariantAtom.complexity` over ``names``."""
    total = 0
    for name in names:
        if name not in atoms:
            raise KeyError(f"unknown invariant atom {name!r}")
        total += atoms[name].complexity
    return total


def _catalog() -> tuple[InvariantAtom, ...]:
    return (
        InvariantAtom(
            SINGLET_TR_F2, mass_dimension=4, cov_order=0, n_traces=1, role="search"
        ),
        InvariantAtom(
            SINGLET_TR_F_FTILDE,
            mass_dimension=4,
            cov_order=0,
            n_traces=1,
            role="search",
        ),
        InvariantAtom(
            SINGLET_SELF_DUAL_SQ,
            mass_dimension=4,
            cov_order=0,
            n_traces=1,
            role="syzygy",
        ),
        InvariantAtom(
            SINGLET_YM_SQ, mass_dimension=6, cov_order=1, n_traces=1, role="search"
        ),
        InvariantAtom(
            SINGLET_DF_SQ, mass_dimension=6, cov_order=1, n_traces=1, role="search"
        ),
        InvariantAtom(
            SINGLET_TR_F3,
            mass_dimension=6,
            cov_order=0,
            n_traces=1,
            # Cyclic d^{abc} F^3 vanishes in 4D (antisymmetric 2-forms).
            role="syzygy",
            algebra_min_n=3,
        ),
        InvariantAtom(
            SINGLET_BIANCHI_SQ,
            mass_dimension=6,
            cov_order=1,
            n_traces=1,
            role="identity",
        ),
    )


def enumerate_gauge_invariants(
    *,
    mass_dimension: int,
    max_cov_order: int,
    algebra: LieAlgebra,
    spacetime_dim: int = 4,
) -> tuple[InvariantAtom, ...]:
    """Closed Weyl patterns with ``atom.mass_dimension <= mass_dimension``.

    ``max_cov_order >= 2`` is clamped to 1: the dictionary never emits
    uncontracted ``D^k F`` components. Use
    :func:`~omnibias.geometry.gauge._core.jet_dimension.refuse_component_fiber_library`
    to reject a component-fiber request.
    """
    if spacetime_dim != 4:
        raise ValueError(
            f"invariant dictionary ships spacetime_dim=4 only, got {spacetime_dim}"
        )
    if mass_dimension < 4:
        raise ValueError(f"mass_dimension must be >= 4, got {mass_dimension}")
    if mass_dimension > 6:
        raise ValueError(
            "invariant dictionary ships mass_dimension<=6 only "
            f"(got {mass_dimension}); not a Hilbert-series claim"
        )
    cov_cap = min(int(max_cov_order), 1)
    if cov_cap < 0:
        raise ValueError(f"max_cov_order must be >= 0, got {max_cov_order}")
    n_fund = 1 if algebra.is_abelian else algebra.n_fundamental
    out: list[InvariantAtom] = []
    for atom in _catalog():
        if atom.mass_dimension > mass_dimension:
            continue
        if atom.cov_order > cov_cap:
            continue
        if n_fund < atom.algebra_min_n:
            continue
        out.append(atom)
    return tuple(out)


def evaluate_named_invariants(
    jet: GaugeCovariantJet, names: Iterable[str]
) -> dict[str, np.ndarray]:
    """Evaluate implemented singlet columns of ``jet`` by name."""
    needed = list(dict.fromkeys(names))
    unknown = [name for name in needed if name not in IMPLEMENTED_INVARIANT_NAMES]
    if unknown:
        raise KeyError(f"unimplemented invariant columns {unknown}")
    base = jet.singlets()
    eta = np.asarray(jet.signature, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for name in needed:
        if name in base:
            out[name] = np.asarray(base[name], dtype=np.float64).reshape(-1)
            continue
        if name == SINGLET_DF_SQ:
            val = kernels.df_square_density(np, jet.DF, eta)
            out[name] = np.asarray(val, dtype=np.float64).reshape(-1)
            continue
        d_abc = jet.algebra.symmetric_constants()
        val = kernels.cubic_casimir_density(np, jet.F, d_abc, eta)
        out[name] = np.asarray(val, dtype=np.float64).reshape(-1)
    return out


@dataclass(frozen=True)
class GaugeInvariantDictionary:
    """Filtered invariant atoms that STLSQ is allowed to see."""

    atoms: tuple[InvariantAtom, ...]
    algebra_name: str
    mass_dimension: int
    max_cov_order: int
    spacetime_dim: int = 4

    @classmethod
    def build(
        cls,
        *,
        mass_dimension: int,
        max_cov_order: int,
        algebra: LieAlgebra,
        spacetime_dim: int = 4,
        roles: tuple[AtomRole, ...] = ("search",),
    ) -> GaugeInvariantDictionary:
        atoms = enumerate_gauge_invariants(
            mass_dimension=mass_dimension,
            max_cov_order=max_cov_order,
            algebra=algebra,
            spacetime_dim=spacetime_dim,
        )
        filtered = tuple(atom for atom in atoms if atom.role in roles)
        return cls(
            atoms=filtered,
            algebra_name=algebra.name,
            mass_dimension=int(mass_dimension),
            max_cov_order=min(int(max_cov_order), 1),
            spacetime_dim=int(spacetime_dim),
        )

    @property
    def legal_names(self) -> frozenset[str]:
        return frozenset(atom.name for atom in self.atoms)

    def atom_map(self) -> dict[str, InvariantAtom]:
        return {atom.name: atom for atom in self.atoms}

    def evaluate(self, jet: GaugeCovariantJet) -> dict[str, np.ndarray]:
        return evaluate_named_invariants(jet, self.legal_names)


__all__ = [
    "IMPLEMENTED_INVARIANT_NAMES",
    "MAX_SEARCHABLE_DIM6_SU3",
    "SINGLET_DF_SQ",
    "SINGLET_TR_F3",
    "GaugeInvariantDictionary",
    "InvariantAtom",
    "enumerate_gauge_invariants",
    "evaluate_named_invariants",
    "representation_complexity",
]
