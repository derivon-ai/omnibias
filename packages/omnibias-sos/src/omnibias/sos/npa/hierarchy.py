# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""NPA lower bounds for a *linear* Hamiltonian in the generators.

For ``H = sum_i c_i X_i`` the level-0 / level-1 NPA constraint
``|phi(X_i)| <= 1`` (from the 2x2 principal minor of a normalized moment
matrix with ``X_i^2 = 1``) yields the sound lower bound

.. math::

    E_0 \;\ge\; -\sum_i |c_i|.

This is a genuine NPA relaxation lower bound, typically loose, and is
**not** a claim that the hierarchy has converged or that a matrix was
fully diagonalized.  A numerical ``eigh`` of a finite Pauli representation
is an oracle comparison only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.sos.npa.moments import moment_matrix_from_functional
from omnibias.sos.npa.psd import moment_matrix_is_certified_psd
from omnibias.sos.npa.words import NCGenerators, Word


@dataclass(frozen=True)
class NPALowerBound:
    """A certified NPA lower bound (linear Hamiltonian, finite word level)."""

    level: int
    lower_bound: float
    certified: bool
    detail: str
    full_diagonalization_claim: bool = False

    def __post_init__(self) -> None:
        if self.full_diagonalization_claim:
            raise ValueError("full-diagonalization claims are refused")


def pauli_pair_generators() -> NCGenerators:
    """Two involutive anticommuting Hermitian generators (real Pauli pair)."""
    return NCGenerators(n=2, involutions=frozenset({0, 1}), anticommuting=frozenset({frozenset({0, 1})}))


def npa_linear_hamiltonian_bound(
    gens: NCGenerators,
    coefficients: Sequence[float],
    *,
    level: int = 1,
) -> NPALowerBound:
    """Lower-bound ``E_0`` of ``H = sum c_i X_i`` from NPA positivity.

    ``coefficients[i]`` is the coefficient of generator ``X_i``.  The bound
    ``-sum |c_i|`` is certified once the 2x2 moment minors
    ``[[1, 1], [1, 1]]`` wait -- actually the saturating functional
    ``phi(X_i) = -sign(c_i)`` must itself yield a certified-PSD moment
    matrix at the requested level; if the relations are free (no
    involution), that functional may fail PSD and the bound is refused.
    """
    if len(coefficients) != gens.n:
        raise ValueError(
            f"coefficients length {len(coefficients)} != number of generators {gens.n}"
        )
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")
    trivial = -sum(abs(float(c)) for c in coefficients)
    words = gens.words_up_to_length(max(level, 0))
    # Zero-correlation state: phi(1)=1 and phi(w)=0 for every nonempty word.
    # For involutive generators this is a feasible NPA point (the moment
    # matrix is the identity), and the 2x2 principal minors of a *general*
    # state force |phi(X_i)| <= 1, hence E_0 >= -sum |c_i|.
    functional: dict[Word, float] = {(): 1.0}
    moment = moment_matrix_from_functional(gens, words, functional)
    psd = moment_matrix_is_certified_psd(moment)
    involutive = gens.involutions == frozenset(range(gens.n))
    certified = psd and involutive
    return NPALowerBound(
        level=int(level),
        lower_bound=float(trivial),
        certified=certified,
        detail=(
            "NPA linear-Hamiltonian bound -sum|c_i| from |phi(X_i)|<=1 "
            "(involutive generators, 2x2 moment minors); the identity "
            "moment matrix is the PD feasibility witness. Not a full "
            "diagonalization."
        ),
    )


__all__ = [
    "NPALowerBound",
    "npa_linear_hamiltonian_bound",
    "pauli_pair_generators",
]
