# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Two-sided finite-lattice ground-state certificate.

Given a **finite**, already-assembled Hermitian matrix (typically a particle-
number / spin sector of :mod:`omnibias.core.verified.lattice_hamiltonian`),
this module sandwiches the lowest eigenvalue between

* a Ritz **upper** bound from a trial vector, and
* a Temple **lower** bound that needs a certified ``rho <= lambda_2``
  (obtained from blocked interval ``LDL^T`` inertia via
  :func:`~omnibias.core.verified.eig_operator.count_eigenvalues_below`),

and optionally a Davis-Kahan invariant-subspace enclosure of the trial
ground state.  A numerical ``numpy.linalg.eigh`` is used **only** as an
oracle / trial-vector proposer; it is never the certificate.

The first shipped slice is the **two-site Hubbard model at half filling**
(open chain, sector ``(N_up, N_down) = (1, 1)``), whose exact singlet
ground energy ``U/2 - sqrt((U/2)^2 + 4 t^2)`` is a regression oracle, not
a continuum thermodynamic-limit claim.

Scope.  One declared lattice, one boundary condition, one sector, one
tolerance.  Not a continuum Hubbard model, not a quantum phase transition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.verified.eig_operator import (
    count_eigenvalues_below,
    ritz_upper_bound,
    temple_lower_bound_vector,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.invariant_subspace import (
    InvariantSubspaceCertificate,
    certified_invariant_subspace,
)
from omnibias.core.verified.lattice_hamiltonian import (
    BoundaryCondition,
    hamiltonian_to_matrix,
    hubbard_hamiltonian,
    numerical_exact_diagonalization,
    restrict_to_indices,
    spin_resolved_occupation,
)
from omnibias.core.verified.linalg import identity_matrix

Vector = Sequence[float]


@dataclass(frozen=True)
class LatticeGroundStateCertificate:
    """Two-sided enclosure of the lowest eigenvalue of one finite matrix."""

    lower: float
    upper: float
    rho: float
    n_sites: int
    boundary: str
    sector: tuple[int, int]
    tolerance: float
    certified: bool
    invariant_subspace: InvariantSubspaceCertificate | None
    detail: str
    numerical_oracle_energy: float | None = None
    continuum_claim: bool = False

    def __post_init__(self) -> None:
        if self.continuum_claim:
            raise ValueError("continuum_claim must stay False")


def _identity_mass(n: int) -> list[list[Interval]]:
    return identity_matrix(n)


def certify_finite_ground_state(
    dense: object,
    trial: Vector,
    *,
    n_sites: int,
    boundary: str,
    sector: tuple[int, int],
    tolerance: float,
    numerical_oracle_energy: float | None = None,
) -> LatticeGroundStateCertificate:
    """Ritz + Temple + inertia sandwich of ``lambda_min`` of ``dense``.

    ``trial`` is any nonzero vector (typically the numerical ground
    eigenvector, labelled as an oracle).  ``rho`` is chosen as the
    mid-point of the two lowest *numerical* eigenvalues and then
    **re-checked** by certified inertia: ``count_eigenvalues_below(A, I, rho)``
    must equal ``1``, which is the statement ``lambda_1 < rho <= lambda_2``.
    """
    import numpy as np

    matrix = np.asarray(dense, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"dense Hamiltonian must be square, got {matrix.shape}")
    n = int(matrix.shape[0])
    if len(trial) != n:
        raise ValueError(f"trial length {len(trial)} != matrix size {n}")
    interval_h = hamiltonian_to_matrix(matrix)
    mass = _identity_mass(n)
    oracle = numerical_exact_diagonalization(matrix)
    lam1 = float(oracle.eigenvalues[0])
    lam2 = float(oracle.eigenvalues[1]) if n >= 2 else lam1 + 1.0
    rho = 0.5 * (lam1 + lam2)
    count = count_eigenvalues_below(interval_h, mass, rho)
    ritz = ritz_upper_bound(interval_h, list(trial))
    upper = float(ritz.hi)
    lower = float("-inf")
    invariant: InvariantSubspaceCertificate | None = None
    certified = False
    detail = "inertia count at rho did not isolate a simple ground eigenvalue"
    if count == 1:
        try:
            temple = temple_lower_bound_vector(interval_h, list(trial), rho)
            lower = float(temple.lo)
        except ValueError as exc:
            detail = f"Temple refused: {exc}"
        else:
            gap = upper - lower
            if gap < 0.0:
                detail = "Temple lower bound exceeded Ritz upper (trial too poor)"
            elif gap <= float(tolerance):
                try:
                    invariant = certified_invariant_subspace(interval_h, [list(trial)])
                except (ValueError, ArithmeticError):
                    invariant = None
                certified = True
                detail = (
                    "two-sided finite-lattice ground-state sandwich "
                    "(Ritz upper + Temple lower + blocked LDLT inertia); "
                    "not a thermodynamic-limit claim"
                )
            else:
                detail = (
                    f"sandwich width {gap} exceeds declared tolerance {tolerance}"
                )
    oracle_energy = (
        float(numerical_oracle_energy) if numerical_oracle_energy is not None else lam1
    )
    return LatticeGroundStateCertificate(
        lower=lower,
        upper=upper,
        rho=float(rho),
        n_sites=int(n_sites),
        boundary=str(boundary),
        sector=(int(sector[0]), int(sector[1])),
        tolerance=float(tolerance),
        certified=certified,
        invariant_subspace=invariant,
        detail=detail,
        numerical_oracle_energy=oracle_energy,
    )


def two_site_hubbard_exact_energy(*, hopping: float, u: float) -> float:
    """Closed-form singlet ground energy of the 2-site open Hubbard chain.

    ``E = U/2 - sqrt((U/2)^2 + 4 t^2)`` in the half-filled ``(1, 1)`` sector.
    This is an analytic oracle for the certificate, not a continuum claim.
    """
    half = 0.5 * float(u)
    return half - (half * half + 4.0 * float(hopping) * float(hopping)) ** 0.5


def hubbard_half_filled_ground_state(
    n_sites: int = 2,
    *,
    hopping: float = 1.0,
    u: float = 4.0,
    boundary: BoundaryCondition = "open",
    tolerance: float = 0.05,
) -> LatticeGroundStateCertificate:
    """Two-sided certificate for the half-filled Hubbard chain's ground energy.

    Default slice: ``n_sites=2``, open chain, ``t=1``, ``U=4``, sector
    ``(1, 1)``.  Larger ``n_sites`` remain finite exact-diagonalization
    instances (Hilbert space ``4**n_sites`` before the sector cut).
    """
    if n_sites < 1:
        raise ValueError(f"n_sites must be >= 1, got {n_sites}")
    ham = hubbard_hamiltonian(n_sites, boundary, hopping=hopping, u=u)
    n_up, n_down = spin_resolved_occupation(n_sites)
    target_up = n_sites // 2
    target_down = n_sites // 2
    if n_sites == 2:
        target_up, target_down = 1, 1
    import numpy as np

    indices = np.flatnonzero((n_up == target_up) & (n_down == target_down))
    if indices.size == 0:
        raise ValueError(
            f"half-filled sector (N_up, N_down)=({target_up}, {target_down}) is empty"
        )
    sector = restrict_to_indices(ham.dense, indices)
    oracle = numerical_exact_diagonalization(sector)
    trial = [float(x) for x in oracle.eigenvectors[:, 0]]
    return certify_finite_ground_state(
        sector,
        trial,
        n_sites=n_sites,
        boundary=str(boundary),
        sector=(int(target_up), int(target_down)),
        tolerance=tolerance,
        numerical_oracle_energy=float(oracle.eigenvalues[0]),
    )


__all__ = [
    "LatticeGroundStateCertificate",
    "certify_finite_ground_state",
    "hubbard_half_filled_ground_state",
    "two_site_hubbard_exact_energy",
]
