# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Named finite-lattice Hamiltonian families, symmetry sectors, and an exact-diagonalization oracle.

Built compositionally on :mod:`omnibias.core.verified.lattice_operators`: four
standard 1-D lattice families -- **TFIM**, **Heisenberg** (XXZ), **Hubbard**,
and **t-J** -- each returned as both a dense matrix (small ``n``) and a
:class:`~omnibias.core.verified.lattice_operators.SparseCOO` sparse action
(efficient matrix-vector product, no dense materialization required), plus:

* symmetry-sector restriction (one conserved quantum number per family), and
* an honestly-labeled **numerical** exact-diagonalization oracle
  (:func:`numerical_exact_diagonalization`, a thin wrapper around
  :func:`numpy.linalg.eigh` -- *not* a certified/interval result), and
* :func:`hamiltonian_to_matrix`, converting a dense Hamiltonian into the
  ``Matrix = Sequence[Sequence[IntervalLike]]`` convention already used by
  :mod:`omnibias.core.verified.eig_operator` /
  :mod:`omnibias.core.verified.linalg_array` /
  :mod:`omnibias.core.verified.invariant_subspace`, ready for those modules'
  certified eigenvalue / inertia / invariant-subspace machinery.

Family conventions (documented explicitly since every one of these has more
than one convention in the literature)
--------------------------------------------------------------------------
* **TFIM** (:func:`tfim_hamiltonian`): ``H = -J sum_{<ij>} sigma^z_i sigma^z_j
  - h sum_i sigma^x_i`` (Ising coupling along ``z``, field transverse along
  ``x`` -- the Pfeuty / Sachdev convention).  Conserves the ``Z2`` spin-flip
  parity ``P = prod_i sigma^x_i``.
* **Heisenberg XXZ** (:func:`heisenberg_hamiltonian`): ``H = sum_{<ij>}
  [J_xy (S^x_i S^x_j + S^y_i S^y_j) + J_z S^z_i S^z_j]`` with spin-1/2
  operators ``S^a = sigma^a / 2``; ``j_xy == j_z`` recovers the isotropic
  XXX chain.  Conserves total ``S^z``.
* **Hubbard** (:func:`hubbard_hamiltonian`): the standard single-band
  ``H = -t sum_{<ij>,sigma} (c^dagger_{i,sigma} c_{j,sigma} + h.c.) + U sum_i
  n_{i,up} n_{i,down}``.  Conserves ``(N_up, N_down)`` separately (hence
  total ``N``).
* **t-J** (:func:`tj_hamiltonian`): the standard large-``U`` projection of
  the Hubbard model, ``H = -t sum_{<ij>,sigma} P (c^dagger_{i,sigma}
  c_{j,sigma} + h.c.) P + J sum_{<ij>} (S_i . S_j - n_i n_j / 4)``, where
  ``P`` projects onto the *no-double-occupancy* subspace (dimension
  ``3**n_sites``, one of ``{empty, up, down}`` per site) -- the exchange term
  is automatically zero whenever either site is empty, so no extra
  case-analysis is needed beyond restricting the Hilbert space itself.
  Conserves ``(N_up, N_down)`` within that subspace, exactly as for Hubbard.

Practical size ceiling
-----------------------
Dense construction and :func:`numerical_exact_diagonalization` are exercised
up to ``n_sites = 12`` (spin chains, dimension ``4096``) and ``n_sites = 6``
(Hubbard / t-J, ``2*n_sites`` orbitals, dimension up to ``4096``) in this
package's test suite; the sparse path (:class:`SparseCOO` triplets plus
:meth:`~omnibias.core.verified.lattice_operators.SparseCOO.matvec`) has no
such ceiling built in, but this module always enumerates every basis state to
build it, so its cost is linear in the (still exponential) Hilbert space
dimension -- it is a sparse *representation*, not a way to avoid the
exponential blow-up itself.

Honesty note
------------
:func:`numerical_exact_diagonalization` is a **numerical** oracle (ordinary
``float64`` linear algebra, no rounding/error control) -- it is explicitly
*not* a certified result.  A future certified ground-state work item should
instead route :func:`hamiltonian_to_matrix`'s output through
:mod:`omnibias.core.verified.eig_operator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .lattice_operators import (
    FloatArray,
    IntArray,
    SparseCOO,
    apply_fermion_string,
    coo_from_triplets,
)

BoundaryCondition: TypeAlias = Literal["open", "periodic"]
#: The `eig_operator` / `linalg_array` / `invariant_subspace` matrix convention.
IntervalMatrixLike: TypeAlias = list[list[Fraction]]


@dataclass(frozen=True)
class HamiltonianMatrix:
    """A finite-lattice Hamiltonian in both dense and sparse-COO form.

    ``dense`` is ``numpy.linalg.eigh``-ready; ``sparse`` gives the same
    matrix as a :class:`~omnibias.core.verified.lattice_operators.SparseCOO`
    for matrix-vector products that never materialize ``dense``.  The two are
    built from one shared set of triplets (``sparse.to_dense() == dense``, up
    to the coalescing order), not cross-checked after the fact.
    """

    dense: FloatArray
    sparse: SparseCOO


@dataclass(frozen=True)
class TJHamiltonian:
    """A t-J Hamiltonian restricted to the no-double-occupancy subspace.

    ``basis_states[k]`` is the full ``2 * n_sites``-orbital Hubbard-Fock-space
    computational-basis index (see the module docstring's orbital ordering)
    that subspace row/column ``k`` corresponds to; ``dense`` and ``sparse``
    are indexed by ``k``, not by the full Fock-space index.
    """

    dense: FloatArray
    sparse: SparseCOO
    basis_states: IntArray


@dataclass(frozen=True)
class NumericalSpectrum:
    """A **numerical** (not certified) eigendecomposition from ``numpy.linalg.eigh``.

    ``eigenvalues`` is ascending; ``eigenvectors[:, k]`` is the eigenvector
    for ``eigenvalues[k]``.  This is an honest ``float64`` oracle with no
    rigorous error control -- do not confuse it with a certified `Interval`
    result from :mod:`omnibias.core.verified.eig_operator`.
    """

    eigenvalues: FloatArray
    eigenvectors: FloatArray


def chain_bonds(n_sites: int, boundary: BoundaryCondition) -> list[tuple[int, int]]:
    """Nearest-neighbor bonds of a 1-D chain of ``n_sites`` sites.

    ``boundary="open"`` gives the ``n_sites - 1`` bonds ``(0, 1), ..., (n-2,
    n-1)``.  ``boundary="periodic"`` additionally wraps around with
    ``(n_sites - 1, 0)`` -- except at ``n_sites == 2``, where that bond would
    coincide with ``(0, 1)`` (there is only one distinct pair), so it is not
    duplicated; a periodic 2-site chain is therefore identical to an open one.
    """
    if n_sites < 1:
        raise ValueError(f"n_sites must be >= 1, got {n_sites}")
    bonds = [(i, i + 1) for i in range(n_sites - 1)]
    if boundary == "periodic":
        if n_sites < 2:
            raise ValueError("a periodic chain needs at least 2 sites")
        if n_sites >= 3:
            bonds.append((n_sites - 1, 0))
    elif boundary != "open":
        raise ValueError(f"boundary must be 'open' or 'periodic', got {boundary!r}")
    return bonds


def _orbital_up(site: int) -> int:
    return 2 * site


def _orbital_down(site: int) -> int:
    return 2 * site + 1


def tfim_hamiltonian(
    n_sites: int, boundary: BoundaryCondition, *, coupling: float, field: float
) -> HamiltonianMatrix:
    """The transverse-field Ising model on a spin-1/2 chain (see module docstring).

    Hilbert space dimension ``2**n_sites``.
    """
    bonds = chain_bonds(n_sites, boundary)
    dim = 1 << n_sites
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for state in range(dim):
        diag = 0.0
        for i, j in bonds:
            zi = 1.0 if not (state >> i) & 1 else -1.0
            zj = 1.0 if not (state >> j) & 1 else -1.0
            diag -= coupling * zi * zj
        if diag != 0.0:
            rows.append(state)
            cols.append(state)
            values.append(diag)
        if field != 0.0:
            for i in range(n_sites):
                flipped = state ^ (1 << i)
                rows.append(flipped)
                cols.append(state)
                values.append(-field)
    coo = coo_from_triplets(rows, cols, values, (dim, dim)).coalesce()
    return HamiltonianMatrix(dense=coo.to_dense(), sparse=coo)


def heisenberg_hamiltonian(
    n_sites: int, boundary: BoundaryCondition, *, j_xy: float, j_z: float
) -> HamiltonianMatrix:
    """The spin-1/2 XXZ Heisenberg chain (see module docstring; ``j_xy == j_z`` is XXX).

    Hilbert space dimension ``2**n_sites``.
    """
    bonds = chain_bonds(n_sites, boundary)
    dim = 1 << n_sites
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for state in range(dim):
        diag = 0.0
        for i, j in bonds:
            szi = 0.5 if not (state >> i) & 1 else -0.5
            szj = 0.5 if not (state >> j) & 1 else -0.5
            diag += j_z * szi * szj
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj and j_xy != 0.0:
                new_state = state ^ (1 << i) ^ (1 << j)
                rows.append(new_state)
                cols.append(state)
                values.append(j_xy / 2.0)
        if diag != 0.0:
            rows.append(state)
            cols.append(state)
            values.append(diag)
    coo = coo_from_triplets(rows, cols, values, (dim, dim)).coalesce()
    return HamiltonianMatrix(dense=coo.to_dense(), sparse=coo)


def hubbard_hamiltonian(
    n_sites: int, boundary: BoundaryCondition, *, hopping: float, u: float
) -> HamiltonianMatrix:
    """The single-band Hubbard model on a chain (see module docstring).

    Orbital ``2 * site`` is spin-up, ``2 * site + 1`` is spin-down; Hilbert
    space dimension ``4**n_sites``.
    """
    bonds = chain_bonds(n_sites, boundary)
    n_orbitals = 2 * n_sites
    dim = 1 << n_orbitals
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for state in range(dim):
        diag = 0.0
        if u != 0.0:
            for site in range(n_sites):
                n_up = (state >> _orbital_up(site)) & 1
                n_down = (state >> _orbital_down(site)) & 1
                diag += u * n_up * n_down
        if diag != 0.0:
            rows.append(state)
            cols.append(state)
            values.append(diag)
        if hopping != 0.0:
            for i, j in bonds:
                for orb_i, orb_j in ((_orbital_up(i), _orbital_up(j)), (_orbital_down(i), _orbital_down(j))):
                    for src, dst in ((orb_i, orb_j), (orb_j, orb_i)):
                        step = apply_fermion_string(state, [(src, False), (dst, True)])
                        if step is not None:
                            new_state, sign = step
                            rows.append(new_state)
                            cols.append(state)
                            values.append(-hopping * sign)
    coo = coo_from_triplets(rows, cols, values, (dim, dim)).coalesce()
    return HamiltonianMatrix(dense=coo.to_dense(), sparse=coo)


def tj_hamiltonian(
    n_sites: int, boundary: BoundaryCondition, *, hopping: float, j_coupling: float
) -> TJHamiltonian:
    """The t-J model restricted to the no-double-occupancy subspace (see module docstring).

    Hilbert space dimension ``3**n_sites``.
    """
    bonds = chain_bonds(n_sites, boundary)
    n_orbitals = 2 * n_sites
    full_dim = 1 << n_orbitals

    def is_allowed(state: int) -> bool:
        return all(
            not ((state >> _orbital_up(site)) & 1 and (state >> _orbital_down(site)) & 1)
            for site in range(n_sites)
        )

    allowed = [state for state in range(full_dim) if is_allowed(state)]
    basis_states = np.asarray(allowed, dtype=np.intp)
    index_of = {state: k for k, state in enumerate(allowed)}
    proj_dim = len(allowed)

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for p_index, state in enumerate(allowed):
        if hopping != 0.0:
            for i, j in bonds:
                for orb_i, orb_j in (
                    (_orbital_up(i), _orbital_up(j)),
                    (_orbital_down(i), _orbital_down(j)),
                ):
                    for src, dst in ((orb_i, orb_j), (orb_j, orb_i)):
                        step = apply_fermion_string(state, [(src, False), (dst, True)])
                        if step is None:
                            continue
                        new_state, sign = step
                        new_index = index_of.get(new_state)
                        if new_index is not None:
                            rows.append(new_index)
                            cols.append(p_index)
                            values.append(-hopping * sign)
        if j_coupling != 0.0:
            for i, j in bonds:
                n_up_i = (state >> _orbital_up(i)) & 1
                n_down_i = (state >> _orbital_down(i)) & 1
                n_up_j = (state >> _orbital_up(j)) & 1
                n_down_j = (state >> _orbital_down(j)) & 1
                n_i = n_up_i + n_down_i
                n_j = n_up_j + n_down_j
                sz_i = 0.5 * (n_up_i - n_down_i)
                sz_j = 0.5 * (n_up_j - n_down_j)
                diag_term = j_coupling * (sz_i * sz_j - 0.25 * n_i * n_j)
                if diag_term != 0.0:
                    rows.append(p_index)
                    cols.append(p_index)
                    values.append(diag_term)
                # S^+_i S^-_j: flips site i down->up and site j up->down.
                step = apply_fermion_string(
                    state,
                    [
                        (_orbital_up(j), False),
                        (_orbital_down(j), True),
                        (_orbital_down(i), False),
                        (_orbital_up(i), True),
                    ],
                )
                if step is not None:
                    new_state, sign = step
                    new_index = index_of.get(new_state)
                    if new_index is not None:
                        rows.append(new_index)
                        cols.append(p_index)
                        values.append(0.5 * j_coupling * sign)
                # S^-_i S^+_j: flips site i up->down and site j down->up.
                step2 = apply_fermion_string(
                    state,
                    [
                        (_orbital_down(j), False),
                        (_orbital_up(j), True),
                        (_orbital_up(i), False),
                        (_orbital_down(i), True),
                    ],
                )
                if step2 is not None:
                    new_state2, sign2 = step2
                    new_index2 = index_of.get(new_state2)
                    if new_index2 is not None:
                        rows.append(new_index2)
                        cols.append(p_index)
                        values.append(0.5 * j_coupling * sign2)
    coo = coo_from_triplets(rows, cols, values, (proj_dim, proj_dim)).coalesce()
    return TJHamiltonian(dense=coo.to_dense(), sparse=coo, basis_states=basis_states)


# --------------------------------------------------------------------------- #
# Symmetry sectors.
# --------------------------------------------------------------------------- #
def total_sz_doubled(n_sites: int) -> IntArray:
    """``2 * S^z_total`` for every computational basis state of a spin-1/2 chain.

    Length ``2**n_sites``; entries are integers in ``[-n_sites, n_sites]``.
    This is the conserved quantum number used to restrict
    :func:`heisenberg_hamiltonian`.
    """
    if n_sites < 1:
        raise ValueError(f"n_sites must be >= 1, got {n_sites}")
    dim = 1 << n_sites
    states = np.arange(dim, dtype=np.int64)
    down_count = np.bitwise_count(states)
    return cast(IntArray, (n_sites - 2 * down_count).astype(np.intp))


def particle_number(n_orbitals: int) -> IntArray:
    """Total occupation number for every state of an ``n_orbitals`` Fock space.

    Length ``2**n_orbitals``; entries are integers in ``[0, n_orbitals]``.
    For :func:`hubbard_hamiltonian` (``n_orbitals = 2 * n_sites``) this is
    total particle number ``N``; the same array indexed by
    :attr:`TJHamiltonian.basis_states` gives ``N`` for the t-J subspace.
    """
    if n_orbitals < 1:
        raise ValueError(f"n_orbitals must be >= 1, got {n_orbitals}")
    dim = 1 << n_orbitals
    states = np.arange(dim, dtype=np.int64)
    return cast(IntArray, np.bitwise_count(states).astype(np.intp))


def spin_resolved_occupation(n_sites: int) -> tuple[IntArray, IntArray]:
    """``(N_up, N_down)`` for every state of the ``2 * n_sites``-orbital Fock space.

    Each array has length ``4**n_sites``.  Orbital ``2 * site`` is spin-up,
    ``2 * site + 1`` is spin-down (the convention used by
    :func:`hubbard_hamiltonian` / :func:`tj_hamiltonian`).  Indexing either
    array by :attr:`TJHamiltonian.basis_states` gives the same quantum number
    restricted to the t-J subspace.
    """
    if n_sites < 1:
        raise ValueError(f"n_sites must be >= 1, got {n_sites}")
    n_orbitals = 2 * n_sites
    dim = 1 << n_orbitals
    states = np.arange(dim, dtype=np.int64)
    up_mask = sum(1 << _orbital_up(site) for site in range(n_sites))
    down_mask = sum(1 << _orbital_down(site) for site in range(n_sites))
    n_up = np.bitwise_count(states & up_mask)
    n_down = np.bitwise_count(states & down_mask)
    return cast(IntArray, n_up.astype(np.intp)), cast(IntArray, n_down.astype(np.intp))


def sector_indices(labels: IntArray, target: int) -> IntArray:
    """Indices where ``labels == target`` -- the sector membership for a diagonal symmetry."""
    return cast(IntArray, np.flatnonzero(np.asarray(labels) == target).astype(np.intp))


def restrict_to_indices(matrix: FloatArray, indices: IntArray) -> FloatArray:
    """The principal submatrix of ``matrix`` on ``indices`` (a diagonal-symmetry sector)."""
    if indices.size == 0:
        raise ValueError("sector is empty")
    return cast(FloatArray, np.asarray(matrix)[np.ix_(indices, indices)])


def z2_parity_basis(n_sites: int, parity: Literal[1, -1]) -> FloatArray:
    """Orthonormal basis of the ``sigma^x``-spin-flip-parity sector of a spin-1/2 chain.

    ``P = prod_i sigma^x_i`` commutes with :func:`tfim_hamiltonian` (see the
    module docstring); its ``+1`` sector is spanned by the symmetric
    combinations ``(|s> + |flip(s)>) / sqrt(2)`` and its ``-1`` sector by the
    antisymmetric ones, where ``flip(s)`` complements every bit of ``s``.
    Returns a ``(2**n_sites, 2**(n_sites - 1))`` matrix ``basis`` with
    orthonormal columns; restrict via :func:`restrict_via_basis`.
    """
    if n_sites < 1:
        raise ValueError(f"n_sites must be >= 1, got {n_sites}")
    if parity not in (1, -1):
        raise ValueError(f"parity must be 1 or -1, got {parity}")
    dim = 1 << n_sites
    flip_all = dim - 1
    seen = np.zeros(dim, dtype=bool)
    columns: list[FloatArray] = []
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for state in range(dim):
        if seen[state]:
            continue
        partner = state ^ flip_all
        seen[state] = True
        seen[partner] = True
        vec = np.zeros(dim, dtype=np.float64)
        vec[state] = inv_sqrt2
        vec[partner] = inv_sqrt2 if parity == 1 else -inv_sqrt2
        columns.append(vec)
    return cast(FloatArray, np.stack(columns, axis=1))


def restrict_via_basis(matrix: FloatArray, basis: FloatArray) -> FloatArray:
    """Restrict ``matrix`` to the subspace spanned by ``basis``'s (orthonormal) columns."""
    return cast(FloatArray, basis.T @ np.asarray(matrix) @ basis)


# --------------------------------------------------------------------------- #
# Numerical (not certified) exact diagonalization oracle.
# --------------------------------------------------------------------------- #
def numerical_exact_diagonalization(hamiltonian: FloatArray, *, atol: float = 1e-9) -> NumericalSpectrum:
    """A **numerical**, uncertified ``numpy.linalg.eigh`` wrapper -- an honest ground-truth oracle.

    Raises :class:`ValueError` if ``hamiltonian`` is not square or not
    symmetric to within ``atol``.  This function makes no rigorous claim
    about the accuracy of its output beyond ordinary ``float64`` arithmetic;
    a *certified* enclosure of the spectrum requires
    :mod:`omnibias.core.verified.eig_operator` on the output of
    :func:`hamiltonian_to_matrix`, not this function.
    """
    matrix = np.asarray(hamiltonian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"hamiltonian must be square, got shape {matrix.shape}")
    if not np.allclose(matrix, matrix.T, atol=atol):
        raise ValueError("hamiltonian must be symmetric (within atol) for numpy.linalg.eigh")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return NumericalSpectrum(
        eigenvalues=cast(FloatArray, eigenvalues), eigenvectors=cast(FloatArray, eigenvectors)
    )


def ground_state_energy(hamiltonian: FloatArray, *, atol: float = 1e-9) -> float:
    """The lowest eigenvalue from the **numerical** oracle (see :func:`numerical_exact_diagonalization`)."""
    return float(numerical_exact_diagonalization(hamiltonian, atol=atol).eigenvalues[0])


# --------------------------------------------------------------------------- #
# Interval-matrix construction (for the certified eig_operator / linalg_array /
# invariant_subspace machinery).
# --------------------------------------------------------------------------- #
def hamiltonian_to_matrix(hamiltonian: NDArray[np.float64]) -> IntervalMatrixLike:
    """Convert a dense Hamiltonian to the ``Matrix = Sequence[Sequence[IntervalLike]]`` convention.

    Every entry becomes an exact :class:`fractions.Fraction` via
    ``Fraction(float(x))`` -- lossless because every finite ``float64`` is
    itself a dyadic rational, so no rounding is introduced here.  The result
    is a plain nested list already accepted, entry-by-entry, by
    :func:`omnibias.core.verified.eig_operator.interval_ldlt_inertia`,
    :func:`omnibias.core.verified.eig_operator.count_eigenvalues_below`, and
    :func:`omnibias.core.verified.linalg.to_interval_matrix` (each promotes an
    ``IntervalLike`` entry via ``Interval.from_value``, which routes a
    ``Fraction`` through :meth:`~omnibias.core.verified.interval.Interval.from_rational`
    for the tightest outward-rounded enclosure of a value that is, here,
    already exact).
    """
    matrix = np.asarray(hamiltonian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"hamiltonian must be square, got shape {matrix.shape}")
    return [[Fraction(float(x)) for x in row] for row in matrix]


__all__ = [
    "BoundaryCondition",
    "HamiltonianMatrix",
    "IntervalMatrixLike",
    "NumericalSpectrum",
    "TJHamiltonian",
    "chain_bonds",
    "ground_state_energy",
    "hamiltonian_to_matrix",
    "heisenberg_hamiltonian",
    "hubbard_hamiltonian",
    "numerical_exact_diagonalization",
    "particle_number",
    "restrict_to_indices",
    "restrict_via_basis",
    "sector_indices",
    "spin_resolved_occupation",
    "tfim_hamiltonian",
    "tj_hamiltonian",
    "total_sz_doubled",
    "z2_parity_basis",
]
