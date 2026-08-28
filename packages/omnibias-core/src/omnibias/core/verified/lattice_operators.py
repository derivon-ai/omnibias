# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact Pauli / Jordan-Wigner fermionic operator algebra on a finite lattice.

This module is the **operator-algebra layer** for
:mod:`omnibias.core.verified.lattice_hamiltonian`: exact single-site Pauli /
spin-raising-lowering matrices, their multi-site embedding into the
``2**n_sites``-dimensional Hilbert space of a spin-1/2 chain, and fermionic
creation/annihilation operators built with the standard Jordan-Wigner string so
that ``{c_i, c_j^dagger} = delta_ij`` and ``{c_i, c_j} = 0`` hold *exactly* (to
machine precision -- every entry is ``0``, ``1``, or ``-1``, never rounded).

It also carries a tiny, self-documented COO sparse-matrix type
(:class:`SparseCOO`) used by :mod:`lattice_hamiltonian` for the "sparse
action" requirement: an efficient matrix-vector product that never
materializes the dense ``2**n x 2**n`` (or ``4**n``) matrix.  This mirrors the
``(rows, columns, data, shape)`` COO convention already used by
:func:`omnibias.core.verified.interval_array.sparse_matvec` -- no ``scipy``
dependency is introduced (``omnibias-core`` depends only on ``numpy``).

Honesty note
------------
Everything in this module is **exact** (finite-dimensional linear algebra with
integer/rational structure), not a rigorous *enclosure*.  A caller who needs
an outward-rounded ``Interval`` version of a resulting matrix should route it
through :func:`omnibias.core.verified.lattice_hamiltonian.hamiltonian_to_interval_matrix`
before handing it to :mod:`omnibias.core.verified.eig_operator`.

Conventions
-----------
* Local 2-D basis ``(|0>, |1>)`` with ``sigma_z = diag(1, -1)`` in that basis
  order: for a **spin** site, ``|0>`` has ``sigma_z = +1`` ("up") and ``|1>``
  has ``sigma_z = -1`` ("down"); for a **fermionic** orbital, ``|0>`` is
  unoccupied and ``|1>`` is occupied (bit value = occupation number).
* Multi-site embedding places site ``0`` as the *least significant* tensor
  factor, i.e. computational basis index ``k`` has bit ``i`` equal to the
  local state of site/orbital ``i``.
* The textbook spin ladder operators are ``sigma^+ = (sigma_x + i sigma_y)/2``
  and ``sigma^- = (sigma_x - i sigma_y)/2``; with the basis order above,
  ``sigma^+`` raises ``sigma_z`` from ``-1`` to ``+1``, i.e. it maps
  ``|1> -> |0>`` (the *same* index-lowering ``2x2`` shape used below as
  :data:`SIGMA_LOWER`), and ``sigma^-`` maps ``|0> -> |1>`` (:data:`SIGMA_RAISE`).
  :func:`spin_raising_operator` / :func:`spin_lowering_operator` use exactly
  this correspondence -- see their docstrings.
* Jordan-Wigner string: ``c_i = (prod_{k<i} sigma_z_k) (fermion annihilation
  at site i)``, where the local fermion annihilation operator maps
  ``|1> -> |0>`` (removes a particle, :data:`SIGMA_LOWER`) and creation maps
  ``|0> -> |1>`` (:data:`SIGMA_RAISE`) -- the standard convention that makes
  fermionic operators on *different* orbitals of one JW chain anticommute
  correctly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

ComplexArray: TypeAlias = NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.intp]

#: 2x2 identity.
PAULI_I: ComplexArray = cast(ComplexArray, np.eye(2, dtype=np.complex128))
#: Pauli matrix ``sigma_x``.
PAULI_X: ComplexArray = cast(ComplexArray, np.array([[0, 1], [1, 0]], dtype=np.complex128))
#: Pauli matrix ``sigma_y``.
PAULI_Y: ComplexArray = cast(ComplexArray, np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
#: Pauli matrix ``sigma_z``.
PAULI_Z: ComplexArray = cast(ComplexArray, np.array([[1, 0], [0, -1]], dtype=np.complex128))
#: Local basis-index-raising operator ``|0> -> |1>`` (fermion: create a particle;
#: spin: ``sigma^-``, lowers ``sigma_z`` from ``+1`` to ``-1``).
SIGMA_RAISE: ComplexArray = cast(ComplexArray, np.array([[0, 0], [1, 0]], dtype=np.complex128))
#: Local basis-index-lowering operator ``|1> -> |0>`` (fermion: annihilate a particle;
#: spin: ``sigma^+``, raises ``sigma_z`` from ``-1`` to ``+1``).
SIGMA_LOWER: ComplexArray = cast(ComplexArray, np.array([[0, 1], [0, 0]], dtype=np.complex128))


def pauli_matrices() -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    """Return the single-site ``(sigma_x, sigma_y, sigma_z)`` as ``(2, 2)`` arrays."""
    return PAULI_X, PAULI_Y, PAULI_Z


def embed_operator_string(
    ops: Mapping[int, ComplexArray], n_sites: int
) -> ComplexArray:
    r"""Kron-embed a collection of local ``2x2`` operators into ``n_sites``.

    ``ops`` maps a site index to the local operator placed there; every other
    site gets the ``2x2`` identity.  Site ``0`` is the least-significant tensor
    factor, matching the bit convention used throughout this module.

    Raises
    ------
    ValueError
        If ``n_sites < 1`` or a key of ``ops`` is out of range.
    """
    if n_sites < 1:
        raise ValueError(f"n_sites must be >= 1, got {n_sites}")
    for site in ops:
        if not 0 <= site < n_sites:
            raise ValueError(f"site index {site} out of range for n_sites={n_sites}")
    result: ComplexArray = cast(ComplexArray, np.array([[1.0 + 0.0j]]))
    for site in reversed(range(n_sites)):
        block = ops.get(site, PAULI_I)
        result = cast(ComplexArray, np.kron(result, block))
    return result


def embed_single_site_operator(op: ComplexArray, site: int, n_sites: int) -> ComplexArray:
    """Kron-embed one local ``2x2`` operator ``op`` at ``site`` (identity elsewhere)."""
    return embed_operator_string({site: op}, n_sites)


def pauli_operator(axis: Literal["x", "y", "z"], site: int, n_sites: int) -> ComplexArray:
    """The embedded ``sigma_{axis}`` acting on ``site`` within an ``n_sites`` chain."""
    table = {"x": PAULI_X, "y": PAULI_Y, "z": PAULI_Z}
    if axis not in table:
        raise ValueError(f"axis must be one of 'x', 'y', 'z'; got {axis!r}")
    return embed_single_site_operator(table[axis], site, n_sites)


def spin_raising_operator(site: int, n_sites: int) -> ComplexArray:
    """Embedded spin-1/2 raising operator ``sigma^+_site`` (``|down> -> |up>``, no JW string).

    ``sigma^+ = (sigma_x + i sigma_y)/2`` maps ``|1> -> |0>`` given this
    module's ``sigma_z = diag(1, -1)`` basis order, i.e. it is
    :data:`SIGMA_LOWER` in the index sense -- see the module docstring.
    """
    return embed_single_site_operator(SIGMA_LOWER, site, n_sites)


def spin_lowering_operator(site: int, n_sites: int) -> ComplexArray:
    """Embedded spin-1/2 lowering operator ``sigma^-_site`` (``|up> -> |down>``, no JW string).

    ``sigma^- = (sigma_x - i sigma_y)/2`` maps ``|0> -> |1>``, i.e. it is
    :data:`SIGMA_RAISE` in the index sense -- see the module docstring.
    """
    return embed_single_site_operator(SIGMA_RAISE, site, n_sites)


def jordan_wigner_string(sites: Sequence[int], n_sites: int) -> ComplexArray:
    """The parity string ``prod_{s in sites} sigma_z_s``, embedded in ``n_sites``."""
    ops = {s: PAULI_Z for s in sites}
    return embed_operator_string(ops, n_sites)


def fermion_annihilation_operator(orbital: int, n_orbitals: int) -> ComplexArray:
    r"""Jordan-Wigner annihilation operator ``c_orbital`` on ``n_orbitals`` modes.

    ``c_i = (prod_{k<i} sigma_z_k) (local |1> -> |0> operator at i)``, i.e.
    :data:`SIGMA_LOWER` at site ``i`` with a ``sigma_z`` string on sites
    ``< i``.  Every entry is exactly
    ``0``, ``1``, or ``-1`` (real), so the returned array's imaginary part is
    identically zero even though the dtype is ``complex128`` (kept for a
    uniform interface with :func:`pauli_operator`).
    """
    if not 0 <= orbital < n_orbitals:
        raise ValueError(f"orbital {orbital} out of range for n_orbitals={n_orbitals}")
    ops: dict[int, ComplexArray] = {k: PAULI_Z for k in range(orbital)}
    ops[orbital] = SIGMA_LOWER
    return embed_operator_string(ops, n_orbitals)


def fermion_creation_operator(orbital: int, n_orbitals: int) -> ComplexArray:
    """Jordan-Wigner creation operator ``c_orbital^dagger`` (the conjugate transpose)."""
    return cast(ComplexArray, fermion_annihilation_operator(orbital, n_orbitals).conj().T)


def apply_fermion_op(state: int, orbital: int, *, dagger: bool) -> tuple[int, int] | None:
    r"""Apply ``c_orbital`` (``dagger=False``) or ``c_orbital^dagger`` (``dagger=True``).

    ``state`` is a computational basis index (bit ``orbital`` is that mode's
    occupation).  Returns ``(new_state, sign)`` with ``sign in {+1, -1}``, or
    ``None`` when the operator annihilates the state (Pauli exclusion, or
    annihilating an empty mode).  This is the elementary Jordan-Wigner step;
    :func:`apply_fermion_string` composes it into multi-operator products.
    """
    occupied = (state >> orbital) & 1
    if dagger:
        if occupied:
            return None
    elif not occupied:
        return None
    sign = -1 if bin(state & ((1 << orbital) - 1)).count("1") % 2 else 1
    new_state = state ^ (1 << orbital)
    return new_state, sign


def apply_fermion_string(
    state: int, ops: Sequence[tuple[int, bool]]
) -> tuple[int, int] | None:
    """Apply a sequence of single-mode fermion operators, ``ops[0]`` acting first.

    Each element of ``ops`` is ``(orbital, dagger)``.  Returns
    ``(final_state, sign)`` or ``None`` as soon as any step is forbidden.  This
    is the safe, general primitive behind every multi-fermion matrix element in
    :mod:`omnibias.core.verified.lattice_hamiltonian` (hopping, and the
    charge-conserving spin-exchange terms of the ``t``-``J`` model) -- no
    matrix element formula is hand-derived without going through this
    step-by-step JW bookkeeping.
    """
    current = state
    total_sign = 1
    for orbital, dagger in ops:
        step = apply_fermion_op(current, orbital, dagger=dagger)
        if step is None:
            return None
        current, sign = step
        total_sign *= sign
    return current, total_sign


@dataclass(frozen=True)
class SparseCOO:
    """A minimal, self-documented COO sparse matrix: no ``scipy`` dependency.

    ``rows[k]``, ``cols[k]``, ``values[k]`` describe one matrix entry each;
    duplicate ``(row, col)`` pairs are accumulated (as in the standard COO
    convention, and matching
    :func:`omnibias.core.verified.interval_array.sparse_matvec`).  Every
    Hamiltonian family in :mod:`lattice_hamiltonian` returns one of these
    alongside its dense materialization, so a Lanczos-style consumer can act
    on a state vector without ever forming the dense matrix.
    """

    rows: IntArray
    cols: IntArray
    values: FloatArray
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.rows.shape != self.cols.shape or self.rows.shape != self.values.shape:
            raise ValueError("rows, cols, and values must have the same shape")
        n_rows, n_cols = self.shape
        if n_rows < 0 or n_cols < 0:
            raise ValueError("shape entries must be non-negative")
        if self.rows.size and (
            int(self.rows.max()) >= n_rows or int(self.cols.max()) >= n_cols
        ):
            raise ValueError("row/col index out of bounds for the declared shape")

    def to_dense(self) -> FloatArray:
        """Materialize the dense matrix (duplicate entries are summed)."""
        dense = np.zeros(self.shape, dtype=np.float64)
        np.add.at(dense, (self.rows, self.cols), self.values)
        return cast(FloatArray, dense)

    def matvec(self, vector: NDArray[np.float64]) -> FloatArray:
        """Sparse matrix-vector product ``A @ vector`` without materializing ``A``."""
        v = np.asarray(vector, dtype=np.float64)
        if v.shape != (self.shape[1],):
            raise ValueError(
                f"vector shape {v.shape} incompatible with matrix shape {self.shape}"
            )
        out = np.zeros(self.shape[0], dtype=np.float64)
        np.add.at(out, self.rows, self.values * v[self.cols])
        return cast(FloatArray, out)

    def coalesce(self) -> SparseCOO:
        """Return an equivalent :class:`SparseCOO` with unique ``(row, col)`` pairs."""
        if self.rows.size == 0:
            return self
        flat = self.rows.astype(np.int64) * self.shape[1] + self.cols.astype(np.int64)
        order = np.argsort(flat, kind="stable")
        flat_sorted = flat[order]
        values_sorted = self.values[order]
        unique_flat, start_indices = np.unique(flat_sorted, return_index=True)
        summed = np.add.reduceat(values_sorted, start_indices)
        new_rows = (unique_flat // self.shape[1]).astype(np.intp)
        new_cols = (unique_flat % self.shape[1]).astype(np.intp)
        return SparseCOO(rows=new_rows, cols=new_cols, values=summed, shape=self.shape)


def coo_from_triplets(
    rows: Sequence[int],
    cols: Sequence[int],
    values: Sequence[float],
    shape: tuple[int, int],
) -> SparseCOO:
    """Build a :class:`SparseCOO` from plain Python sequences."""
    return SparseCOO(
        rows=np.asarray(rows, dtype=np.intp),
        cols=np.asarray(cols, dtype=np.intp),
        values=np.asarray(values, dtype=np.float64),
        shape=shape,
    )


def coo_add(*coos: SparseCOO) -> SparseCOO:
    """Concatenate (and coalesce) several :class:`SparseCOO` matrices of equal shape."""
    if not coos:
        raise ValueError("coo_add requires at least one operand")
    shape = coos[0].shape
    for coo in coos[1:]:
        if coo.shape != shape:
            raise ValueError("all operands must share the same shape")
    rows = np.concatenate([c.rows for c in coos])
    cols = np.concatenate([c.cols for c in coos])
    values = np.concatenate([c.values for c in coos])
    return SparseCOO(rows=rows, cols=cols, values=values, shape=shape).coalesce()


def dense_to_coo(dense: NDArray[np.float64], *, tol: float = 0.0) -> SparseCOO:
    """Convert a dense real matrix to :class:`SparseCOO`, dropping entries ``<= tol`` in magnitude."""
    mask = np.abs(dense) > tol
    rows_arr, cols_arr = np.nonzero(mask)
    values = dense[rows_arr, cols_arr]
    return SparseCOO(
        rows=rows_arr.astype(np.intp),
        cols=cols_arr.astype(np.intp),
        values=values.astype(np.float64),
        shape=(dense.shape[0], dense.shape[1]),
    )


__all__ = [
    "ComplexArray",
    "FloatArray",
    "IntArray",
    "PAULI_I",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "SIGMA_LOWER",
    "SIGMA_RAISE",
    "SparseCOO",
    "apply_fermion_op",
    "apply_fermion_string",
    "coo_add",
    "coo_from_triplets",
    "dense_to_coo",
    "embed_operator_string",
    "embed_single_site_operator",
    "fermion_annihilation_operator",
    "fermion_creation_operator",
    "jordan_wigner_string",
    "pauli_matrices",
    "pauli_operator",
    "spin_lowering_operator",
    "spin_raising_operator",
]
