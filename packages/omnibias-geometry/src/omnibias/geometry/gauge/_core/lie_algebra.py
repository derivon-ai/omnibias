# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Compact Lie algebras for gauge theory (pure Python + numpy).

A :class:`LieAlgebra` describes the Lie algebra of a compact gauge group by its
Hermitian generators ``T^a`` (in the fundamental representation), the totally
antisymmetric structure constants ``f^{abc}`` defined by

.. math::

    [T^a, T^b] = i\,f^{abc}\,T^c,

and the totally symmetric ``d^{abc}`` from the anticommutator. We use the
physics normalization ``tr(T^a T^b) = 1/2\,\delta^{ab}``, which fixes

.. math::

    f^{abc} = -2i\,\operatorname{tr}([T^a, T^b]\,T^c),\qquad
    d^{abc} = 2\,\operatorname{tr}(\{T^a, T^b\}\,T^c).

``su(2)`` returns the Pauli matrices over two (``T^a = sigma^a/2``, giving
``f^{abc} = epsilon^{abc}``) and ``su(3)`` returns the Gell-Mann matrices over
two, both in their conventional ordering. General ``su(N)`` uses the generalized
Gell-Mann basis. ``u(1)`` is the one-dimensional abelian algebra (``f = 0``).

This module is backend-neutral: it builds generators / constants with numpy and
exposes them as numpy arrays. The torch / jax backends convert them to tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]


def _pauli_generators() -> ComplexArray:
    """``su(2)`` generators ``sigma^a / 2`` (a = 1, 2, 3)."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return np.stack([sx, sy, sz]) / 2.0


def _gell_mann_generators() -> ComplexArray:
    """``su(3)`` generators ``lambda^a / 2`` in conventional Gell-Mann order."""
    lam = np.zeros((8, 3, 3), dtype=np.complex128)
    lam[0] = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    lam[1] = [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]
    lam[2] = [[1, 0, 0], [0, -1, 0], [0, 0, 0]]
    lam[3] = [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
    lam[4] = [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]
    lam[5] = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    lam[6] = [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]
    lam[7] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3.0)
    return lam / 2.0


def _generalized_gell_mann_generators(n: int) -> ComplexArray:
    """Generalized Gell-Mann ``su(N)`` generators, normalized ``tr(T^aT^b)=1/2``."""
    gens: list[ComplexArray] = []
    # Symmetric and antisymmetric off-diagonal generators.
    for j in range(n):
        for k in range(j + 1, n):
            sym = np.zeros((n, n), dtype=np.complex128)
            sym[j, k] = 0.5
            sym[k, j] = 0.5
            gens.append(sym)
            asym = np.zeros((n, n), dtype=np.complex128)
            asym[j, k] = -0.5j
            asym[k, j] = 0.5j
            gens.append(asym)
    # Diagonal (Cartan) generators.
    for ell in range(1, n):
        diag = np.zeros(n, dtype=np.float64)
        diag[:ell] = 1.0
        diag[ell] = -float(ell)
        norm = 1.0 / np.sqrt(2.0 * ell * (ell + 1))
        gens.append(np.diag(diag).astype(np.complex128) * norm)
    return np.stack(gens)


@cache
def _generators(name: str, n_fundamental: int) -> ComplexArray:
    if name == "u(1)":
        return np.array([[[1.0 / np.sqrt(2.0)]]], dtype=np.complex128)
    if name == "su(2)":
        return _pauli_generators()
    if name == "su(3)":
        return _gell_mann_generators()
    return _generalized_gell_mann_generators(n_fundamental)


@cache
def _constants(name: str, n_fundamental: int) -> tuple[FloatArray, FloatArray]:
    """Return ``(f^{abc}, d^{abc})`` real arrays of shape ``(dim, dim, dim)``."""
    t = _generators(name, n_fundamental)
    dim = t.shape[0]
    f = np.zeros((dim, dim, dim), dtype=np.float64)
    d = np.zeros((dim, dim, dim), dtype=np.float64)
    for a in range(dim):
        for b in range(dim):
            comm = t[a] @ t[b] - t[b] @ t[a]
            anti = t[a] @ t[b] + t[b] @ t[a]
            for c in range(dim):
                f[a, b, c] = float(np.real(-2.0j * np.trace(comm @ t[c])))
                d[a, b, c] = float(np.real(2.0 * np.trace(anti @ t[c])))
    # Clean numerical dust below double-precision trace error.
    f[np.abs(f) < 1e-12] = 0.0
    d[np.abs(d) < 1e-12] = 0.0
    return f, d


@dataclass(frozen=True)
class LieAlgebra:
    """The Lie algebra of a compact gauge group in the fundamental representation.

    Parameters
    ----------
    name
        One of ``"u(1)"``, ``"su(2)"``, ``"su(3)"``, or ``"su(N)"`` (the generic
        label used for ``N >= 4``).
    n_fundamental
        The dimension ``N`` of the fundamental representation. ``1`` for
        ``u(1)``.

    Notes
    -----
    The instance is lightweight (just metadata); the generators and structure
    constants are built lazily and cached by ``(name, n_fundamental)``.
    """

    name: str
    n_fundamental: int

    def __post_init__(self) -> None:
        if self.n_fundamental < 1:
            raise ValueError(f"n_fundamental must be >= 1, got {self.n_fundamental}")
        if self.name == "u(1)" and self.n_fundamental != 1:
            raise ValueError("u(1) has n_fundamental == 1")

    @property
    def dim(self) -> int:
        """Number of generators (the adjoint dimension): ``N^2 - 1`` for su(N)."""
        if self.name == "u(1)":
            return 1
        return self.n_fundamental * self.n_fundamental - 1

    @property
    def is_abelian(self) -> bool:
        return self.dim == 1

    def generators(self) -> ComplexArray:
        """The Hermitian generators ``T^a``, shape ``(dim, N, N)``."""
        return _generators(self.name, self.n_fundamental).copy()

    def structure_constants(self) -> FloatArray:
        """Totally antisymmetric ``f^{abc}``, shape ``(dim, dim, dim)``."""
        return _constants(self.name, self.n_fundamental)[0].copy()

    def symmetric_constants(self) -> FloatArray:
        """Totally symmetric ``d^{abc}``, shape ``(dim, dim, dim)``."""
        return _constants(self.name, self.n_fundamental)[1].copy()

    def dual_coxeter_number(self) -> int:
        """``N`` for su(N); ``0`` for u(1) (the adjoint Casimir ``C_2(G) = N``)."""
        return 0 if self.name == "u(1)" else self.n_fundamental


def su(n: int) -> LieAlgebra:
    """The ``su(N)`` Lie algebra (``N >= 2``)."""
    if n < 2:
        raise ValueError(f"su(N) needs N >= 2, got {n}")
    name = f"su({n})" if n in (2, 3) else "su(N)"
    return LieAlgebra(name=name, n_fundamental=n)


def u1() -> LieAlgebra:
    """The abelian ``u(1)`` Lie algebra."""
    return LieAlgebra(name="u(1)", n_fundamental=1)


def as_lie_algebra(spec: str) -> LieAlgebra:
    """Resolve a string label such as ``"su(2)"`` / ``"su(3)"`` / ``"u(1)"``."""
    label = spec.strip().lower().replace(" ", "")
    if label in {"u(1)", "u1"}:
        return u1()
    if label.startswith("su(") and label.endswith(")"):
        return su(int(label[3:-1]))
    raise ValueError(f"unrecognized Lie algebra label: {spec!r}")


__all__ = [
    "LieAlgebra",
    "as_lie_algebra",
    "su",
    "u1",
]
