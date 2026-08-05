# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic containers for omnibias-qubo.

The differentiable annealed relaxation lives in :mod:`omnibias.qubo.jax` and
:mod:`omnibias.qubo.torch`; the numpy decoder / exact oracle live in
:mod:`omnibias.qubo._core.decode`; the certificate lives in
:mod:`omnibias.qubo.certify`. These containers only hold data (numpy) so the two
backends present an identical surface.

The shared schedule / solution / certificate containers now live in the
``omnibias-discrete`` substrate; ``AnnealSchedule``, ``QUBOSolution`` and
``QUBOCertificate`` are re-exported here so existing ``omnibias.qubo`` imports keep
working unchanged. ``QUBOProblem`` implements the substrate's ``DiscreteProblem`` seam
(``energy`` + ``to_polynomial`` + the closed-form ``flip_deltas`` fast path).

Two quadratic pseudo-Boolean models are supported and are exactly
inter-convertible (:mod:`omnibias.qubo._core.convert`):

* :class:`QUBOProblem` -- ``E(x) = x^T Q x + c^T x + const`` over ``x in {0, 1}^n``;
* :class:`IsingProblem` -- ``E(s) = s^T J s + h^T s + const`` over ``s in {-1, +1}^n``.

Terminology: the relaxation that consumes these containers hardens
``sigmoid(beta z)`` as ``beta -> inf`` -- the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias
``delta -> 0`` limit to the closed-form derivative ``sigma^(K-1)``; see
``docs/theory.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from omnibias.discrete import AnnealSchedule, DiscreteSolution, GapCertificate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.qubo.problem import IsingProblem as _IsingProblem
    from omnibias.qubo.problem import QUBOProblem as _QUBOProblem
    from omnibias.sos import Polynomial

FloatArray = NDArray[np.float64]

# The generic schedule / result containers are owned by the discrete substrate; keep the
# historical QUBO names as aliases so ``omnibias.qubo`` imports are unchanged.
QUBOSolution = DiscreteSolution
QUBOCertificate = GapCertificate


@dataclass(frozen=True)
class QUBOProblem:
    r"""A QUBO instance ``E(x) = x^T Q x + c^T x + const`` over ``x in {0, 1}^n``.

    ``Q`` is symmetrised on construction (only ``(Q + Q^T) / 2`` affects the energy on
    the cube, so the stored matrix is symmetric). Because ``x_i^2 = x_i`` on the cube,
    the diagonal of ``Q`` is an equally valid *linear* term; both forms are kept as
    given (the diagonal stays on ``Q``) so the polynomial handed to the SOS certificate
    is faithful.

    Attributes
    ----------
    Q:
        ``(n, n)`` numpy coefficient matrix (symmetrised, float).
    c:
        ``(n,)`` linear coefficients (defaults to zeros).
    const:
        Constant offset added to every energy.
    name:
        Optional label.
    """

    Q: FloatArray
    c: FloatArray | None = None
    const: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        q = np.asarray(self.Q, dtype=float)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise ValueError(f"Q must be a square (n, n) matrix, got shape {q.shape}")
        if q.shape[0] < 1:
            raise ValueError("Q must have at least one variable")
        q = 0.5 * (q + q.T)
        n = q.shape[0]
        if self.c is None:
            c = np.zeros(n)
        else:
            c = np.asarray(self.c, dtype=float).reshape(-1)
            if c.shape[0] != n:
                raise ValueError(f"c must have length {n}, got {c.shape[0]}")
        object.__setattr__(self, "Q", q)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "const", float(self.const))

    @property
    def n(self) -> int:
        return int(self.Q.shape[0])

    def energy(self, x: object) -> float | FloatArray:
        r"""Energy ``x^T Q x + c^T x + const`` for one point ``(n,)`` or a batch ``(m, n)``."""
        xv = np.asarray(x, dtype=float)
        c = np.asarray(self.c, dtype=float)
        quad = np.sum((xv @ self.Q) * xv, axis=-1)
        lin = xv @ c
        out = quad + lin + self.const
        return float(out) if xv.ndim == 1 else out

    def flip_deltas(self, x: object) -> FloatArray:
        r"""Closed-form energy change of flipping each single bit of ``x`` (one matvec).

        On the cube (``x_i^2 = x_i``) the flip-delta is
        ``(1 - 2 x_i)(Q_ii + 2 (Q x)_i - 2 Q_ii x_i + c_i)`` -- the fast path the shared
        local-search decoder uses instead of the generic batched-energy fallback.
        """
        q = np.asarray(self.Q, dtype=float)
        c = np.asarray(self.c, dtype=float)
        diag = np.diag(q).copy()
        xv = np.asarray(x, dtype=float)
        grad = diag + 2.0 * (q @ xv) - 2.0 * diag * xv + c
        deltas: FloatArray = (1.0 - 2.0 * xv) * grad
        return deltas

    def to_polynomial(self) -> Polynomial:
        """The energy as an :class:`omnibias.sos.Polynomial` (see :mod:`omnibias.qubo._core.convert`)."""
        from omnibias.qubo._core.convert import to_polynomial

        return to_polynomial(self)

    def to_ising(self) -> _IsingProblem:
        """Exact ``{0,1}`` -> ``{-1,+1}`` conversion (see :mod:`omnibias.qubo._core.convert`)."""
        from omnibias.qubo._core.convert import qubo_to_ising

        return qubo_to_ising(self)


@dataclass(frozen=True)
class IsingProblem:
    r"""An Ising instance ``E(s) = s^T J s + h^T s + const`` over ``s in {-1, +1}^n``.

    ``J`` is symmetrised and its diagonal folded into ``const`` (``s_i^2 = 1``), so the
    stored coupling matrix is symmetric with a zero diagonal.

    Attributes
    ----------
    J:
        ``(n, n)`` coupling matrix (symmetrised, zero diagonal, float).
    h:
        ``(n,)`` external field (defaults to zeros).
    const:
        Constant offset (absorbs ``trace(J)``).
    name:
        Optional label.
    """

    J: FloatArray
    h: FloatArray | None = None
    const: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        j = np.asarray(self.J, dtype=float)
        if j.ndim != 2 or j.shape[0] != j.shape[1]:
            raise ValueError(f"J must be a square (n, n) matrix, got shape {j.shape}")
        if j.shape[0] < 1:
            raise ValueError("J must have at least one variable")
        j = 0.5 * (j + j.T)
        trace = float(np.trace(j))
        np.fill_diagonal(j, 0.0)
        n = j.shape[0]
        if self.h is None:
            h = np.zeros(n)
        else:
            h = np.asarray(self.h, dtype=float).reshape(-1)
            if h.shape[0] != n:
                raise ValueError(f"h must have length {n}, got {h.shape[0]}")
        object.__setattr__(self, "J", j)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "const", float(self.const) + trace)

    @property
    def n(self) -> int:
        return int(self.J.shape[0])

    def energy(self, s: object) -> float | FloatArray:
        r"""Energy ``s^T J s + h^T s + const`` for one spin config ``(n,)`` or a batch."""
        sv = np.asarray(s, dtype=float)
        h = np.asarray(self.h, dtype=float)
        quad = np.sum((sv @ self.J) * sv, axis=-1)
        lin = sv @ h
        out = quad + lin + self.const
        return float(out) if sv.ndim == 1 else out

    def to_qubo(self) -> _QUBOProblem:
        """Exact ``{-1,+1}`` -> ``{0,1}`` conversion (see :mod:`omnibias.qubo._core.convert`)."""
        from omnibias.qubo._core.convert import ising_to_qubo

        return ising_to_qubo(self)


__all__ = [
    "AnnealSchedule",
    "IsingProblem",
    "QUBOCertificate",
    "QUBOProblem",
    "QUBOSolution",
]
