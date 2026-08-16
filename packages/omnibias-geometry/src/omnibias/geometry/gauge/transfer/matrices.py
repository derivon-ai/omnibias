# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite lattice transfer matrices with rigorously enclosed entries.

Each constructor returns a :class:`TransferMatrix`: a **fixed**, finite-dimensional
matrix whose entries are outward-rounded :class:`~omnibias.core.verified.Interval`
enclosures, together with the approximate eigenvectors the certified gap engines in
:mod:`omnibias.core.verified.eig` want and -- where the spectrum is known in closed
form -- the true eigenvalues to check a certificate against.

Two bases appear:

``character``
    The group-character basis, in which the single-plaquette / single-link weight
    is **diagonal** by orthogonality of characters.  Truncating the character sum
    is a modelling choice; the resulting matrix is exactly what the certificate
    then talks about.
``angle``
    The position (angle) basis of ``U(1)``, in which the same operator is a dense,
    real, symmetric **circulant** matrix.  Its spectrum is still ``e^{-t n^2}`` in
    closed form, so it is a non-trivial matrix with *known truth* -- the anchor
    that lets a certified bound be checked against the exact answer, and the only
    basis here in which the entrywise-positive Birkhoff-Hopf engine applies.

Nothing in this module is a continuum statement.  A truncation order and a lattice
spacing are inputs, and every result is a statement about the finite matrix that
comes out.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import PI_IV, besseli_iv, cos_iv, exp_iv
from omnibias.geometry.gauge._core.representation import (
    Irrep,
    dimension,
    quadratic_casimir,
)

Scalar = float | int | Fraction
Basis = Literal["character", "angle"]


@dataclass(frozen=True)
class TransferMatrix:
    """A finite, fixed-spacing transfer matrix with enclosed entries.

    Attributes
    ----------
    model
        Which construction produced it (``"u1_heat_kernel"``, ``"su2_wilson"``, ...).
    basis
        ``"character"`` (diagonal) or ``"angle"`` (dense circulant).
    entries
        The matrix, row-major, as outward-rounded enclosures.
    mode_labels
        A human label per basis vector, in the same order as ``entries``.
    exact_eigenvalues
        Enclosures of the true spectrum when it is known in closed form, sorted
        non-increasing, else ``None``.  Present for every constructor here, which
        is what makes them usable as soundness anchors.
    parameters
        Everything that fixed this matrix -- the constructor's name under
        ``"builder"`` plus its arguments -- so a certificate can be replayed by
        rebuilding the matrix from scratch rather than trusting the sealed numbers.
        Scalars that may be rational are stored as round-trippable strings.
    perron_vector
        An approximate dominant eigenvector.  The certified engines are rigorous
        for *any* test vector, so this only affects tightness, never soundness.
    subdominant_vectors
        Approximate eigenvectors for the modes below the dominant one, ordered by
        decreasing eigenvalue.  Feeding the whole chain to
        :func:`~omnibias.core.verified.eig.certified_symmetric_spectral_gap`
        deflates both the degeneracy of the subdominant mode and the polluting
        tail behind it.
    symmetric
        Whether the construction is symmetric by design (all of them are).
    """

    model: str
    basis: str
    entries: tuple[tuple[Interval, ...], ...]
    mode_labels: tuple[str, ...]
    exact_eigenvalues: tuple[Interval, ...] | None
    parameters: Mapping[str, object]
    perron_vector: tuple[float, ...]
    subdominant_vectors: tuple[tuple[float, ...], ...]
    symmetric: bool = True

    @property
    def dimension(self) -> int:
        return len(self.entries)

    def matrix(self) -> list[list[Interval]]:
        """The entries as the nested lists the ``eig`` engines take."""
        return [list(row) for row in self.entries]

    @property
    def entrywise_positive(self) -> bool:
        """Whether every entry is *certifiably* positive (Birkhoff-Hopf's precondition)."""
        return all(cell.lo > 0.0 for row in self.entries for cell in row)

    def exact_subdominant_ratio(self) -> Interval | None:
        """Enclosure of the true ``|lambda_1| / lambda_0``, when the spectrum is known.

        This is the *truth* a certificate is checked against, not part of any
        certificate itself.
        """
        if self.exact_eigenvalues is None or len(self.exact_eigenvalues) < 2:
            return None
        dominant = self.exact_eigenvalues[0]
        if dominant.lo <= 0.0:
            return None
        return self.exact_eigenvalues[1] / dominant


def _scalar_interval(value: Scalar) -> Interval:
    """Enclose a coupling exactly when it is rational, tightly when it is a float."""
    return Interval.from_value(value)


def encode_scalar(value: Scalar) -> str:
    """Round-trippable text for a coupling: ``"3/4"`` stays exact, ``"0.8"`` stays a double."""
    return str(value)


def decode_scalar(text: str) -> Scalar:
    """Inverse of :func:`encode_scalar`, preserving rational-versus-double exactly."""
    return Fraction(text) if "/" in text else float(text)


def _diagonal(values: Sequence[Interval]) -> tuple[tuple[Interval, ...], ...]:
    zero = Interval.point(0.0)
    n = len(values)
    return tuple(
        tuple(values[i] if i == j else zero for j in range(n)) for i in range(n)
    )


def _standard_basis(n: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(1.0 if i == j else 0.0 for j in range(n)) for i in range(n)
    )


def _heat_kernel_eigenvalue(coupling: Scalar, casimir: Fraction) -> Interval:
    """``exp(-t C2)`` from an *exact* rational Casimir: one ``exp_iv`` of a rational."""
    exponent = -(_scalar_interval(coupling) * Interval.from_value(casimir))
    return exp_iv(exponent)


# --------------------------------------------------------------------------- #
# U(1)
# --------------------------------------------------------------------------- #
def u1_heat_kernel_transfer(
    coupling: Scalar,
    *,
    n_max: int = 4,
    basis: Basis = "character",
    lattice_spacing: float = 1.0,
) -> TransferMatrix:
    r"""The ``U(1)`` heat-kernel transfer matrix truncated to ``|n| <= n_max``.

    The spectrum is ``lambda_n = e^{-t n^2}`` for ``n = 0, \pm 1, ..., \pm n_max``,
    so the exact lattice-unit mass gap is ``-ln(lambda_1 / lambda_0) = t``.  Knowing
    the answer in closed form is the point: this is the soundness anchor against
    which a certified lower bound can be checked exactly.

    Every non-zero mode is **doubly degenerate** (``+n`` and ``-n`` share an
    eigenvalue), which is precisely the degeneracy that inflates an undeflated
    power-sum bound by ``sqrt(2)``.

    Parameters
    ----------
    coupling
        The heat-kernel time ``t > 0``.  A :class:`~fractions.Fraction` keeps the
        exponent exactly rational.
    n_max
        Truncation order; the matrix has dimension ``2 n_max + 1``.
    basis
        ``"character"`` for the diagonal form, ``"angle"`` for the dense positive
        circulant on ``2 n_max + 1`` equally spaced angles.  Both have the same
        spectrum by construction.
    lattice_spacing
        Recorded so a gap in lattice units can be converted; it does not change
        the matrix.
    """
    if n_max < 1:
        raise ValueError(f"n_max must be >= 1, got {n_max}")
    if _scalar_interval(coupling).lo <= 0.0:
        raise ValueError(f"coupling must be > 0, got {coupling!r}")

    modes = _u1_modes(n_max)
    eigenvalues = tuple(
        _heat_kernel_eigenvalue(coupling, Fraction(n * n)) for n in modes
    )
    labels = tuple(f"n={n:+d}" if n else "n=0" for n in modes)
    parameters: dict[str, object] = {
        "builder": "u1_heat_kernel_transfer",
        "coupling": encode_scalar(coupling),
        "n_max": int(n_max),
        "basis": basis,
        "lattice_spacing": float(lattice_spacing),
    }
    if basis == "character":
        return TransferMatrix(
            model="u1_heat_kernel",
            basis="character",
            entries=_diagonal(eigenvalues),
            mode_labels=labels,
            exact_eigenvalues=eigenvalues,
            parameters=parameters,
            perron_vector=_standard_basis(len(modes))[0],
            subdominant_vectors=_standard_basis(len(modes))[1:],
            symmetric=True,
        )
    if basis == "angle":
        entries, perron, partners = _u1_angle_basis(eigenvalues, modes)
        return TransferMatrix(
            model="u1_heat_kernel",
            basis="angle",
            entries=entries,
            mode_labels=tuple(f"theta_{j}" for j in range(len(modes))),
            exact_eigenvalues=eigenvalues,
            parameters=parameters,
            perron_vector=perron,
            subdominant_vectors=partners,
            symmetric=True,
        )
    raise ValueError(f"basis must be 'character' or 'angle', got {basis!r}")


def _u1_modes(n_max: int) -> tuple[int, ...]:
    """Winding numbers ordered by decreasing eigenvalue: ``0, +1, -1, +2, -2, ...``."""
    modes: list[int] = [0]
    for n in range(1, n_max + 1):
        modes.extend((n, -n))
    return tuple(modes)


def _u1_angle_basis(
    eigenvalues: Sequence[Interval], modes: Sequence[int]
) -> tuple[tuple[tuple[Interval, ...], ...], tuple[float, ...], tuple[tuple[float, ...], ...]]:
    r"""The same operator as a real symmetric circulant on ``N = 2 n_max + 1`` angles.

    ``T_{jk} = (1/N) \sum_n \lambda_n e^{2 \pi i n (j-k) / N}``.  Choosing exactly
    ``N`` sample angles for ``N`` retained modes makes the discrete Fourier
    transform an exact diagonalisation, so the circulant's spectrum is *the same*
    closed-form ``e^{-t n^2}`` -- a dense, non-diagonal matrix whose eigenvalues
    are nevertheless known exactly.
    """
    n_points = len(modes)
    lam = {mode: eigenvalues[i] for i, mode in enumerate(modes)}
    inverse_n = Interval.from_value(Fraction(1, n_points))

    first_row: list[Interval] = []
    for shift in range(n_points):
        acc = Interval.point(0.0)
        for mode, value in lam.items():
            angle = PI_IV * Interval.from_value(Fraction(2 * mode * shift, n_points))
            acc = acc + value * cos_iv(angle)
        first_row.append(acc * inverse_n)

    entries = tuple(
        tuple(first_row[(j - k) % n_points] for k in range(n_points))
        for j in range(n_points)
    )
    perron = tuple(1.0 for _ in range(n_points))
    # The real degenerate pair for each |n|: cos and sin of the n-th harmonic.
    # These are only *test* vectors, so plain float trig is fine -- the certified
    # engines stay rigorous whatever is handed to them.
    partners: list[tuple[float, ...]] = []
    for mode in modes[1:]:
        phase = [2.0 * math.pi * abs(mode) * j / n_points for j in range(n_points)]
        trig = math.cos if mode > 0 else math.sin
        partners.append(tuple(trig(p) for p in phase))
    return entries, perron, tuple(partners)


# --------------------------------------------------------------------------- #
# SU(N) heat kernel
# --------------------------------------------------------------------------- #
def _su_heat_kernel(
    model: str,
    builder: str,
    coupling: Scalar,
    irreps: Sequence[Irrep],
    max_dynkin: int,
    lattice_spacing: float,
) -> TransferMatrix:
    if _scalar_interval(coupling).lo <= 0.0:
        raise ValueError(f"coupling must be > 0, got {coupling!r}")
    ordered = sorted(irreps, key=lambda r: (quadratic_casimir(r), r.dynkin))
    eigenvalues = tuple(
        _heat_kernel_eigenvalue(coupling, quadratic_casimir(rep)) for rep in ordered
    )
    labels = tuple(
        f"{rep.dynkin} (dim {dimension(rep)}, C2 {quadratic_casimir(rep)})"
        for rep in ordered
    )
    basis_vectors = _standard_basis(len(ordered))
    return TransferMatrix(
        model=model,
        basis="character",
        entries=_diagonal(eigenvalues),
        mode_labels=labels,
        exact_eigenvalues=eigenvalues,
        parameters={
            "builder": builder,
            "coupling": encode_scalar(coupling),
            "max_dynkin": int(max_dynkin),
            "n_irreps": len(ordered),
            "lattice_spacing": float(lattice_spacing),
        },
        perron_vector=basis_vectors[0],
        subdominant_vectors=basis_vectors[1:],
        symmetric=True,
    )


def su2_heat_kernel_transfer(
    coupling: Scalar,
    *,
    max_dynkin: int = 5,
    lattice_spacing: float = 1.0,
) -> TransferMatrix:
    r"""The ``su(2)`` heat-kernel transfer matrix over irreps ``a = 0 .. max_dynkin``.

    Eigenvalues are ``e^{-t C2(R)}`` with ``C2`` the **exact** rational returned by
    :func:`~omnibias.geometry.gauge.quadratic_casimir` (``C2 = a(a+2)/4``, i.e.
    ``j(j+1)`` for spin ``j = a/2``), so each eigenvalue is a single ``exp_iv`` of a
    rational and the exact lattice-unit gap is ``3t/4``.

    ``su(2)`` irreps are self-conjugate, so unlike ``su(3)`` the spectrum here is
    **non-degenerate** -- the clean contrast case.
    """
    if max_dynkin < 1:
        raise ValueError(f"max_dynkin must be >= 1, got {max_dynkin}")
    irreps = [Irrep(n=2, dynkin=(a,)) for a in range(max_dynkin + 1)]
    return _su_heat_kernel(
        "su2_heat_kernel",
        "su2_heat_kernel_transfer",
        coupling,
        irreps,
        max_dynkin,
        lattice_spacing,
    )


def su3_heat_kernel_transfer(
    coupling: Scalar,
    *,
    max_dynkin: int = 2,
    lattice_spacing: float = 1.0,
) -> TransferMatrix:
    r"""The ``su(3)`` heat-kernel transfer matrix over irreps ``(p, q)``, ``p, q <= max_dynkin``.

    The conjugate pair ``(p, q) <-> (q, p)`` shares a Casimir exactly -- ``C2(1,0) =
    C2(0,1) = 4/3`` -- so the subdominant eigenvalue is **doubly degenerate**.  That
    is the degeneracy an undeflated power-sum bound pays a ``sqrt(2)`` inflation
    for, and the reason
    :func:`~omnibias.core.verified.eig.certified_symmetric_spectral_gap` takes a
    partner chain.
    """
    if max_dynkin < 1:
        raise ValueError(f"max_dynkin must be >= 1, got {max_dynkin}")
    irreps = [
        Irrep(n=3, dynkin=(p, q))
        for p in range(max_dynkin + 1)
        for q in range(max_dynkin + 1)
    ]
    return _su_heat_kernel(
        "su3_heat_kernel",
        "su3_heat_kernel_transfer",
        coupling,
        irreps,
        max_dynkin,
        lattice_spacing,
    )


def su2_class_angle_transfer(
    coupling: Scalar,
    *,
    max_dynkin: int = 5,
    lattice_spacing: float = 1.0,
) -> TransferMatrix:
    r"""The ``su(2)`` heat kernel in the **class-angle** basis: dense, positive, isospectral.

    Sampling ``su(2)`` in the character basis is impossible -- that matrix is
    diagonal, so the induced Markov chain never leaves its starting state.  This is
    the same operator written in the configuration variable it can actually be
    simulated in, which is what makes a Monte-Carlo cross-check of the certified
    gap meaningful.

    On the class angles ``theta_j = j pi / (N + 1)``, ``j = 1 .. N``, the ``su(2)``
    characters ``chi_m(theta) = sin(m theta) / sin(theta)`` carry their Haar weight
    ``sin^2 theta`` into the plain sines ``sin(m theta)``, whose discrete transform

    .. math::

        S_{jm} = \sqrt{\tfrac{2}{N + 1}}\, \sin\!\Big(\frac{j m \pi}{N + 1}\Big)

    is real, symmetric and **exactly orthogonal** (``S S = I``).  Hence
    ``T = S diag(lambda) S`` is symmetric and *isospectral* to
    :func:`su2_heat_kernel_transfer` -- a dense matrix whose spectrum
    ``lambda_a = e^{-t C2(a)}`` is still known in closed form, so a certified bound
    on it can still be checked against the exact answer.

    Entries are computed through ``sin A sin B = (cos(A - B) - cos(A + B)) / 2`` so
    only :func:`~omnibias.core.verified.cos_iv` is needed, at exactly rational
    multiples of ``pi``.

    Positivity is *not* automatic: the character sum is truncated, and for small
    ``coupling`` the truncation drives some entries slightly negative.  The matrix is
    returned either way -- it is a perfectly good symmetric operator -- and
    :attr:`TransferMatrix.entrywise_positive` reports the truth, which the
    Birkhoff-Hopf engine and the path-measure sampler both check before running.
    """
    if max_dynkin < 1:
        raise ValueError(f"max_dynkin must be >= 1, got {max_dynkin}")
    if _scalar_interval(coupling).lo <= 0.0:
        raise ValueError(f"coupling must be > 0, got {coupling!r}")

    n_points = max_dynkin + 1
    eigenvalues = tuple(
        _heat_kernel_eigenvalue(coupling, quadratic_casimir(Irrep(n=2, dynkin=(a,))))
        for a in range(n_points)
    )
    entries = _class_angle_entries(eigenvalues)
    vectors = _class_angle_vectors(n_points)
    return TransferMatrix(
        model="su2_class_angle",
        basis="angle",
        entries=entries,
        mode_labels=tuple(f"theta_{j}=pi*{j}/{n_points + 1}" for j in range(1, n_points + 1)),
        exact_eigenvalues=eigenvalues,
        parameters={
            "builder": "su2_class_angle_transfer",
            "coupling": encode_scalar(coupling),
            "max_dynkin": int(max_dynkin),
            "lattice_spacing": float(lattice_spacing),
        },
        perron_vector=vectors[0],
        subdominant_vectors=vectors[1:],
        symmetric=True,
    )


def _class_angle_entries(
    eigenvalues: Sequence[Interval],
) -> tuple[tuple[Interval, ...], ...]:
    r"""``T_{jk} = (2/(N+1)) sum_m lambda_m sin(m j pi/(N+1)) sin(m k pi/(N+1))``."""
    n_points = len(eigenvalues)
    denom = n_points + 1
    scale = Interval.from_value(Fraction(2, denom))

    rows: list[tuple[Interval, ...]] = []
    for j in range(1, n_points + 1):
        row: list[Interval] = []
        for k in range(1, n_points + 1):
            acc = Interval.point(0.0)
            for index, value in enumerate(eigenvalues):
                m = index + 1
                minus = PI_IV * Interval.from_value(Fraction(m * (j - k), denom))
                plus = PI_IV * Interval.from_value(Fraction(m * (j + k), denom))
                acc = acc + value * (cos_iv(minus) - cos_iv(plus)) * Interval.from_value(
                    Fraction(1, 2)
                )
            row.append(acc * scale)
        rows.append(tuple(row))
    return tuple(rows)


def _class_angle_vectors(n_points: int) -> tuple[tuple[float, ...], ...]:
    """The exact eigenvectors: the columns of the orthogonal sine transform.

    Only test vectors, so plain float trig is fine -- the certified engines stay
    rigorous whatever they are handed, and exact vectors merely make them tight.
    """
    denom = n_points + 1
    return tuple(
        tuple(math.sin(m * j * math.pi / denom) for j in range(1, n_points + 1))
        for m in range(1, n_points + 1)
    )


# --------------------------------------------------------------------------- #
# SU(2) Wilson
# --------------------------------------------------------------------------- #
def su2_wilson_transfer(
    beta: float,
    *,
    n_modes: int = 6,
    lattice_spacing: float = 1.0,
) -> TransferMatrix:
    r"""The ``SU(2)`` Wilson transfer matrix from the character expansion.

    The Wilson weight expands on ``SU(2)`` characters as

    .. math::

        e^{\beta \cos\theta}
          = \frac{2}{\beta} \sum_{m \ge 1} m\, I_m(\beta)\, \frac{\sin m\theta}{\sin\theta},

    whose ``m``-th character coefficient is ``(2/\beta) m I_m(\beta)``.  Dividing by
    the dimension ``m`` leaves the per-mode eigenvalue ``\propto I_m(\beta)``, so the
    matrix is diagonal in the character basis with entries
    :func:`~omnibias.core.verified.besseli_iv` at integer orders ``m = 1 .. n_modes``
    (spin ``j = (m-1)/2``).

    This is the hard case the partner chain exists for: ``I_m(\beta)`` decays far
    more slowly in ``m`` than a heat kernel's ``e^{-t C2}``, so the tail behind the
    subdominant mode stays heavy and an undeflated power-sum bound is badly
    polluted by it.
    """
    if n_modes < 2:
        raise ValueError(f"n_modes must be >= 2, got {n_modes}")
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    eigenvalues = tuple(
        besseli_iv(m, Interval.point(float(beta))) for m in range(1, n_modes + 1)
    )
    labels = tuple(f"m={m} (j={(m - 1) / 2})" for m in range(1, n_modes + 1))
    basis_vectors = _standard_basis(n_modes)
    return TransferMatrix(
        model="su2_wilson",
        basis="character",
        entries=_diagonal(eigenvalues),
        mode_labels=labels,
        exact_eigenvalues=eigenvalues,
        parameters={
            "builder": "su2_wilson_transfer",
            "beta": float(beta),
            "n_modes": int(n_modes),
            "lattice_spacing": float(lattice_spacing),
        },
        perron_vector=basis_vectors[0],
        subdominant_vectors=basis_vectors[1:],
        symmetric=True,
    )


def su3_wilson_transfer(
    beta: float,
    *,
    max_dynkin: int = 1,
    n_cells: int = 10,
    lattice_spacing: float = 1.0,
) -> TransferMatrix:
    r"""Diagonal SU(3) Wilson transfer from enclosed Haar character coefficients.

    The Wilson weight ``exp((β/3) Re χ_fund)`` is expanded on SU(3) characters
    by integrating against the Weyl measure on the maximal torus.  Each
    coefficient is an interval enclosure of that integral on a finite box
    partition of ``[0, 2π]²`` (cellwise interval range times cell area; see
    :func:`~omnibias.geometry.gauge.transfer.su3_wilson.su3_wilson_haar_coefficient`).
    The matrix **is** the ``(p, q)`` truncation with ``p, q <= max_dynkin``.

    ``max_dynkin`` is locked at ``1``, ``2``, or ``3``.  Characters with
    ``p, q ≤ 2`` are explicit trigonometric identities; ``p, q = 3`` uses
    the SU(3) Clebsch recurrence from those.  Not a guessed product of
    ordinary Bessel functions.  This is one coupling and one truncation
    -- not 4-D SU(3) Yang-Mills.
    """
    if max_dynkin not in (1, 2, 3):
        raise ValueError(
            f"su3_wilson_transfer locks max_dynkin in {{1, 2, 3}} (Haar characters), "
            f"got {max_dynkin}"
        )
    if float(beta) <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    from omnibias.geometry.gauge.transfer.su3_wilson import su3_wilson_haar_coefficient

    irreps = [
        Irrep(n=3, dynkin=(p, q))
        for p in range(max_dynkin + 1)
        for q in range(max_dynkin + 1)
    ]
    coefficients = tuple(
        su3_wilson_haar_coefficient(rep.dynkin, beta, n_cells=n_cells) for rep in irreps
    )
    ordered = sorted(
        zip(irreps, coefficients, strict=True),
        key=lambda item: (quadratic_casimir(item[0]), item[0].dynkin),
    )
    eigenvalues = tuple(coeff for _rep, coeff in ordered)
    labels = tuple(
        f"{rep.dynkin} (dim {dimension(rep)})" for rep, _coeff in ordered
    )
    return TransferMatrix(
        model="su3_wilson",
        basis="character",
        entries=_diagonal(eigenvalues),
        mode_labels=labels,
        exact_eigenvalues=eigenvalues,
        parameters={
            "builder": "su3_wilson_transfer",
            "beta": float(beta),
            "max_dynkin": int(max_dynkin),
            "n_cells": int(n_cells),
            "n_irreps": len(ordered),
            "lattice_spacing": float(lattice_spacing),
        },
        perron_vector=_standard_basis(len(ordered))[0],
        subdominant_vectors=_standard_basis(len(ordered))[1:],
        symmetric=True,
    )


def _strip_builder(*args: object, **kwargs: object) -> TransferMatrix:
    from omnibias.geometry.gauge.transfer.strip import su2_spatial_strip_transfer

    return su2_spatial_strip_transfer(*args, **kwargs)  # type: ignore[arg-type]


def _torus_builder(*args: object, **kwargs: object) -> TransferMatrix:
    from omnibias.geometry.gauge.transfer.strip import su2_spatial_torus_transfer

    return su2_spatial_torus_transfer(*args, **kwargs)  # type: ignore[arg-type]


#: Constructor name (as recorded in ``TransferMatrix.parameters["builder"]``) to the
#: callable, so a sealed certificate can be replayed by rebuilding its matrix.
BUILDERS: Mapping[str, object] = {
    "u1_heat_kernel_transfer": u1_heat_kernel_transfer,
    "su2_heat_kernel_transfer": su2_heat_kernel_transfer,
    "su2_class_angle_transfer": su2_class_angle_transfer,
    "su3_heat_kernel_transfer": su3_heat_kernel_transfer,
    "su2_wilson_transfer": su2_wilson_transfer,
    "su3_wilson_transfer": su3_wilson_transfer,
    "su2_spatial_strip_transfer": _strip_builder,
    "su2_spatial_torus_transfer": _torus_builder,
}


def rebuild(parameters: Mapping[str, object]) -> TransferMatrix:
    """Reconstruct a transfer matrix from a recorded ``parameters`` block.

    The replay path: a certificate stores only these inputs, so re-deriving the
    matrix from them is genuinely independent of the sealed numbers it will then
    be compared against.
    """
    spec = dict(parameters)
    name = spec.pop("builder", None)
    builder = BUILDERS.get(str(name))
    if builder is None:
        raise ValueError(f"unknown transfer-matrix builder {name!r}")
    spec.pop("n_irreps", None)  # derived, not an input
    if "coupling" in spec:
        first: object = decode_scalar(str(spec.pop("coupling")))
    elif "beta" in spec:
        first = float(str(spec.pop("beta")))
    else:
        raise ValueError("parameters must carry a 'coupling' or 'beta'")
    kwargs = {key: value for key, value in spec.items()}
    if not callable(builder):  # pragma: no cover - registry is callables only
        raise TypeError(f"builder {name!r} is not callable")
    result = builder(first, **kwargs)
    if not isinstance(result, TransferMatrix):  # pragma: no cover - registry invariant
        raise TypeError(f"builder {name!r} did not return a TransferMatrix")
    return result


__all__ = [
    "BUILDERS",
    "TransferMatrix",
    "decode_scalar",
    "encode_scalar",
    "rebuild",
    "su2_class_angle_transfer",
    "su2_heat_kernel_transfer",
    "su2_wilson_transfer",
    "su3_heat_kernel_transfer",
    "su3_wilson_transfer",
    "u1_heat_kernel_transfer",
]
