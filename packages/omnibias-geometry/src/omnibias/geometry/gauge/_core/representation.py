# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite-dimensional representation theory of ``su(N)`` (pure Python + numpy).

An :class:`Irrep` is a finite-dimensional irreducible representation of
``su(N)`` labelled by its **Dynkin labels** ``(a_1, ..., a_{N-1})`` (non-negative
integers), equivalently the highest weight ``lambda = sum_i a_i omega_i``. We
work in the ``epsilon`` (partition) basis: the Dynkin labels map to a partition
``lambda_i`` with ``a_i = lambda_i - lambda_{i+1}`` and ``lambda_N = 0``.

Exact invariants (closed rational / integer formulas)
-----------------------------------------------------
- :func:`dimension` -- Weyl dimension formula
  ``dim = prod_{i<j} (lambda_i - lambda_j + j - i)/(j - i)``.
- :func:`quadratic_casimir` -- the highest-weight eigenvalue in the physics
  normalization ``tr_fund(T^a T^b) = 1/2 delta^{ab}``:
  ``C2 = 1/2 [ sum_i lambda_i (lambda_i + N + 1 - 2 i) - (sum_i lambda_i)^2 / N ]``.
- :func:`dynkin_index` -- ``T(R) = C2(R) dim(R) / dim(G)``, ``dim(G) = N^2 - 1``.

Weight system + products (rigorous algorithms)
----------------------------------------------
- :func:`weight_multiplicities` -- Freudenthal recursion (dominant weights) plus
  Weyl-orbit expansion; for ``su(N)`` these are the Kostka numbers of the Schur
  function.
- :func:`tensor_product_decomposition` -- the Racah-Speiser / Brauer-Klimyk
  algorithm (Weyl-group reflection of ``lambda_A + rho + beta`` for every weight
  ``beta`` of ``B``).
- :func:`branching_to_subalgebra` -- restriction ``su(N) -> su(N-1)`` via the
  Gelfand-Tsetlin interlacing rule (sum over the ``u(1)`` charge).
- :func:`character` -- the Weyl character = Schur polynomial via the bialternant
  ``det(x_i^{lambda_j + N - j}) / det(x_i^{N - j})``.

Explicit representation matrices
--------------------------------
- :func:`su2_spin_matrices` -- the spin-``j`` angular-momentum matrices
  ``(J_x, J_y, J_z)`` (``2 j + 1`` dimensional).
- :func:`adjoint_rep_matrices` -- ``(T^a_{adj})_{bc} = -i f^{abc}`` from a
  :class:`~omnibias.geometry.gauge._core.lie_algebra.LieAlgebra`.

Out of thesis
-------------
Finite (non-Lie) group character tables and general abstract-group theory are
**not** implemented (documented in ``docs/scope-and-guarantees.md``); this module
is the Lie-algebra / highest-weight slice only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import (
    combinations,
    combinations_with_replacement,
    permutations,
    product,
)

import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complex128]

# --------------------------------------------------------------------------- #
# Irrep dataclass
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Irrep:
    """An irreducible representation of ``su(N)`` by its Dynkin labels.

    Parameters
    ----------
    n
        The rank parameter ``N`` of ``su(N)`` (``N >= 2``).
    dynkin
        The ``N - 1`` non-negative Dynkin labels ``(a_1, ..., a_{N-1})``.
    """

    n: int
    dynkin: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError(f"su(N) needs N >= 2, got N={self.n}")
        if len(self.dynkin) != self.n - 1:
            raise ValueError(
                f"su({self.n}) needs {self.n - 1} Dynkin labels, got {self.dynkin!r}"
            )
        if any((not isinstance(a, int)) or a < 0 for a in self.dynkin):
            raise ValueError(f"Dynkin labels must be non-negative ints: {self.dynkin!r}")

    @property
    def algebra(self) -> str:
        return f"su({self.n})"

    def partition(self) -> tuple[int, ...]:
        """The highest weight as a length-``N`` partition (``lambda_N = 0``)."""
        return _dynkin_to_partition(self.dynkin, self.n)

    def __repr__(self) -> str:
        return f"Irrep(su({self.n}), dynkin={self.dynkin})"


def irrep(n: int, *dynkin: int) -> Irrep:
    """Convenience constructor: ``irrep(3, 1, 0)`` is the ``su(3)`` fundamental."""
    return Irrep(n=n, dynkin=tuple(dynkin))


def fundamental(n: int) -> Irrep:
    """The defining ``N``-dimensional representation ``[1, 0, ..., 0]``."""
    return Irrep(n=n, dynkin=(1,) + (0,) * (n - 2))


def adjoint(n: int) -> Irrep:
    """The adjoint ``(N^2 - 1)``-dimensional representation ``[1, 0, ..., 0, 1]``."""
    if n == 2:
        return Irrep(n=2, dynkin=(2,))
    return Irrep(n=n, dynkin=(1,) + (0,) * (n - 3) + (1,))


def trivial(n: int) -> Irrep:
    """The 1-dimensional trivial representation ``[0, ..., 0]``."""
    return Irrep(n=n, dynkin=(0,) * (n - 1))


# --------------------------------------------------------------------------- #
# partition <-> Dynkin helpers
# --------------------------------------------------------------------------- #


def _dynkin_to_partition(dynkin: tuple[int, ...], n: int) -> tuple[int, ...]:
    """Partition ``lambda`` (length ``N``, ``lambda_N = 0``) from Dynkin labels."""
    parts = [0] * n
    for i in range(n - 2, -1, -1):
        parts[i] = parts[i + 1] + dynkin[i]
    return tuple(parts)


def _partition_to_dynkin(part: tuple[int, ...]) -> tuple[int, ...]:
    """Dynkin labels ``a_i = lambda_i - lambda_{i+1}`` from a partition."""
    return tuple(part[i] - part[i + 1] for i in range(len(part) - 1))


def _normalize_partition(vec: tuple[int, ...]) -> tuple[int, ...]:
    """Shift a dominant weight so the smallest entry is 0 (su(N) equivalence)."""
    shift = vec[-1]
    return tuple(v - shift for v in vec)


# --------------------------------------------------------------------------- #
# exact invariants
# --------------------------------------------------------------------------- #


def dimension(rep: Irrep) -> int:
    """Weyl dimension formula (exact integer)."""
    lam = rep.partition()
    n = rep.n
    num = Fraction(1)
    den = Fraction(1)
    for i in range(n):
        for j in range(i + 1, n):
            num *= lam[i] - lam[j] + (j - i)
            den *= j - i
    result = num / den
    if result.denominator != 1:  # pragma: no cover -- guards a formula regression
        raise ArithmeticError(f"non-integer dimension {result} for {rep!r}")
    return int(result)


def quadratic_casimir(rep: Irrep) -> Fraction:
    """Quadratic Casimir eigenvalue ``C2`` (``tr_fund(T^aT^b)=1/2 delta``)."""
    lam = rep.partition()
    n = rep.n
    total = sum(lam)
    acc = Fraction(0)
    for i in range(n):  # i is 0-based; formula uses 1-based index (i+1)
        acc += lam[i] * (lam[i] + n + 1 - 2 * (i + 1))
    return Fraction(1, 2) * (acc - Fraction(total * total, n))


def dynkin_index(rep: Irrep) -> Fraction:
    """Dynkin index ``T(R) = C2(R) dim(R) / dim(G)`` (``1/2`` for the fundamental)."""
    dim_g = rep.n * rep.n - 1
    return quadratic_casimir(rep) * dimension(rep) / dim_g


def dual_coxeter_number(n: int) -> int:
    """Dual Coxeter number ``h^vee = N`` of ``su(N)`` (the adjoint Casimir)."""
    return n


# --------------------------------------------------------------------------- #
# weight system (Freudenthal + Weyl orbits)
# --------------------------------------------------------------------------- #


def _positive_roots(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _partitions_leq(lam: tuple[int, ...], n: int) -> list[tuple[int, ...]]:
    """Dominant weights below ``lam``: partitions of ``|lam|`` (<= N parts) that
    are ``<=`` ``lam`` in dominance order, sorted from ``lam`` downward."""
    total = sum(lam)
    top = lam[0]
    out: list[tuple[int, ...]] = []

    def rec(prefix: list[int], remaining: int, max_part: int) -> None:
        k = len(prefix)
        if k == n:
            if remaining == 0:
                out.append(tuple(prefix))
            return
        # each remaining part <= max_part, weakly decreasing, fill n-k slots
        hi = min(max_part, remaining)
        lo = -(-remaining // (n - k))  # ceil(remaining/(n-k)) keeps it feasible
        for p in range(hi, lo - 1, -1):
            rec([*prefix, p], remaining - p, p)

    rec([], total, top)
    lam_ps = _prefix_sums(lam)
    dominant = [mu for mu in out if _dominates(lam_ps, _prefix_sums(mu))]
    dominant.sort(key=lambda mu: _prefix_sums(mu), reverse=True)
    return dominant


def _prefix_sums(vec: tuple[int, ...]) -> tuple[int, ...]:
    acc = 0
    out = []
    for v in vec:
        acc += v
        out.append(acc)
    return tuple(out)


def _dominates(lam_ps: tuple[int, ...], mu_ps: tuple[int, ...]) -> bool:
    return all(lp >= mp for lp, mp in zip(lam_ps, mu_ps, strict=True))


def _dominant_form(vec: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(vec, reverse=True))


@cache
def _dominant_multiplicities(rep: Irrep) -> dict[tuple[int, ...], int]:
    """Freudenthal multiplicities on the dominant weights below the h.w."""
    lam = rep.partition()
    n = rep.n
    rho = tuple(n - 1 - i for i in range(n))
    doms = _partitions_leq(lam, n)
    mult: dict[tuple[int, ...], int] = {lam: 1}
    lam_rho2 = _norm2(_add(lam, rho))
    roots = _positive_roots(n)
    kmax = sum(lam) + n
    for mu in doms:
        if mu == lam:
            continue
        acc = 0
        for i, j in roots:
            alpha = _root_vec(i, j, n)
            for k in range(1, kmax + 1):
                nu = _add(mu, _scale(alpha, k))
                md = mult.get(_dominant_form(nu), 0)
                if md == 0:
                    continue
                acc += _dot(nu, alpha) * md
        denom = lam_rho2 - _norm2(_add(mu, rho))
        value = Fraction(2 * acc, denom)
        if value.denominator != 1:  # pragma: no cover -- formula regression guard
            raise ArithmeticError(f"non-integer weight multiplicity {value}")
        mult[mu] = int(value)
    return mult


def weight_multiplicities(rep: Irrep) -> dict[tuple[int, ...], int]:
    """Full weight system: every weight (``epsilon`` basis) with its multiplicity."""
    out: dict[tuple[int, ...], int] = {}
    for mu, m in _dominant_multiplicities(rep).items():
        for perm in set(permutations(mu)):
            out[perm] = m
    return out


# small integer-vector helpers (epsilon basis) ----------------------------- #
def _add(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _scale(a: tuple[int, ...], k: int) -> tuple[int, ...]:
    return tuple(k * x for x in a)


def _dot(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm2(a: tuple[int, ...]) -> int:
    return _dot(a, a)


def _root_vec(i: int, j: int, n: int) -> tuple[int, ...]:
    v = [0] * n
    v[i] = 1
    v[j] = -1
    return tuple(v)


def _sort_desc_with_sign(vec: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """Descending sort with the sign of the sorting permutation (bubble parity)."""
    seq = list(vec)
    sign = 1
    for a in range(len(seq)):
        for b in range(a + 1, len(seq)):
            if seq[a] < seq[b]:
                seq[a], seq[b] = seq[b], seq[a]
                sign = -sign
    return tuple(seq), sign


# --------------------------------------------------------------------------- #
# tensor product (Racah-Speiser / Brauer-Klimyk)
# --------------------------------------------------------------------------- #


def tensor_product_decomposition(a: Irrep, b: Irrep) -> dict[Irrep, int]:
    """Decompose ``a (x) b`` into irreps with multiplicities (same ``su(N)``)."""
    if a.n != b.n:
        raise ValueError(f"cannot tensor su({a.n}) with su({b.n})")
    n = a.n
    lam = a.partition()
    rho = tuple(n - 1 - i for i in range(n))
    result: Counter[tuple[int, ...]] = Counter()
    for beta, m in weight_multiplicities(b).items():
        gamma = _add(_add(lam, rho), beta)
        srt, sign = _sort_desc_with_sign(gamma)
        if len(set(srt)) != n:  # hit a Weyl wall -> contributes zero
            continue
        part = tuple(srt[i] - rho[i] for i in range(n))
        norm = _normalize_partition(part)
        result[norm] += sign * m
    return {
        Irrep(n=n, dynkin=_partition_to_dynkin(p)): c
        for p, c in sorted(result.items(), reverse=True)
        if c != 0
    }


def branching_to_subalgebra(rep: Irrep) -> dict[Irrep, int]:
    """Branch ``su(N) -> su(N-1)`` via the Gelfand-Tsetlin interlacing rule.

    Every partition ``mu`` (length ``N-1``) that *interlaces* the highest weight
    ``lambda`` (``lambda_1 >= mu_1 >= lambda_2 >= mu_2 >= ... >= lambda_N``)
    contributes one ``su(N-1)`` weight; summing over the ``u(1)`` charge gives the
    ``su(N-1)`` irrep content. Requires ``N >= 3``. Validated by ``8 -> 3+2+2+1``
    (``su(3) -> su(2)``).
    """
    n = rep.n
    if n < 3:
        raise ValueError("branching su(N)->su(N-1) needs N >= 3")
    lam = rep.partition()
    out: Counter[tuple[int, ...]] = Counter()
    ranges = [range(lam[i + 1], lam[i] + 1) for i in range(n - 1)]
    for mu in product(*ranges):
        out[_partition_to_dynkin(mu)] += 1
    return {Irrep(n=n - 1, dynkin=d): c for d, c in out.items() if c != 0}


# --------------------------------------------------------------------------- #
# Weyl character = Schur polynomial (bialternant)
# --------------------------------------------------------------------------- #


def character(rep: Irrep, x: npt.NDArray[np.complex128]) -> complex:
    """Weyl character at a torus point ``x`` (an ``N``-vector); the Schur poly.

    ``chi = det(x_i^{lambda_j + N - j}) / det(x_i^{N - j})`` (the bialternant).
    ``x`` should have distinct entries with ``prod(x) = 1`` for ``su(N)``.
    """
    lam = rep.partition()
    n = rep.n
    xv = np.asarray(x, dtype=np.complex128).reshape(-1)
    if xv.shape[0] != n:
        raise ValueError(f"character needs an {n}-vector, got {xv.shape[0]}")
    expo_top = [lam[j] + n - 1 - j for j in range(n)]
    expo_bot = [n - 1 - j for j in range(n)]
    top = np.array([[xv[i] ** e for e in expo_top] for i in range(n)])
    bot = np.array([[xv[i] ** e for e in expo_bot] for i in range(n)])
    return complex(np.linalg.det(top) / np.linalg.det(bot))


# --------------------------------------------------------------------------- #
# explicit representation matrices
# --------------------------------------------------------------------------- #


def su2_spin_matrices(two_j: int) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    """Spin-``j`` angular-momentum matrices ``(J_x, J_y, J_z)`` (``two_j = 2j``).

    Satisfy ``[J_a, J_b] = i eps_{abc} J_c`` with ``J^2 = j(j+1) I`` and
    dimension ``2 j + 1``. ``two_j`` is a non-negative integer (``2j``).
    """
    if two_j < 0:
        raise ValueError(f"two_j must be >= 0, got {two_j}")
    dim = two_j + 1
    j = Fraction(two_j, 2)
    ms = [j - k for k in range(dim)]  # m = j, j-1, ..., -j
    jz = np.diag([complex(float(m)) for m in ms])
    jp = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(1, dim):
        m = ms[k]  # J_+ |m> = sqrt(j(j+1)-m(m+1)) |m+1>
        coeff = float(j * (j + 1) - m * (m + 1))
        jp[k - 1, k] = np.sqrt(coeff)
    jm = jp.conj().T
    jx = 0.5 * (jp + jm)
    jy = (jp - jm) / 2.0j
    return jx, jy, jz


def adjoint_rep_matrices(structure_constants: npt.NDArray[np.float64]) -> ComplexArray:
    """Adjoint generators ``(T^a_{adj})_{bc} = -i f^{abc}`` from ``f^{abc}``.

    ``structure_constants`` is the ``(dim, dim, dim)`` array ``f^{abc}``; the
    returned Hermitian matrices satisfy ``[T^a, T^b] = i f^{abc} T^c`` and
    ``tr(T^a_{adj} T^b_{adj}) = N delta^{ab}`` (the dual Coxeter number).
    """
    f = np.asarray(structure_constants, dtype=np.float64)
    return (-1j) * f.astype(np.complex128)


def symmetric_power(n: int, k: int) -> Irrep:
    """Highest weight of ``Sym^k`` of the ``su(N)`` fundamental: Dynkin ``[k,0..0]``."""
    if k < 0:
        raise ValueError(f"tensor power k must be >= 0, got {k}")
    return Irrep(n=n, dynkin=(k,) + (0,) * (n - 2))


def antisymmetric_power(n: int, k: int) -> Irrep:
    """Highest weight of ``Lambda^k`` of the ``su(N)`` fundamental (``0 <= k <= N``).

    ``Lambda^k`` has Dynkin label ``1`` in slot ``k``; ``Lambda^0`` and
    ``Lambda^N`` are the trivial (determinant) representation.
    """
    if not 0 <= k <= n:
        raise ValueError(f"antisymmetric power needs 0 <= k <= {n}, got {k}")
    if k in (0, n):
        return trivial(n)
    dynk = [0] * (n - 1)
    dynk[k - 1] = 1
    return Irrep(n=n, dynkin=tuple(dynk))


def symmetric_power_rep_matrices(generators: ComplexArray, k: int) -> ComplexArray:
    r"""``Sym^k`` rep matrices from fundamental generators (bosonic Fock formula).

    ``generators`` is a ``(g, N, N)`` stack of Hermitian generators ``T^a``. The
    returned ``(g, D, D)`` stack (``D = binom(N+k-1, k)``) acts on the symmetric
    power as the derivation ``rho(T) = sum_{pq} T_{pq} a^+_p a_q`` in the
    orthonormal occupation-number basis, so it is Hermitian and satisfies
    ``[rho(T^a), rho(T^b)] = i f^{abc} rho(T^c)``.
    """
    return _fock_rep_matrices(generators, k, fermionic=False)


def antisymmetric_power_rep_matrices(generators: ComplexArray, k: int) -> ComplexArray:
    r"""``Lambda^k`` rep matrices from fundamental generators (fermionic formula).

    ``generators`` is a ``(g, N, N)`` stack of Hermitian generators ``T^a``. The
    returned ``(g, D, D)`` stack (``D = binom(N, k)``) acts on the antisymmetric
    power as ``rho(T) = sum_{pq} T_{pq} c^+_p c_q`` with Jordan-Wigner signs in the
    ordered occupation basis, so it is Hermitian and satisfies
    ``[rho(T^a), rho(T^b)] = i f^{abc} rho(T^c)``.
    """
    return _fock_rep_matrices(generators, k, fermionic=True)


def _fock_rep_matrices(
    generators: ComplexArray, k: int, *, fermionic: bool,
) -> ComplexArray:
    gens = np.asarray(generators, dtype=np.complex128)
    if gens.ndim != 3 or gens.shape[1] != gens.shape[2]:
        raise ValueError(f"generators must be (g, N, N), got {gens.shape}")
    ngen, n, _ = gens.shape
    if k < 0 or (fermionic and k > n):
        raise ValueError(f"invalid tensor power k={k} for N={n}")

    if fermionic:
        states = [tuple(s) for s in combinations(range(n), k)]
        occ = {s: _occupation(s, n) for s in states}
    else:
        states = [
            _occupation(t, n) for t in combinations_with_replacement(range(n), k)
        ]
        occ = {s: s for s in states}
    index = {s: i for i, s in enumerate(states)}
    dim = len(states)
    out = np.zeros((ngen, dim, dim), dtype=np.complex128)

    for state in states:
        col = index[state]
        counts = occ[state]
        for q in range(n):
            if counts[q] == 0:
                continue
            annih, sign_q, amp_q = _lower(counts, q, fermionic)
            for p in range(n):
                created = _raise(annih, p, fermionic)
                if created is None:
                    continue
                new_state, sign_p, amp_p = created
                row = index[_state_key(new_state, states_are_subsets=fermionic)]
                out[:, row, col] += gens[:, p, q] * (sign_q * sign_p * amp_q * amp_p)
    return out


def _occupation(indices: tuple[int, ...], n: int) -> tuple[int, ...]:
    counts = [0] * n
    for i in indices:
        counts[i] += 1
    return tuple(counts)


def _state_key(counts: tuple[int, ...], *, states_are_subsets: bool) -> tuple[int, ...]:
    if states_are_subsets:
        return tuple(i for i, c in enumerate(counts) for _ in range(c))
    return counts


def _lower(
    counts: tuple[int, ...], q: int, fermionic: bool,
) -> tuple[tuple[int, ...], float, float]:
    new = list(counts)
    if fermionic:
        sign = (-1.0) ** sum(counts[:q])
        new[q] = 0
        return tuple(new), sign, 1.0
    amp = float(np.sqrt(counts[q]))
    new[q] -= 1
    return tuple(new), 1.0, amp


def _raise(
    counts: tuple[int, ...], p: int, fermionic: bool,
) -> tuple[tuple[int, ...], float, float] | None:
    new = list(counts)
    if fermionic:
        if counts[p] == 1:
            return None
        sign = (-1.0) ** sum(counts[:p])
        new[p] = 1
        return tuple(new), sign, 1.0
    amp = float(np.sqrt(counts[p] + 1))
    new[p] += 1
    return tuple(new), 1.0, amp


__all__ = [
    "Irrep",
    "adjoint",
    "adjoint_rep_matrices",
    "antisymmetric_power",
    "antisymmetric_power_rep_matrices",
    "branching_to_subalgebra",
    "character",
    "dimension",
    "dual_coxeter_number",
    "dynkin_index",
    "fundamental",
    "irrep",
    "quadratic_casimir",
    "su2_spin_matrices",
    "symmetric_power",
    "symmetric_power_rep_matrices",
    "tensor_product_decomposition",
    "trivial",
    "weight_multiplicities",
]
