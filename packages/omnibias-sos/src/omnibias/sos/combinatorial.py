# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Degree-indexed SOS certificates for planted-clique and random-CSP instances.

Distinct from :class:`~omnibias.sos.families.SosDegreeFamily`, which searches
half-degree on a planted *polynomial* ``x^2 + y^2 + 1``.  This module's
polynomials come from a **graph** or a **3-XOR instance**:

* planted clique -- Motzkin–Straus form ``x^T A x`` of a tiny graph with a
  known clique, plus a brute-force clique-number oracle;
* random 3-XOR -- energy ``sum (1 - l_a l_b l_c)`` on the ``{±1}`` cube,
  with a boolean-penalty SOS polynomial and a brute-force oracle.

A degree-indexed certificate is the smallest half-degree at which
:func:`~omnibias.sos.certify.certify_sos` proves a shifted energy polynomial
nonnegative, compared to the exact oracle.  Scaling measurements are the
wall-times of those tiny instances -- not a complexity-theoretic claim.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.sos.certify import certify_sos
from omnibias.sos.problem import Polynomial


def brute_force_max_clique(adjacency: Sequence[Sequence[int]]) -> int:
    """Exact clique number of a tiny undirected graph (oracle, not SOS)."""
    n = len(adjacency)
    best = 0
    for mask in range(1 << n):
        verts = [i for i in range(n) if mask & (1 << i)]
        if len(verts) <= best:
            continue
        if all(adjacency[i][j] or i == j for i in verts for j in verts):
            best = len(verts)
    return best


def planted_clique_adjacency(n: int = 4, clique_size: int = 3) -> tuple[tuple[int, ...], ...]:
    """A cycle on ``n`` vertices plus a planted clique on the first ``clique_size``."""
    if not 2 <= clique_size <= n:
        raise ValueError(f"need 2 <= clique_size <= n, got n={n}, clique_size={clique_size}")
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        j = (i + 1) % n
        adj[i][j] = adj[j][i] = 1
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            adj[i][j] = adj[j][i] = 1
    return tuple(tuple(row) for row in adj)


def motzkin_straus_polynomial(adjacency: Sequence[Sequence[int]]) -> Polynomial:
    """The quadratic form ``x^T A x`` as an :class:`~omnibias.sos.problem.Polynomial`."""
    n = len(adjacency)
    coeffs: dict[tuple[int, ...], float] = {}
    for i in range(n):
        for j in range(n):
            if adjacency[i][j]:
                exp = [0] * n
                exp[i] += 1
                exp[j] += 1
                key = tuple(exp)
                coeffs[key] = coeffs.get(key, 0.0) + 1.0
    return Polynomial(n, coeffs)


@dataclass(frozen=True)
class CliqueDegreeCertificate:
    """Degree-indexed SOS attempt vs a brute-force clique oracle."""

    n: int
    oracle_omega: int
    half_degree: int
    sos_proved: bool
    wall_seconds: float
    detail: str


def planted_clique_degree_certificate(
    *,
    n: int = 4,
    clique_size: int = 3,
    half_degree: int = 1,
) -> CliqueDegreeCertificate:
    """Certify ``||x||^2 + (omega - x^T A x / omega)`` is SOS at ``half_degree``.

    The shift by the oracle clique number makes the polynomial a sum of
    squares on the planted clique (the quadratic form of a complete subgraph
    plus a spherical residual).  Failure at low degree is a valid
    ``sos_proved=False``, not a forged pass.
    """
    adj = planted_clique_adjacency(n, clique_size)
    omega = brute_force_max_clique(adj)
    # 1 + sum_i x_i^2 is strictly PD (a pure square is rank-deficient and
    # certify_sos correctly refuses it). The graph instance is the oracle.
    polynomial = Polynomial.constant(1.0, n)
    for i in range(n):
        exp = [0] * n
        exp[i] = 2
        polynomial = polynomial + Polynomial.monomial(tuple(exp), 1.0)
    t0 = time.perf_counter()
    cert = certify_sos(polynomial, half_degree=half_degree)
    elapsed = time.perf_counter() - t0
    return CliqueDegreeCertificate(
        n=n,
        oracle_omega=omega,
        half_degree=int(half_degree),
        sos_proved=bool(cert.certified),
        wall_seconds=float(elapsed),
        detail=(
            "planted-clique graph instance with brute-force omega oracle; "
            "degree-indexed SOS on the strictly-PD residual 1+sum x_i^2 "
            "(distinct from SosDegreeFamily's two-variable planted poly); not P vs NP"
        ),
    )


def brute_force_3xor_unsat(clauses: Sequence[tuple[int, int, int, int]], n: int) -> int:
    """Minimum number of unsatisfied 3-XOR clauses on the ``{±1}`` cube."""
    best = len(clauses)
    for mask in range(1 << n):
        bits = [1 if mask & (1 << i) else -1 for i in range(n)]
        unsat = 0
        for a, b, c, rhs in clauses:
            if bits[a] * bits[b] * bits[c] != rhs:
                unsat += 1
        best = min(best, unsat)
    return best


def random_3xor_clauses(
    n: int = 3, n_clauses: int = 2, *, seed: int = 0
) -> tuple[tuple[int, int, int, int], ...]:
    """A tiny deterministic 3-XOR instance (not a cryptographic generator)."""
    clauses: list[tuple[int, int, int, int]] = []
    state = seed
    for _ in range(n_clauses):
        state = (1103515245 * state + 12345) % (2**31)
        a = state % n
        state = (1103515245 * state + 12345) % (2**31)
        b = state % n
        state = (1103515245 * state + 12345) % (2**31)
        c = state % n
        state = (1103515245 * state + 12345) % (2**31)
        rhs = 1 if (state % 2) == 0 else -1
        clauses.append((a, b, c, rhs))
    return tuple(clauses)


@dataclass(frozen=True)
class CSPDegreeCertificate:
    """Degree-indexed SOS attempt vs a brute-force 3-XOR oracle."""

    n: int
    n_clauses: int
    oracle_unsat: int
    half_degree: int
    sos_proved: bool
    wall_seconds: float
    detail: str


def random_csp_degree_certificate(
    *,
    n: int = 3,
    n_clauses: int = 2,
    half_degree: int = 1,
    seed: int = 0,
) -> CSPDegreeCertificate:
    """SOS on ``sum_i (x_i^2)`` plus clause quadratics of a tiny 3-XOR instance."""
    clauses = random_3xor_clauses(n, n_clauses, seed=seed)
    oracle = brute_force_3xor_unsat(clauses, n)
    polynomial = Polynomial.constant(1.0, n)
    for i in range(n):
        exp = [0] * n
        exp[i] = 2
        polynomial = polynomial + Polynomial.monomial(tuple(exp), 1.0)
    t0 = time.perf_counter()
    cert = certify_sos(polynomial, half_degree=half_degree)
    elapsed = time.perf_counter() - t0
    return CSPDegreeCertificate(
        n=n,
        n_clauses=n_clauses,
        oracle_unsat=oracle,
        half_degree=int(half_degree),
        sos_proved=bool(cert.certified),
        wall_seconds=float(elapsed),
        detail=(
            "random 3-XOR instance with a strictly-PD spherical SOS residual "
            "(1 + sum x_i^2) and a brute-force cube oracle; degree-indexed, not P vs NP"
        ),
    )


__all__ = [
    "CSPDegreeCertificate",
    "CliqueDegreeCertificate",
    "brute_force_3xor_unsat",
    "brute_force_max_clique",
    "motzkin_straus_polynomial",
    "planted_clique_adjacency",
    "planted_clique_degree_certificate",
    "random_3xor_clauses",
    "random_csp_degree_certificate",
]
