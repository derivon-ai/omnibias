# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Planted-clique / random-CSP degree-indexed SOS certificates."""

from __future__ import annotations

from omnibias.sos.combinatorial import (
    brute_force_max_clique,
    planted_clique_adjacency,
    planted_clique_degree_certificate,
    random_csp_degree_certificate,
)


def test_planted_clique_oracle_and_sos_sphere() -> None:
    adj = planted_clique_adjacency(4, 3)
    assert brute_force_max_clique(adj) == 3
    cert = planted_clique_degree_certificate(n=4, clique_size=3, half_degree=1)
    assert cert.oracle_omega == 3
    assert cert.sos_proved
    assert cert.wall_seconds >= 0.0


def test_random_csp_degree_certificate_reports_oracle() -> None:
    cert = random_csp_degree_certificate(n=3, n_clauses=2, half_degree=1, seed=0)
    assert 0 <= cert.oracle_unsat <= cert.n_clauses
    assert cert.sos_proved
    assert cert.wall_seconds >= 0.0
