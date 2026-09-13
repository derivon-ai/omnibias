# SPDX-License-Identifier: Apache-2.0
"""Actual SU(2) Hamiltonian-vacuum conditional Poincare bounds.

The partial-Bochner proof in docs/api/gauge-vacuum-local.md applies to the
positive groundstate on the entire product of link groups, with no spin
truncation. Local constants depend on weighted plaquette incidence, not
total volume. A separate dense Dobrushin gate can certify a finite-graph
gap; its row sums generally grow with volume. Neither is a continuum claim.
"""
from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import PI_IV, certificate_mode, exp_iv
from omnibias.geometry.gauge.transfer.static_sources import (
    Edge,
    _cycles,
    _graph,
    _integer,
    _rational,
)


def _comparison_floor(g: Fraction, incidence: Fraction) -> tuple[Fraction, Fraction]:
    """Enclose (3/4) exp(-8*pi*g*incidence), NOT the actual spectral gap."""
    if incidence == 0:
        return Fraction(3, 4), Fraction(3, 4)
    exponent = -PI_IV * Interval.from_value(8 * g * incidence)
    result = exp_iv(exponent) * Interval.from_value(Fraction(3, 4))
    return max(Fraction(0), Fraction(result.lo)), Fraction(result.hi)


def su2_vacuum_local_bounds(
    n_vertices: int,
    edges: Sequence[Edge],
    *,
    kappa: int | Fraction = 1,
    plaquettes: Sequence[Sequence[int]] = (),
    magnetic_weights: Sequence[int | Fraction] | None = None,
    blocks: Sequence[Sequence[int]] = (),
) -> dict[str, Any]:
    """Seal local inequalities for the actual density psi_0**2 relative to Haar.

    Electric weights are one. Plaquettes are simple oriented cycles with
    signed one-based edge indices. Optional blocks contain distinct positive
    one-based edge indices; they need not cover the graph. Singleton bounds
    are always computed, independently of these optional blocks.

    PASS means every local comparison floor has a strictly positive certified
    lower endpoint. The additional finite_graph_gap_verified flag requires a
    separate actual influence-row-sum gate q<1. Underflow never earns positive
    numeric bounds. All analytic implications are written, not Lean checked.
    """
    graph = _graph(n_vertices, edges)
    if not graph:
        raise ValueError("at least one graph edge is required")
    coupling = _rational(kappa, "kappa")
    if coupling <= 0:
        raise ValueError("kappa must be strictly positive")
    loops = _cycles(graph, plaquettes)
    magnetic = tuple(_rational(x, "magnetic weight") for x in (
        [1] * len(loops) if magnetic_weights is None else magnetic_weights))
    if len(magnetic) != len(loops) or any(x < 0 for x in magnetic):
        raise ValueError("one nonnegative magnetic weight per plaquette is required")
    selected: list[tuple[int, ...]] = []
    for block in blocks:
        tokens = tuple(_integer(x, "block edge") for x in block)
        if not tokens or len(set(tokens)) != len(tokens):
            raise ValueError("blocks must be nonempty and contain distinct edge indices")
        if any(x < 1 or x > len(graph) for x in tokens):
            raise ValueError("block edges use positive one-based edge indices")
        selected.append(tokens)
    incidence = [Fraction(0)] * len(graph)
    for loop, weight in zip(loops, magnetic, strict=True):
        for token in loop:
            incidence[abs(token) - 1] += weight
    g = 4 / coupling**2
    block_incidence = [sum((incidence[i - 1] for i in block), Fraction(0))
                       for block in selected]
    # The dense matrix is A_ij <= 4*pi*g*min(d_i,d_j), i != j.
    # Its conservative weighted-row test here uses v_i=1. A sharper local
    # matrix must control the TRUE log-vacuum's connected tails first.
    pi_upper = Fraction(PI_IV.hi)
    row_upper = [4 * pi_upper * g * sum((min(d, other)
                 for j, other in enumerate(incidence) if j != i), Fraction(0))
                 for i, d in enumerate(incidence)]
    q = max(row_upper)
    with certificate_mode():
        floors = {d: _comparison_floor(g, d) for d in set(incidence + block_incidence)}
        singleton_floors = [floors[d] for d in incidence]
        gamma = min(pair[0] for pair in singleton_floors)
        local_pass = all(pair[0] > 0 for pair in floors.values())
        factorization_pass = q < 1
        delta = coupling * gamma * (1 - q) / 2 if factorization_pass else Fraction(0)
        witness = {
            "model": "su2_actual_vacuum_local_v1",
            "normalization": "aH=kappa/2*sum(C_e)+2/kappa*sum(v_p*(2-Tr(U_p)))",
            "measure": "psi_0**2 times product Haar; psi_0 is the actual all-spin groundstate",
            "n_vertices": n_vertices, "edges": [list(edge) for edge in graph],
            "kappa": str(coupling), "g": str(g),
            "plaquettes": [list(loop) for loop in loops],
            "magnetic_weights": list(map(str, magnetic)),
            "weighted_incidence": list(map(str, incidence)),
            "log_vacuum_gradient_upper": [str(2 * g * d) for d in incidence],
            "singleton_comparison_floor_enclosures": [list(map(str, p)) for p in singleton_floors],
            "comparison_floor_formula": "(3/4)*exp(-8*pi*g*sum_block_incidence)",
            "comparison_floor_is_actual_gap_enclosure": False,
            "blocks": [list(block) for block in selected],
            "block_incidence": list(map(str, block_incidence)),
            "block_comparison_floor_enclosures": [list(map(str, floors[d])) for d in block_incidence],
            "local_bounds_independent_of_exterior_configuration": True,
            "local_volume_dependence": "only block size and weighted incidence; unit electric weights",
            "dense_influence_row_upper": list(map(str, row_upper)),
            "dense_influence_q_upper": str(q),
            "dense_factorization_gate_passed": factorization_pass,
            "finite_graph_gap_lower": str(delta),
            "finite_graph_gap_verified": delta > 0,
            "volume_uniform_global_factorization_verified": False,
            "continuum_claim": False, "yang_mills_claim": False,
        }
        certificate = make_certificate(
            claim="actual SU(2) vacuum conditional Poincare floors and a separate dense finite-graph gate",
            payload={"type": "su2_actual_vacuum_local_v1", "witness": witness},
            honesty={"unconditional_transcendentals": True,
                     "yang_mills_claim": False, "continuum_claim": False},
            meta={"analytic_implication": "docs/api/gauge-vacuum-local.md",
                  "scope": "written all-spin finite-graph analysis; local constants depend on incidence"},
        )
    return {
        "status": "PASS" if local_pass else "INCONCLUSIVE",
        "verification_kind": "CERTIFIED_INTERVAL_WITH_WRITTEN_ANALYTIC_IMPLICATION",
        "local_comparison_floors_positive": local_pass,
        "finite_graph_gap_verified": delta > 0,
        "witness": witness, "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        "theorem_prover_verified": False, "mathlib_verified": False,
        "analytic_implication_formally_verified": False,
        "volume_uniform_neutral_gap_claim": False, "infinite_volume_claim": False,
        "uniform_in_a_claim": False, "continuum_claim": False, "yang_mills_claim": False,
    }


def replay_su2_vacuum_local_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute bounds and scope; a rehashed fabricated witness is refused."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_actual_vacuum_local_v1":
            return False
        w = payload["witness"]
        report = su2_vacuum_local_bounds(
            w["n_vertices"], [tuple(edge) for edge in w["edges"]],
            kappa=Fraction(w["kappa"]), plaquettes=w["plaquettes"],
            magnetic_weights=[Fraction(x) for x in w["magnetic_weights"]], blocks=w["blocks"],
        )
        return bool(report["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False
