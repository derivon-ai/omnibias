# SPDX-License-Identifier: Apache-2.0
"""Exact original-link Langevin generator on finite SU(2) Wilson graphs.

This is a pointwise generator evaluator, not a time discretization or a
sampled ensemble. No continuum dimension is inferred from a finite graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.stochastic.finite import rational
from omnibias.geometry.gauge.transfer.static_sources import _cycles, _graph

Quaternion = tuple[Q, Q, Q, Q]


def _unit(q: Sequence[int | Q]) -> Quaternion:
    if len(q) != 4:
        raise ValueError("SU2 links require four rational quaternion entries")
    values = tuple(rational(x, "quaternion entry") for x in q)
    if sum(v*v for v in values) != 1:
        raise ValueError("SU2 links must have exact unit norm")
    return values[0], values[1], values[2], values[3]


def quaternion_product(a: Sequence[int | Q], b: Sequence[int | Q]) -> Quaternion:
    """Bilinear quaternion multiplication, also valid for tangent insertions."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError("quaternion factors must have length four")
    w, x, y, z = (rational(v, "quaternion entry") for v in a)
    v, i, j, k = (rational(u, "quaternion entry") for u in b)
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


def quaternion_inverse(a: Sequence[int | Q]) -> Quaternion:
    q = _unit(a)
    return q[0], -q[1], -q[2], -q[3]


def _product(factors: Sequence[Quaternion]) -> Quaternion:
    value: Quaternion = (Q(1), Q(0), Q(0), Q(0))
    for factor in factors:
        value = quaternion_product(value, factor)
    return value


def wilson_langevin_generator(
    n_vertices: int, edges: Sequence[tuple[int, int]], faces: Sequence[Sequence[int]],
    links: Sequence[Sequence[int | Q]], *, coupling: int | Q = 1,
    temperature: int | Q = 1, observable_weights: Sequence[int | Q] | None = None,
) -> dict[str, Any]:
    """Evaluate LF for S=b sum_p(2-Tr U_p), F=sum_p w_p(2-Tr U_p).

    L=T sum_(e,a) X_(e,a)^2 - grad(S).grad with generators i*sigma_a/2,
    so the fundamental Casimir is3/4. Finite stationary density is
    proportional to exp(-S/T) against independent normalized Haar links.
    A numerical Euler chain need not preserve it. Faces are simple signed
    one-based edge cycles; all derivatives retain the original edge metric.
    """
    graph = _graph(n_vertices, edges)
    cycles = _cycles(graph, faces)
    if not cycles or len(links) != len(graph):
        raise ValueError("nonempty faces and one SU2 link per edge are required")
    field = tuple(_unit(a) for a in links)
    beta, temp = rational(coupling, "coupling"), rational(temperature, "temperature")
    if beta < 0 or temp <= 0:
        raise ValueError("coupling>=0 and temperature>0 are required")
    weights = tuple(Q(1) for _ in cycles) if observable_weights is None else tuple(rational(w, "observable weight") for w in observable_weights)
    if len(weights) != len(cycles):
        raise ValueError("one observable weight per face is required")
    action = observable = laplacian = Q(0)
    grad_s = [[Q(0) for _ in range(3)] for _ in graph]
    grad_f = [[Q(0) for _ in range(3)] for _ in graph]
    traces: list[Q] = []
    for cycle, weight in zip(cycles, weights, strict=True):
        factors = [field[token-1] if token > 0 else quaternion_inverse(field[-token-1]) for token in cycle]
        trace = 2*_product(factors)[0]
        traces.append(trace)
        action += beta*(2-trace)
        observable += weight*(2-trace)
        laplacian += weight*Q(3*len(cycle), 4)*trace
        for j, token in enumerate(cycle):
            edge = abs(token)-1
            for axis in range(3):
                tangent = tuple(Q(1, 2) if k == axis+1 else Q(0) for k in range(4))
                if token > 0:
                    derivative = quaternion_product(tangent, field[edge])
                else:
                    negative = quaternion_product(factors[j], tangent)
                    derivative = (-negative[0], -negative[1], -negative[2], -negative[3])
                differentiated = [*factors[:j], derivative, *factors[j+1:]]
                d_action = -2*_product(differentiated)[0]
                grad_s[edge][axis] += beta*d_action
                grad_f[edge][axis] += weight*d_action
    gradient_pairing = sum((a*b for row_a, row_b in zip(grad_s, grad_f, strict=True)
                            for a, b in zip(row_a, row_b, strict=True)), Q(0))
    gradient_squared = sum((a*a for row in grad_f for a in row), Q(0))
    payload = {"type": "finite_su2_wilson_langevin_generator_v1", "inputs": {
        "n_vertices": n_vertices, "edges": [list(e) for e in graph], "faces": [list(f) for f in cycles],
        "links": [[str(x) for x in q] for q in field], "coupling": str(beta), "temperature": str(temp),
        "observable_weights": [str(w) for w in weights]},
        "arithmetic": {"action": str(action), "observable": str(observable), "laplacian": str(laplacian),
                       "drift_pairing": str(gradient_pairing), "gradient_squared": str(gradient_squared),
                       "generator": str(temp*laplacian-gradient_pairing), "face_traces": [str(t) for t in traces]},
        "normalization": "L=T sum X_e,a^2-grad S.grad; T_a=i sigma_a/2; S=b sum(2-Tr U_p)",
        "source_scope": "one rational link configuration on the supplied finite graph; not ensemble statistics"}
    return {**payload, "certificate": make_certificate(claim="exact finite original-link Wilson generator evaluation", payload=payload, meta={"transcend_backend": "not_used"}),
        "method": "EXACT_RATIONAL", "theorem_prover_verified": False, "mathlib_verified": False,
        "continuum_claim": False, "invariant_continuum_measure_constructed": False,
        "yang_mills_mass_gap_claim": False}


def replay_lattice_generator_certificate(certificate: dict[str, Any]) -> bool:
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "finite_su2_wilson_langevin_generator_v1":
            return False
        raw = payload["inputs"]
        expected = wilson_langevin_generator(raw["n_vertices"], raw["edges"], raw["faces"],
            [[Q(x) for x in q] for q in raw["links"]], coupling=Q(raw["coupling"]),
            temperature=Q(raw["temperature"]), observable_weights=[Q(w) for w in raw["observable_weights"]])
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = ["quaternion_inverse", "quaternion_product", "replay_lattice_generator_certificate", "wilson_langevin_generator"]
