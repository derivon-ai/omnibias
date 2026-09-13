# SPDX-License-Identifier: Apache-2.0
"""Actual finite Wilson-word vacuum conditionals, uniformly over the exterior.

The local Feynman--Kac comparison and compact heat Harnack proof are in
docs/api/gauge-ambient-conditional.md. Exponential and factored dyadic bounds
remain symbolic: positivity never depends on a floating-point evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest

_TIME_RATIOS = (Q(1, 4), Q(1, 2), Q(4, 7), Q(1), Q(2), Q(4))


def _positive_rational(value: int | Q | str, name: str) -> Q:
    if isinstance(value, bool) or not isinstance(value, (int, Q, str)):
        raise TypeError(f"{name} must be an integer, Fraction or rational string")
    try:
        result = Q(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} must be a strictly positive rational") from exc
    if result <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _sequence(value: Sequence[Any], name: str) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")


def su2_ambient_conditional_gap(
    kappa: int | Q | str,
    *,
    block_link_ids: Sequence[int],
    plaquette_words: Sequence[Sequence[int]],
    n_links: int,
    heat_time: int | Q | str | None = None,
) -> dict[str, Any]:
    """Certify the actual full-scalar conditional quantum gap at every exterior.

    Link IDs are one-based. Each four-letter signed word contributes its own
    Wilson term, including duplicates and words containing repeated links.
    These inputs define a scalar operator; closed-loop graph geometry and a
    local gauge action are not inferred from link IDs alone.

    With b block links and p touching words, the positive lower bound is
    (3*kappa/8)*2**(-3*b)*exp(-32*s*p/kappa**2-1936*b/(49*s)).
    If p=0 the exact Haar gap 3*kappa/8 is returned instead. A supplied s must
    be positive. Otherwise a finite rational grid for s/kappa is searched;
    this is not a global optimization over all positive heat times.
    """
    coupling = _positive_rational(kappa, "kappa")
    requested_time = None if heat_time is None else _positive_rational(heat_time, "heat_time")
    total_links = _integer(n_links, "n_links")
    if total_links < 1:
        raise ValueError("n_links must be positive")
    _sequence(block_link_ids, "block_link_ids")
    block = [_integer(i, "block link ID") for i in block_link_ids]
    if not block or len(set(block)) != len(block):
        raise ValueError("block_link_ids must be nonempty and distinct")
    if any(i < 1 or i > total_links for i in block):
        raise ValueError("block link IDs must lie in 1..n_links")
    block.sort()
    block_set = set(block)
    _sequence(plaquette_words, "plaquette_words")
    words: list[list[int]] = []
    for word in plaquette_words:
        _sequence(word, "plaquette word")
        if len(word) != 4:
            raise ValueError("each plaquette word must contain four signed link IDs")
        converted = [_integer(i, "signed link ID") for i in word]
        if any(i == 0 or abs(i) > total_links for i in converted):
            raise ValueError("signed link IDs must be nonzero with absolute value <= n_links")
        words.append(converted)
    touched = [i for i, word in enumerate(words) if any(abs(e) in block_set for e in word)]
    b, p = len(block), len(touched)

    def exponent(s: Q) -> Q:
        return 32 * s * p / coupling**2 + Q(1936 * b, 49) / s

    candidates = (
        [
            {"heat_time_ratio": str(q), "negative_exponent": str(exponent(q * coupling))}
            for q in _TIME_RATIOS
        ]
        if p
        else []
    )
    if p == 0:
        selected_time, omega = None, Q(0)
        selection = "exact Haar factorization; heat time is unused"
    elif requested_time is not None:
        selected_time = requested_time
        omega = exponent(selected_time)
        selection = "supplied positive rational heat time"
    else:
        q_best = min(_TIME_RATIOS, key=lambda q: exponent(q * coupling))
        selected_time = q_best * coupling
        omega = exponent(selected_time)
        selection = "minimum on the recorded finite rational time-ratio grid"

    haar_gap = 3 * coupling / 8
    dyadic_heat_exponent = 3 * b if p else 0
    quantum_prefactor = haar_gap / 2**dyadic_heat_exponent
    poincare_prefactor = Q(3, 4) / 2**dyadic_heat_exponent
    ceiling_argument = 3 * omega / 2
    extra_exponent = -(-ceiling_argument.numerator // ceiling_argument.denominator)
    total_exponent = dyadic_heat_exponent + extra_exponent
    arithmetic = {
        "kappa": str(coupling),
        "b": b,
        "p": p,
        "selected_heat_time": str(selected_time) if selected_time is not None else None,
        "selected_heat_time_ratio": str(selected_time / coupling)
        if selected_time is not None
        else None,
        "physical_semigroup_time": str(2 * selected_time / coupling)
        if selected_time is not None
        else None,
        "touching_potential_upper": str(8 * p / coupling),
        "heat_time_selection": selection,
        "default_heat_time_candidates": candidates,
        "negative_exponent": str(omega),
        "quantum_gap_prefactor": str(quantum_prefactor),
        "poincare_gap_prefactor": str(poincare_prefactor),
        "haar_quantum_gap": str(haar_gap),
        "exact_haar_quantum_gap": str(haar_gap) if p == 0 else None,
        "density_log_oscillation_upper": {
            "rational_part": str(omega),
            "log_two_coefficient": dyadic_heat_exponent,
        },
        "log_two_rational_lower": "2/3",
        "dyadic_ceiling_argument": str(ceiling_argument),
        "dyadic_extra_exponent": extra_exponent,
        "dyadic_total_exponent": total_exponent,
    }
    payload = {
        "type": "su2_ambient_conditional_gap_v1",
        "status": "PASS",
        "inputs": {
            "kappa": str(coupling),
            "n_links": total_links,
            "block_link_ids": block,
            "plaquette_words": words,
            "heat_time": str(requested_time) if requested_time is not None else None,
        },
        "touched_plaquette_indices": touched,
        "plaquette_index_convention": "zero-based input positions; duplicate words retain multiplicity",
        "normalization": "H=kappa*sum(C_e)/2+2*sum(2-Tr(U_word))/kappa; C_fund=3/4",
        "operator_domain": "full scalar L2(SU2^n_links, product normalized Haar)",
        "conditional_density": "psi(x,y)^2 divided by its Haar integral over x; psi is the actual positive ground state",
        "conditional_operator": "-kappa/2*(Delta_X+2*grad_X(log psi).grad_X)",
        "quantum_gap_lower": {
            "prefactor": str(quantum_prefactor),
            "negative_exponent": str(omega),
            "meaning": "prefactor * exp(-negative_exponent)",
        },
        "poincare_gap_lower": {
            "prefactor": str(poincare_prefactor),
            "negative_exponent": str(omega),
            "meaning": "prefactor * exp(-negative_exponent)",
        },
        "factored_dyadic_quantum_gap_lower": {
            "prefactor": str(haar_gap),
            "base": 2,
            "negative_integer_exponent": total_exponent,
            "meaning": "prefactor * base**(-negative_integer_exponent); power is not materialized",
        },
        "arithmetic": arithmetic,
        "actual_positive_vacuum_identified_in_written_analysis": True,
        "actual_conditional_gap_verified_in_written_analysis": True,
        "ambient_exterior_uniformity_verified_in_written_analysis": True,
        "all_positive_couplings_covered_in_written_analysis": True,
        "all_scalar_block_modes_covered_in_written_analysis": True,
        "exact_haar_conditional_gap_verified_in_written_analysis": p == 0,
        "ambient_bound_depends_only_on_block_and_touching_count": True,
        "graph_cycle_geometry_verified": False,
        "local_gauge_invariance_verified": False,
        "physical_gauge_sector_application_verified": False,
        "gauge_application_condition": "requires independently established closed-loop gauge invariance and the chosen internal-Gauss subspace",
        "frozen_bare_hamiltonian_gap_verified": False,
        "global_hamiltonian_gap_verified": False,
        "bulk_uniform_gap_verified": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "source_gap_used_as_premise": False,
        "spin_truncation_used": False,
        "analytic_proof_formally_verified": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }
    certificate = make_certificate(
        claim="actual Wilson-word vacuum conditional gap, uniform over the exterior with explicit local counts",
        payload=payload,
        meta={
            "analytic_implication": "docs/api/gauge-ambient-conditional.md",
            "proof_register": "written local Feynman-Kac/Harnack/variance comparison; exact rational canonical replay",
            "transcend_backend": "symbolic exponential; no floating-point positivity test",
        },
    )
    return {**payload, "certificate": certificate}


def replay_su2_ambient_conditional_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild the operator, local counts, time selection, bounds and scope."""
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_ambient_conditional_gap_v1":
            return False
        inputs = payload["inputs"]
        expected = su2_ambient_conditional_gap(
            inputs["kappa"],
            block_link_ids=inputs["block_link_ids"],
            plaquette_words=inputs["plaquette_words"],
            n_links=inputs["n_links"],
            heat_time=inputs["heat_time"],
        )
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = [
    "replay_su2_ambient_conditional_certificate",
    "su2_ambient_conditional_gap",
]
