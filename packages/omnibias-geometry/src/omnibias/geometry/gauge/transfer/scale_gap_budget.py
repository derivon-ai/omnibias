# SPDX-License-Identifier: Apache-2.0
"""Exact finite Schur-loss budgets with explicit lattice-spacing normalization.

Inputs are comparison numbers, not verified operators or RG steps. A finite
geometric prefix does not establish a bound for every subsequent scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _rational

SCOPE = {
    "operator_block_premises_verified": False,
    "vacuum_alignment_verified": False,
    "complete_complement_verified": False,
    "common_physical_units_verified": False,
    "actual_rg_steps_verified": False,
    "all_scale_envelope_verified": False,
    "physical_gap_verified": False,
    "uniform_in_volume_claim": False,
    "infinite_volume_claim": False,
    "uniform_in_a_claim": False,
    "all_scale_refinement_claim": False,
    "continuum_claim": False,
    "yang_mills_claim": False,
    "yang_mills_mass_gap_claim": False,
    "mathlib_verified": False,
}


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an explicit finite sequence")
    return value


def scale_gap_budget(
    initial_spacing: int | Q,
    initial_lattice_gap: int | Q,
    steps: Sequence[Sequence[int | Q]],
    *,
    geometric_envelope: Sequence[int | Q] | None = None,
) -> dict[str, Any]:
    """Check supplied finite steps (spacing ratio, epsilon, beta, high buffer).

    A step compares A_next / spacing_ratio to the preceding centered
    operator: low >= (1-epsilon)*d, all-complement high >= (1+h)*d,
    and cross norm <= beta*d. It gives relative loss epsilon+beta^2/h.
    The optional (c, theta) envelope is checked only on the supplied prefix.
    Invalid exact inputs raise; deficient positive budgets are INCONCLUSIVE.
    """
    a0 = _rational(initial_spacing, "initial_spacing")
    d0 = _rational(initial_lattice_gap, "initial_lattice_gap")
    if a0 <= 0 or d0 <= 0:
        raise ValueError("initial spacing and lattice gap must be positive")
    parsed: list[tuple[Q, Q, Q, Q]] = []
    for raw in _sequence(steps, "steps"):
        row = _sequence(raw, "step")
        if len(row) != 4:
            raise ValueError("each step must contain spacing ratio, epsilon, beta, high buffer")
        s, epsilon, beta, h = (_rational(v, "step entry") for v in row)
        if s <= 0 or h <= 0 or epsilon < 0 or beta < 0:
            raise ValueError("spacing ratio and high buffer must be positive; epsilon and beta nonnegative")
        parsed.append((s, epsilon, beta, h))
    envelope: tuple[Q, Q] | None = None
    if geometric_envelope is not None:
        raw_envelope = _sequence(geometric_envelope, "geometric_envelope")
        if len(raw_envelope) != 2:
            raise ValueError("geometric_envelope must be (c, theta)")
        c, theta = (_rational(v, "envelope entry") for v in raw_envelope)
        if c < 0 or not 0 <= theta < 1:
            raise ValueError("c must be nonnegative and theta in [0,1)")
        envelope = c, theta
    losses = [epsilon + beta * beta / h for _, epsilon, beta, h in parsed]
    spacings, gaps, products, sums = [a0], [d0], [Q(1)], [Q(0)]
    rows: list[dict[str, Any]] = []
    for index, ((s, _epsilon, beta, h), delta) in enumerate(zip(parsed, losses, strict=True)):
        spacings.append(spacings[-1] * s)
        products.append(products[-1] * (1 - delta))
        gaps.append(gaps[-1] * s * (1 - delta))
        sums.append(sums[-1] + delta)
        # The residual 2x2 comparison at z=1-delta is positive semidefinite.
        low, high = beta * beta / h, h + delta
        determinant = low * high - beta * beta
        rows.append({
            "index": index, "relative_loss": str(delta), "retained_fraction": str(1 - delta),
            "strict_positive_step": delta < 1,
            "schur_residual_matrix": [[str(low), str(-beta)], [str(-beta), str(high)]],
            "schur_residual_determinant": str(determinant),
            "schur_matrix_psd_verified": low >= 0 and high >= 0 and determinant >= 0,
        })
    chain_passed = all(delta < 1 for delta in losses)
    scale_identity = all(a0 * gaps[i] == spacings[i] * d0 * products[i] for i in range(len(gaps)))
    product_bound = all(products[i] >= 1 - sums[i] for i in range(len(gaps))) if chain_passed else False
    envelope_prefix: bool | None = None
    envelope_sum: Q | None = None
    envelope_floor: Q | None = None
    if envelope is not None:
        c, theta = envelope
        envelope_prefix = all(delta <= c * theta**i for i, delta in enumerate(losses))
        envelope_sum = c / (1 - theta)
        if envelope_prefix and envelope_sum < 1 and chain_passed:
            envelope_floor = (d0 / a0) * (1 - envelope_sum)
    envelope_passed = envelope is None or (envelope_prefix is True and envelope_sum is not None and envelope_sum < 1)
    psd_passed = all(row["schur_matrix_psd_verified"] for row in rows)
    passed = chain_passed and envelope_passed and scale_identity and product_bound and psd_passed
    failures = []
    if not psd_passed:
        failures.append("schur_residual_matrix_not_psd")
    if not chain_passed:
        failures.append("nonpositive_retained_fraction")
    if envelope_prefix is False:
        failures.append("supplied_prefix_exceeds_geometric_envelope")
    if envelope_sum is not None and envelope_sum >= 1:
        failures.append("geometric_total_loss_not_below_one")
    witness = {
        "inputs": {
            "initial_spacing": str(a0), "initial_lattice_gap": str(d0),
            "steps": [[str(v) for v in row] for row in parsed],
            "geometric_envelope": [str(v) for v in envelope] if envelope is not None else None,
        },
        "step_order": ["spacing_ratio", "compression_loss", "relative_cross_norm", "high_buffer"],
        "steps": rows, "finite_step_count": len(parsed),
        "spacing_sequence": [str(v) for v in spacings],
        "computed_lattice_gap_sequence": [str(v) for v in gaps],
        "retained_product_sequence": [str(v) for v in products],
        "cumulative_relative_loss_sequence": [str(v) for v in sums],
        "exact_scale_cancellation_verified": scale_identity,
        "finite_loss_product_inequality_verified": product_bound,
        "finite_chain_budget_verified": chain_passed,
        "conditional_finite_lattice_gap_floor": str(gaps[-1]) if chain_passed else None,
        "conditional_finite_physical_gap_floor": str(gaps[-1] / spacings[-1]) if chain_passed else None,
        "bernoulli_physical_comparison_value": str((d0 / a0) * (1 - sums[-1])),
        "geometric_prefix_verified": envelope_prefix,
        "geometric_all_terms_sum": str(envelope_sum) if envelope_sum is not None else None,
        "conditional_all_stage_physical_floor": str(envelope_floor) if envelope_floor is not None else None,
        "all_stage_floor_premise": "the same operator hypotheses and geometric envelope hold at EVERY future scale",
        "failure_reasons": failures,
    }
    honesty = {"finite_gate_verified": passed, "finite_arithmetic_only": True, **SCOPE}
    certificate = make_certificate(
        claim="finite exact scale-normalized Schur and relative-loss arithmetic, conditional on unverified operator inputs",
        payload={"type": "scale_gap_budget_v1", "witness": witness},
        honesty=honesty,
        meta={"analytic_implication": "docs/api/gauge-scale-gap-budget.md", "transcend_backend": "not_used"},
    )
    return {"status": "PASS" if passed else "INCONCLUSIVE", "witness": witness,
            "certificate": certificate, "theorem_prover_verified": False, **honesty}


def replay_scale_gap_budget_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute supplied arithmetic and refuse rehashed premise promotion."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "scale_gap_budget_v1":
            return False
        inputs = payload["witness"]["inputs"]
        envelope = inputs["geometric_envelope"]
        result = scale_gap_budget(
            Q(inputs["initial_spacing"]), Q(inputs["initial_lattice_gap"]),
            [[Q(v) for v in row] for row in inputs["steps"]],
            geometric_envelope=[Q(v) for v in envelope] if envelope is not None else None,
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_scale_gap_budget_certificate",
    "scale_gap_budget",
]
