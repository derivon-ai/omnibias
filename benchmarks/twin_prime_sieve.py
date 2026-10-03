# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite Möbius-bilinear diagnostics for the twin-prime research ledger.

The exact gates cover local-factor algebra and certificate honesty.  The
finite-scale bilinear masses are explicitly exploratory: float logarithms and
four dyadic scales do not establish uniform cancellation.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.proof import generate_obligation  # noqa: E402
from omnibias.core.proof.certificate import verify_certificate_digest  # noqa: E402
from omnibias.holonomic.twin_prime import (  # noqa: E402
    asymptotic_sieve_honesty,
    canonical_fi_term_ledger,
    certify_asymptotic_sieve_local_product,
    certify_fi_combinatorial_replay,
    certify_fi_terminal_atlas,
    certify_fixed_shift_determinant_replay,
    certify_fixed_shift_kernel_cell,
    certify_rho_insertion,
    fi_atlas_obligation_certificates,
    fi_combinatorial_obligation_certificates,
    fi_rational_terminal_family,
    local_factor_obligation_certificates,
    seal_asymptotic_sieve_local_product,
    seal_fi_combinatorial_replay,
    seal_fi_terminal_atlas,
    seal_fixed_shift_determinant_replay,
    select_fi_dyadic_s,
    validate_fi_term_ledger,
    verify_dyadic_membership,
    wright_fixed_shift_range,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _arithmetic_tables(limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Möbius, total-prime-factor count, and von Mangoldt tables."""

    least = np.zeros(limit + 1, dtype=np.int32)
    mobius = np.zeros(limit + 1, dtype=np.int8)
    omega = np.zeros(limit + 1, dtype=np.int16)
    mobius[1] = 1
    primes: list[int] = []
    for value in range(2, limit + 1):
        if least[value] == 0:
            least[value] = value
            mobius[value] = -1
            omega[value] = 1
            primes.append(value)
        for prime in primes:
            product = prime * value
            if product > limit:
                break
            least[product] = prime
            omega[product] = omega[value] + 1
            if value % prime == 0:
                mobius[product] = 0
                break
            mobius[product] = -mobius[value]
    mangoldt = np.zeros(limit + 1, dtype=np.float64)
    for prime in primes:
        prime_log = math.log(prime)
        power = prime
        while power <= limit:
            mangoldt[power] = prime_log
            if power > limit // prime:
                break
            power *= prime
    return mobius, omega, mangoldt


def _gamma_table(
    mobius: np.ndarray,
    *,
    max_s: int,
    cutoffs: tuple[int, ...],
) -> dict[int, np.ndarray]:
    tables: dict[int, np.ndarray] = {}
    for cutoff in cutoffs:
        gamma = np.zeros(max_s + 1, dtype=np.int16)
        for divisor in range(1, cutoff + 1):
            coefficient = int(mobius[divisor])
            if coefficient:
                gamma[divisor : max_s + 1 : divisor] += coefficient
        tables[cutoff] = gamma
    return tables


def _bilinear_mass(
    *,
    x_value: int,
    lower_s: int,
    cutoff: int,
    gamma: np.ndarray,
    mobius: np.ndarray,
    sequence: np.ndarray,
) -> float:
    upper_s = min(2 * lower_s, x_value)
    max_r = x_value // (lower_s + 1)
    inner = np.zeros(max_r + 1, dtype=np.float64)
    for s_value in range(lower_s + 1, upper_s + 1):
        coefficient = int(gamma[s_value])
        if coefficient == 0:
            continue
        r_values = np.arange(1, x_value // s_value + 1, dtype=np.int64)
        products = r_values * s_value
        inner[r_values] += (
            coefficient
            * mobius[products].astype(np.float64)
            * sequence[products]
        )
    del cutoff
    return float(np.sum(np.abs(inner), dtype=np.float64))


def _dyadic_cutoffs(maximum: int) -> tuple[int, ...]:
    values = [1]
    while values[-1] < maximum:
        values.append(min(2 * values[-1], maximum))
    return tuple(dict.fromkeys(values))


def _dyadic_bands(x_value: int, nu: Fraction) -> tuple[int, ...]:
    lower = max(2, math.ceil(x_value ** (float(nu) / 2)))
    upper = max(lower, math.floor(math.sqrt(x_value)))
    first = 1 << max(1, (lower - 1).bit_length())
    values: list[int] = []
    current = first
    while current <= upper:
        values.append(current)
        current *= 2
    if not values:
        values.append(lower)
    return tuple(values)


def _scale_diagnostics(x_value: int, nu: Fraction) -> dict[str, Any]:
    mobius, omega, mangoldt = _arithmetic_tables(x_value)
    shifted = np.zeros(x_value + 1, dtype=np.float64)
    shifted[3:] = (mobius[3:] != 0) * mangoldt[1 : x_value - 1]
    liouville = np.where(omega % 2 == 0, 1.0, -1.0)
    parity_model = 0.5 * (1.0 + liouville)
    max_c = max(1, math.floor(x_value ** (1 - float(nu))))
    cutoffs = _dyadic_cutoffs(max_c)
    bands = _dyadic_bands(x_value, nu)
    gammas = _gamma_table(
        mobius,
        max_s=min(2 * max(bands), x_value),
        cutoffs=cutoffs,
    )
    shifted_mass = float(np.sum(shifted, dtype=np.float64))
    parity_mass = float(np.sum(parity_model, dtype=np.float64))
    rows: list[dict[str, Any]] = []
    for band in bands:
        for cutoff in cutoffs:
            shifted_b = _bilinear_mass(
                x_value=x_value,
                lower_s=band,
                cutoff=cutoff,
                gamma=gammas[cutoff],
                mobius=mobius,
                sequence=shifted,
            )
            parity_b = _bilinear_mass(
                x_value=x_value,
                lower_s=band,
                cutoff=cutoff,
                gamma=gammas[cutoff],
                mobius=mobius,
                sequence=parity_model,
            )
            rows.append(
                {
                    "N": band,
                    "C": cutoff,
                    "shifted_prime_normalized_mass": shifted_b / shifted_mass,
                    "parity_model_normalized_mass": parity_b / parity_mass,
                }
            )
    return {
        "x": x_value,
        "shifted_sequence_mass": shifted_mass,
        "parity_model_mass": parity_mass,
        "rows": rows,
    }


def _top_band_c4(scale: dict[str, Any]) -> dict[str, Any]:
    rows = scale["rows"]
    max_band = max(int(row["N"]) for row in rows)
    return next(
        row
        for row in rows
        if int(row["N"]) == max_band and int(row["C"]) == 4
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    scales = (2**14, 2**16, 2**18, 2**20, 2**22) if full else (2**14,)
    nu = Fraction(3, 4)
    started = time.perf_counter()

    local = certify_asymptotic_sieve_local_product(97)
    sealed_local = seal_asymptotic_sieve_local_product(local)
    local_obligations = local_factor_obligation_certificates(local)
    g1 = {
        "name": "g1_exact_local_factors",
        "passed": (
            local.proved
            and local.combined_product == local.twin_product
            and verify_certificate_digest(sealed_local)
            and all(verify_certificate_digest(item) for item in local_obligations)
            and all(generate_obligation(item) is not None for item in local_obligations)
        ),
        "prime_limit": local.prime_limit,
        "kernel_obligations": len(local_obligations),
        "detail": "finite Euler-product prefixes agree exactly over the rationals",
    }
    replay = certify_fi_combinatorial_replay(n_max=128, y=4, z=4, s=8)
    sealed_replay = seal_fi_combinatorial_replay(replay)
    replay_obligations = fi_combinatorial_obligation_certificates(replay)
    g1a = {
        "name": "g1a_exact_fi_combinatorial_replay",
        "passed": (
            replay.status == "PROVED"
            and replay.identity_proved
            and replay.terminal_partition_proved
            and replay.dyadic_partition_proved
            and verify_certificate_digest(sealed_replay)
            and all(verify_certificate_digest(item) for item in replay_obligations)
            and all(generate_obligation(item) is not None for item in replay_obligations)
        ),
        "n_max": replay.n_max,
        "coefficient_cases": len(replay.cases),
        "kernel_obligations": len(replay_obligations),
        "detail": (
            "FI equation (3.1) holds coefficientwise for every arithmetic "
            "function on each declared finite divisor lattice"
        ),
    }
    dyadic_selection = select_fi_dyadic_s(Fraction(25))
    rho_insertion = certify_rho_insertion(
        n_max=128,
        delta=2,
        z=4,
        lambdas={1: 1, 2: -1},
    )
    term_ledger = validate_fi_term_ledger(canonical_fi_term_ledger())
    pointwise_dyadic = all(
        verify_dyadic_membership(Fraction(value, 2), y=4, s=8)
        for value in range(0, 73)
    )
    g1a2 = {
        "name": "g1a2_exact_fi_structural_subchecks",
        "passed": (
            dyadic_selection.status == "PROVED"
            and dyadic_selection.s == 8
            and rho_insertion.status == "PROVED"
            and term_ledger.status == "PROVED"
            and pointwise_dyadic
        ),
        "term_leaves": len(term_ledger.uses),
        "detail": (
            "exact dyadic selection/membership, finite rho insertion, and "
            "single-owner non-circular R-prime/B-prime routing"
        ),
    }
    determinant = certify_fixed_shift_determinant_replay(
        r_max=16,
        s_max=16,
        cutoff=4,
    )
    sealed_determinant = seal_fixed_shift_determinant_replay(determinant)
    kernel_cell = certify_fixed_shift_kernel_cell()
    wright_range = wright_fixed_shift_range(
        short_exponent=Fraction(1, 7),
        modulus_exponent=Fraction(1, 2),
        epsilon=Fraction(1, 1000),
    )
    g1a3 = {
        "name": "g1a3_exact_fixed_shift_determinant_route",
        "passed": (
            determinant.status == "PROVED"
            and determinant.shift == 2
            and determinant.no_shift_average
            and determinant.no_termwise_absolute_value
            and verify_certificate_digest(sealed_determinant)
            and kernel_cell.status == "PROVED"
            and kernel_cell.kernel_saving_floor == Fraction(41, 640)
            and kernel_cell.completion_loss_budget == Fraction(1009, 16000)
            and wright_range.fixed_small_shift_range
        ),
        "coefficient_cases": len(determinant.cases),
        "kernel_saving_floor": [
            str(kernel_cell.kernel_saving_floor.numerator),
            str(kernel_cell.kernel_saving_floor.denominator),
        ],
        "completion_loss_budget": [
            str(kernel_cell.completion_loss_budget.numerator),
            str(kernel_cell.completion_loss_budget.denominator),
        ],
        "detail": (
            "exact signed r*d*t-q*k=2 replay and rational theorem-range/cell "
            "bookkeeping; the fixed-shift completion estimate remains absent"
        ),
    }
    atlas = certify_fi_terminal_atlas()
    terminal_family = fi_rational_terminal_family(
        (Fraction(1, 64), Fraction(1, 128), Fraction(1, 256))
    )
    sealed_atlas = seal_fi_terminal_atlas(atlas)
    atlas_obligations = fi_atlas_obligation_certificates(atlas)
    g1b = {
        "name": "g1b_exact_fi_terminal_atlas",
        "passed": (
            atlas.partition_proved
            and atlas.unresolved_fraction == Fraction(1, 38)
            and atlas.far_log_saving == 3
            and tuple(report.unresolved_fraction for report in terminal_family)
            == (Fraction(1, 38), Fraction(1, 76), Fraction(1, 152))
            and verify_certificate_digest(sealed_atlas)
            and all(verify_certificate_digest(item) for item in atlas_obligations)
            and all(generate_obligation(item) is not None for item in atlas_obligations)
        ),
        "unresolved_fraction": [
            str(atlas.unresolved_fraction.numerator),
            str(atlas.unresolved_fraction.denominator),
        ],
        "terminal_family": [
            {
                "epsilon": [
                    str((1 - report.parameters.terminal_cutoff).numerator),
                    str((1 - report.parameters.terminal_cutoff).denominator),
                ],
                "unresolved_fraction": [
                    str(report.unresolved_fraction.numerator),
                    str(report.unresolved_fraction.denominator),
                ],
            }
            for report in terminal_family
        ],
        "kernel_obligations": len(atlas_obligations),
        "detail": (
            "exact log-polyhedral geometry; the far-region analytic bound "
            "remains conditional on named established theorems"
        ),
    }
    diagnostics = [_scale_diagnostics(x_value, nu) for x_value in scales]
    rows = [row for scale in diagnostics for row in scale["rows"]]
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in ("shifted_prime_normalized_mass", "parity_model_normalized_mass")
    )
    g2 = {
        "name": "g2_finite_bilinear_diagnostic",
        "passed": bool(rows) and finite,
        "n_rows": len(rows),
        "detail": "finite-scale values generated; this is not a cancellation bound",
    }
    distinct = any(
        row["shifted_prime_normalized_mass"]
        != row["parity_model_normalized_mass"]
        for row in rows
    )
    g3 = {
        "name": "g3_parity_model_negative_control",
        "passed": distinct,
        "detail": "the shifted sequence and parity model remain separately reported",
    }
    honesty = asymptotic_sieve_honesty(
        finite_reduction=atlas.partition_proved,
        finite_local_factors=local.proved,
    )
    honesty.update(
        {
            "finite_fi_vaughan_identity_check": replay.identity_proved,
            "finite_fi_terminal_partition_check": replay.terminal_partition_proved,
            "finite_fi_dyadic_partition_check": replay.dyadic_partition_proved,
            "friedlander_iwaniec_combinatorial_replay": (
                replay.identity_proved
                and replay.terminal_partition_proved
                and replay.dyadic_partition_proved
            ),
            "finite_fi_dyadic_selection_check": dyadic_selection.unique,
            "finite_fi_rho_insertion_check": rho_insertion.status == "PROVED",
            "finite_fi_term_ledger_check": term_ledger.status == "PROVED",
            "finite_fixed_shift_determinant_replay": (
                determinant.status == "PROVED"
            ),
            "finite_fixed_shift_kernel_cell_check": kernel_cell.status == "PROVED",
            "fixed_shift_2_completion_lemma_proved": False,
            "friedlander_iwaniec_structural_replay": False,
        }
    )
    g4 = {
        "name": "g4_parent_honesty",
        "passed": (
            honesty["twin_prime_conjecture_proof_claim"] is False
            and honesty["friedlander_iwaniec_structural_replay"] is False
            and honesty["fixed_shift_2_completion_lemma_proved"] is False
            and honesty["mobius_bilinear_estimate_proved"] is False
            and honesty["uniform_asymptotic_passage_proved"] is False
        ),
        "detail": "finite diagnostics do not discharge any asymptotic premise",
    }
    entries = [g1, g1a, g1a2, g1a3, g1b, g2, g3, g4]
    if not all(bool(entry["passed"]) for entry in entries):
        raise AssertionError(f"twin-prime sieve smoke failed: {entries}")
    if full:
        heldout = _top_band_c4(diagnostics[-1])
        shifted_value = float(heldout["shifted_prime_normalized_mass"])
        parity_value = float(heldout["parity_model_normalized_mass"])
        entries.append(
            {
                "name": "g5_unseen_scale_separation",
                "passed": shifted_value < 0.025 and parity_value > 0.08,
                "x": diagnostics[-1]["x"],
                "N": heldout["N"],
                "C": heldout["C"],
                "shifted_prime_normalized_mass": shifted_value,
                "shifted_max": 0.025,
                "parity_model_normalized_mass": parity_value,
                "parity_min": 0.08,
                "detail": (
                    "finite held-out diagnostic only; thresholds were frozen "
                    "after scales through 2^20"
                ),
            }
        )
    payload = {
        **provenance(
            schema="omnibias.benchmarks.twin_prime_sieve.v5",
            config={
                "family": "twin_prime_sieve",
                "full": full,
                "scales": list(scales),
                "nu": [str(nu.numerator), str(nu.denominator)],
                "research_status": "BLOCKED",
                "honesty": honesty,
            },
        ),
        "gates": dict(gates_block(entries)),
        "finite_certificates": {
            "local_factor_prefix": sealed_local,
            "local_factor_kernel_obligations": list(local_obligations),
            "fi_combinatorial_replay": sealed_replay,
            "fi_combinatorial_kernel_obligations": list(replay_obligations),
            "fi_terminal_atlas": sealed_atlas,
            "fi_atlas_kernel_obligations": list(atlas_obligations),
            "fi_structural_subchecks": {
                "dyadic_selection": dyadic_selection.to_payload(),
                "rho_insertion": rho_insertion.to_payload(),
                "term_ledger": term_ledger.to_payload(),
                "pointwise_dyadic_membership": pointwise_dyadic,
            },
            "fixed_shift_determinant_route": {
                "replay": sealed_determinant,
                "kernel_cell": kernel_cell.to_payload(),
                "wright_range": wright_range.to_payload(),
            },
        },
        "diagnostics": diagnostics,
        "wall_seconds": time.perf_counter() - started,
    }
    artifact = "twin_prime_sieve.json" if full else "twin_prime_sieve_smoke.json"
    if full:
        destination = SCRATCH / "training" / "twin_prime_sieve"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / artifact
        path.write_text(
            __import__("json").dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
