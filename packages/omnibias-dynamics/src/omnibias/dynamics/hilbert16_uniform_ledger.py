# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Phase 1-10 and adversarial obligations for uniform Hilbert-16 finiteness."""

from __future__ import annotations

from omnibias.dynamics.hilbert16_ledger import H16Obligation

PHASE_NAMES: tuple[str, ...] = tuple(f"P{index}" for index in range(1, 11))
ADVERSARIAL_NAMES: tuple[str, ...] = tuple(f"ADV{index}" for index in range(1, 16))

UNIFORM_PARENT_FLAG = "h16_uniform_finiteness_proved"

UNIFORMITY_FIELD_NAMES: tuple[str, ...] = (
    "uniformity_no_hidden_majorant",
    "uniformity_no_oracle_premise",
    "uniformity_finite_parameter_box",
    "uniformity_adversarial_audit_required",
)


def uniformity_field_entries() -> tuple[H16Obligation, ...]:
    statements = {
        "uniformity_no_hidden_majorant": (
            "Every majorant constant is either derived by certificate or named in external_premises."
        ),
        "uniformity_no_oracle_premise": (
            "No granted primitive may assume the open theorem it is meant to prove."
        ),
        "uniformity_finite_parameter_box": (
            "Uniform bounds are stated only over an explicit compact parameter box."
        ),
        "uniformity_adversarial_audit_required": (
            "Track-C candidates must pass the executable Section-IV adversarial suite."
        ),
    }
    return tuple(
        H16Obligation(
            name,
            statements[name],
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.hilbert16_adversarial",
        )
        for name in UNIFORMITY_FIELD_NAMES
    )


def uniform_phase_entries() -> tuple[H16Obligation, ...]:
    source = "packages/omnibias-dynamics/HILBERT16-PROGRAM.md"
    return (
        H16Obligation(
            "P1",
            "Formalize V_n, equivalences, and compact normalized parameter space K_n.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.normalized_family",
        ),
        H16Obligation(
            "P2",
            "Compactify the phase plane and verify cycle correspondence at infinity.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.compactify",
        ),
        H16Obligation(
            "P3",
            "Finite effective stratification of K_n by dynamical data.",
            "BLOCKED",
            ("global separatrix connections are not semialgebraic",),
            source=source,
        ),
        H16Obligation(
            "P4",
            "Finite classification or isolating-neighborhood cover of limit-periodic sets.",
            "BLOCKED",
            ("no exhaustive limit-periodic classification ships",),
            source=source,
        ),
        H16Obligation(
            "P5",
            "Uniform desingularization tree with complexity bounded by D(n).",
            "BLOCKED",
            ("degenerate strata blow-up count not uniformly bounded here",),
            source=source,
        ),
        H16Obligation(
            "P6",
            "Finite library of parameterized displacement maps with bounded R(n).",
            "BLOCKED",
            ("physical return-map membership not proved for arbitrary graphics",),
            source=source,
        ),
        H16Obligation(
            "P7",
            "Uniform zero theorem: at most Z(n) isolated zeros for every displacement map.",
            "BLOCKED",
            ("G1 coalescing passage and degenerate strata remain open",),
            source=source,
        ),
        H16Obligation(
            "P8",
            "Uniform local cyclicity for every limit-periodic model in L_n.",
            "BLOCKED",
            ("depends on P7 and graphic capture",),
            source=source,
        ),
        H16Obligation(
            "P9",
            "Global assembly: finite cover and B(n) summing local bounds.",
            "BLOCKED",
            ("requires P7-P8",),
            source=source,
        ),
        H16Obligation(
            "P10",
            "Configuration classification or honest partial enumeration.",
            "BLOCKED",
            ("Part A octic target not certified; Part B configuration open",),
            source=source,
        ),
    )


def adversarial_entries() -> tuple[H16Obligation, ...]:
    sequences = (
        "hyperbolicity multiplier tends to 1",
        "transversal becomes tangent to the flow",
        "cycle shrinks to a multiple equilibrium",
        "cycles approach infinity",
        "period annulus breaks into isolated cycles",
        "simultaneous equilibrium and separatrix degeneration",
        "return map becomes flat to increasing order",
        "analytic domains shrink to zero",
        "blow-up count appears to increase",
        "normal-form denominator tends to zero",
        "distinct local models collide",
        "displacement map becomes identically zero",
        "roots enter through a domain boundary",
        "finite cover ceases to be uniform",
        "complexity parameter becomes unbounded",
    )
    return tuple(
        H16Obligation(
            name,
            f"Adversarial degeneration sequence: {text}",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.hilbert16_adversarial",
        )
        for name, text in zip(ADVERSARIAL_NAMES, sequences, strict=True)
    )


def uniform_finiteness_discharged(ledger_entries: tuple[H16Obligation, ...]) -> bool:
    """``True`` only if every phase obligation fully discharges with empty premises."""
    phases = tuple(entry for entry in ledger_entries if entry.name in PHASE_NAMES)
    return bool(phases) and all(entry.fully_discharged for entry in phases)
