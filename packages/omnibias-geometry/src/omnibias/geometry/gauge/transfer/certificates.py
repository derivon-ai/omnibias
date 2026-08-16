# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sealed certificates for a certified transfer-matrix spectral gap.

Mirrors the Birkhoff-Hopf certificate in :mod:`omnibias.pinn.certified.machine`,
**sealed from the start**: the rational ``subdominant_ratio_upper`` these payloads
carry is exactly the finite obligation
:func:`omnibias.core.proof.lean_check.generate_obligation` turns into the kernel's
``spectral_gap_pos`` lemma, and ``check_certificate`` refuses an unsealed payload
before emitting any Lean.  A certificate with an obligation but no digest is a
certificate that can never reach the kernel.

Honesty
-------
``continuum_claim`` is hard-wired ``False`` and the honesty block records that the
statement is about a fixed matrix at fixed spacing in finite dimension.  A
conjecture asserting a continuum or Yang-Mills claim is downgraded by the
machine's honesty gate, because nothing here supports one.

What the two registers each prove is worth stating plainly, since it is easy to
conflate.  The **rigorous** register -- outward-rounded interval arithmetic in
:mod:`omnibias.core.verified` -- is what establishes that
``subdominant_ratio_upper`` really bounds ``|lambda_1| / lambda_0``.  The **formal**
register only checks the finite integer inequality ``rn < rd => 0 < rd - rn``; the
kernel's own ``spectral_gap_pos`` docstring says so explicitly.  So
``theorem_prover_verified`` means "the arithmetic step was kernel-checked", not
"Lean proved the spectral gap".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omnibias.core.proof import Certificate, seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.gap import TransferGapResult
from omnibias.geometry.gauge.transfer.matrices import TransferMatrix, rebuild
from omnibias.geometry.gauge.transfer.strong_coupling import (
    BACKTRACK_POLYMER_METHOD,
    CRUDE_POLYMER_METHOD,
    POLYMER_METHOD,
    StrongCouplingGapResult,
    certified_strong_coupling_glueball_bound,
)

#: Schema version of the sealed transfer-matrix gap certificate.
TRANSFER_GAP_SCHEMA_VERSION = "verified-transfer-matrix-gap-1"

#: The :class:`~omnibias.core.proof.Conjecture` kind this certificate answers.
TRANSFER_GAP_KIND = "transfer_matrix_spectral_gap"

_HONESTY_NOTE = (
    "fixed-matrix (fixed lattice spacing / finite truncation) certified spectral "
    "gap of an explicitly constructed transfer matrix; NOT a continuum-limit, "
    "infinite-volume, or Yang-Mills mass-gap claim"
)

#: Schema version of the sealed crude polymer-bound certificate.
STRONG_COUPLING_SCHEMA_VERSION = "verified-strong-coupling-gap-1"

#: The :class:`~omnibias.core.proof.Conjecture` kind this certificate answers.
STRONG_COUPLING_KIND = "strong_coupling_glueball_gap"

_STRONG_COUPLING_NOTE = (
    "polymer-count lower bound for SU(2) Wilson at one fixed beta and "
    "spacing; method is two_scale_polymer_count, backtrack_polymer_count, "
    "or crude_polymer_count, NOT Osterwalder-Seiler, NOT a continuum-limit, "
    "infinite-volume, or Yang-Mills mass-gap claim"
)


def seal_transfer_gap_certificate(
    result: TransferGapResult,
    transfer: TransferMatrix,
    *,
    claim: str = "",
) -> Certificate:
    """Seal a :class:`~.gap.TransferGapResult` into a tamper-evident certificate.

    The ``parameters`` block records the constructor and its inputs rather than the
    matrix itself, which is what lets :func:`replay_transfer_matrix_gap` rebuild
    the matrix from scratch and compare against the sealed numbers.
    """
    return seal_certificate(
        {
            "schema_version": TRANSFER_GAP_SCHEMA_VERSION,
            "claim": claim or f"{result.model} spectral gap",
            "observable": "transfer_matrix_spectral_gap",
            "model": result.model,
            "basis": result.basis,
            "method": result.method,
            "dimension": int(result.dimension),
            "subdominant_ratio_upper": float(result.subdominant_ratio_upper),
            "spectral_gap_lower": float(result.spectral_gap_lower),
            "spectral_gap_lower_per_unit": float(result.spectral_gap_lower_per_unit),
            "lattice_spacing": float(result.lattice_spacing),
            "partners_deflated": int(result.partners_deflated),
            "parameters": dict(transfer.parameters),
            "trial_gram_condition": result.trial_gram_condition,
            "trial_flagged": bool(result.trial_flagged),
            "trial_remainder_width": float(result.trial_remainder_width),
            "continuum_claim": False,
            "honesty": {
                "unproven_claim": False,
                "continuum_claim": False,
                "fixed_matrix": True,
                "finite_truncation": True,
                "interval_verified": True,
                "yang_mills_claim": False,
                "note": _HONESTY_NOTE,
            },
        }
    )


def transfer_gap_schema_errors(cert: Certificate) -> list[str]:
    """Validate a ``verified-transfer-matrix-gap-1`` certificate (empty list == valid)."""
    errors: list[str] = []
    required = (
        "schema_version",
        "observable",
        "model",
        "basis",
        "method",
        "dimension",
        "subdominant_ratio_upper",
        "spectral_gap_lower",
        "spectral_gap_lower_per_unit",
        "lattice_spacing",
        "parameters",
        "trial_gram_condition",
        "trial_flagged",
        "trial_remainder_width",
        "continuum_claim",
        "honesty",
        "digest",
    )
    for key in required:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    if cert.get("schema_version") != TRANSFER_GAP_SCHEMA_VERSION:
        errors.append(f"schema_version must be {TRANSFER_GAP_SCHEMA_VERSION!r}")
    if cert.get("continuum_claim", True):
        errors.append("continuum_claim must be False")
    honesty = cert.get("honesty", {})
    if not isinstance(honesty, Mapping):
        errors.append("honesty must be a mapping")
        honesty = {}
    if honesty.get("unproven_claim", True):
        errors.append("honesty.unproven_claim must be False")
    if honesty.get("continuum_claim", True):
        errors.append("honesty.continuum_claim must be False")
    if honesty.get("yang_mills_claim", True):
        errors.append("honesty.yang_mills_claim must be False")
    if cert.get("trial_flagged") not in (True, False):
        errors.append("trial_flagged must be a bool")
    cond = cert.get("trial_gram_condition")
    if cond is not None:
        cond_f = _as_float(cond)
        if cond_f is None or cond_f < 0.0:
            errors.append("trial_gram_condition must be None or a non-negative float")
    remainder = cert.get("trial_remainder_width")
    if remainder is not None:
        rem_f = _as_float(remainder)
        if rem_f is None or rem_f < 0.0:
            errors.append("trial_remainder_width must be a non-negative float")
    ratio = _as_float(cert.get("subdominant_ratio_upper"))
    if ratio is None or not 0.0 <= ratio < 1.0:
        errors.append("subdominant_ratio_upper must lie in [0, 1)")
    gap = _as_float(cert.get("spectral_gap_lower"))
    if gap is None or gap <= 0.0:
        errors.append("spectral_gap_lower must be > 0")
    spacing = _as_float(cert.get("lattice_spacing"))
    if spacing is None or spacing <= 0.0:
        errors.append("lattice_spacing must be > 0")
    if "digest" in cert and not verify_certificate_digest(cert):
        errors.append("digest does not match the certificate body (tampered/stale)")
    return errors


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def replay_transfer_matrix_gap(cert: Certificate) -> bool | None:
    r"""Independently re-derive the gap by rebuilding the matrix from its inputs.

    Reconstructs the transfer matrix from the recorded ``parameters`` and re-runs
    the certified gap engine, then checks that the sealed bound is **not stronger**
    than the freshly derived one.  A tampered ``subdominant_ratio_upper`` (or a
    ``spectral_gap_lower`` inflated beyond what the matrix supports) fails here even
    though the interval arithmetic behind it was sound, because the replay never
    reads the sealed numbers to produce its own.

    Returns ``None`` when there is nothing to replay (no recorded parameters), and
    never raises.
    """
    from omnibias.geometry.gauge.transfer.gap import certified_transfer_matrix_gap
    from omnibias.geometry.gauge.transfer.trial import holonomy_trial_space

    parameters = cert.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        return None
    try:
        transfer = rebuild(parameters)
        trial = (
            holonomy_trial_space(transfer)
            if cert.get("trial_gram_condition") is not None
            else None
        )
        fresh = certified_transfer_matrix_gap(
            transfer,
            lattice_spacing=float(cert.get("lattice_spacing", 1.0)),
            trial=trial,
        )
    except (ValueError, TypeError, KeyError):
        return False

    sealed_ratio = _as_float(cert.get("subdominant_ratio_upper"))
    sealed_gap = _as_float(cert.get("spectral_gap_lower"))
    if sealed_ratio is None or sealed_gap is None:
        return False
    # A valid seal may be looser than the replay (a different engine could win a
    # tie differently) but never tighter: claiming a smaller ratio or a bigger gap
    # than an independent derivation supports is exactly the forgery to catch.
    tolerance = 1e-9
    if sealed_ratio < fresh.subdominant_ratio_upper - tolerance:
        return False
    return not sealed_gap > fresh.spectral_gap_lower + tolerance


def seal_strong_coupling_certificate(
    result: StrongCouplingGapResult,
    *,
    claim: str = "",
) -> Certificate:
    """Seal a certified polymer bound.  Refuses an uncertified (out-of-domain) result."""
    if not result.certified:
        raise ValueError(
            "refusing to seal an uncertified strong-coupling bound "
            "(out of domain or non-positive gap)"
        )
    return seal_certificate(
        {
            "schema_version": STRONG_COUPLING_SCHEMA_VERSION,
            "claim": claim or "SU(2) strong-coupling polymer gap",
            "observable": STRONG_COUPLING_KIND,
            "model": "su2_wilson_polymer",
            "method": result.method,
            "beta": float(result.beta),
            "spacetime_dim": int(result.spacetime_dim),
            "coordination": int(result.coordination),
            "first_step": result.first_step,
            "counting": result.counting,
            "subdominant_ratio_upper": float(result.subdominant_ratio_upper),
            "spectral_gap_lower": float(result.spectral_gap_lower),
            "in_convergence_domain": True,
            "continuum_claim": False,
            "honesty": {
                "unproven_claim": False,
                "continuum_claim": False,
                "fixed_spacing": True,
                "finite_truncation": True,
                "interval_verified": True,
                "yang_mills_claim": False,
                "note": _STRONG_COUPLING_NOTE,
            },
        }
    )


def strong_coupling_schema_errors(cert: Certificate) -> list[str]:
    """Validate a ``verified-strong-coupling-gap-1`` certificate."""
    errors: list[str] = []
    required = (
        "schema_version",
        "observable",
        "model",
        "method",
        "beta",
        "spacetime_dim",
        "coordination",
        "subdominant_ratio_upper",
        "spectral_gap_lower",
        "in_convergence_domain",
        "continuum_claim",
        "honesty",
        "digest",
    )
    for key in required:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    if cert.get("schema_version") != STRONG_COUPLING_SCHEMA_VERSION:
        errors.append(f"schema_version must be {STRONG_COUPLING_SCHEMA_VERSION!r}")
    if cert.get("continuum_claim", True):
        errors.append("continuum_claim must be False")
    if cert.get("method") not in (
        POLYMER_METHOD,
        BACKTRACK_POLYMER_METHOD,
        CRUDE_POLYMER_METHOD,
    ):
        errors.append(
            f"method must be {POLYMER_METHOD!r}, {BACKTRACK_POLYMER_METHOD!r}, "
            f"or {CRUDE_POLYMER_METHOD!r}"
        )
    if cert.get("in_convergence_domain") is not True:
        errors.append("in_convergence_domain must be True")
    honesty = cert.get("honesty", {})
    if not isinstance(honesty, Mapping):
        errors.append("honesty must be a mapping")
        honesty = {}
    if honesty.get("unproven_claim", True):
        errors.append("honesty.unproven_claim must be False")
    if honesty.get("continuum_claim", True):
        errors.append("honesty.continuum_claim must be False")
    if honesty.get("yang_mills_claim", True):
        errors.append("honesty.yang_mills_claim must be False")
    ratio = _as_float(cert.get("subdominant_ratio_upper"))
    if ratio is None or not 0.0 <= ratio < 1.0:
        errors.append("subdominant_ratio_upper must lie in [0, 1)")
    gap = _as_float(cert.get("spectral_gap_lower"))
    if gap is None or gap <= 0.0:
        errors.append("spectral_gap_lower must be > 0")
    beta = _as_float(cert.get("beta"))
    if beta is None or beta <= 0.0:
        errors.append("beta must be > 0")
    if "digest" in cert and not verify_certificate_digest(cert):
        errors.append("digest does not match the certificate body (tampered/stale)")
    return errors


def replay_strong_coupling_gap(cert: Certificate) -> bool | None:
    """Independently re-derive the polymer bound from the recorded ``beta``."""
    beta = _as_float(cert.get("beta"))
    dim = cert.get("spacetime_dim")
    if beta is None or not isinstance(dim, int) or isinstance(dim, bool):
        return None
    counting = cert.get("counting", "two_scale")
    if counting not in ("two_scale", "backtrack", "crude"):
        return False
    try:
        fresh = certified_strong_coupling_glueball_bound(
            beta, spacetime_dim=dim, counting=counting
        )
    except (ValueError, TypeError):
        return False
    if not fresh.certified:
        return False
    sealed_ratio = _as_float(cert.get("subdominant_ratio_upper"))
    sealed_gap = _as_float(cert.get("spectral_gap_lower"))
    if sealed_ratio is None or sealed_gap is None:
        return False
    tolerance = 1e-9
    if sealed_ratio < fresh.subdominant_ratio_upper - tolerance:
        return False
    return not sealed_gap > fresh.spectral_gap_lower + tolerance


#: Schema version of the sealed two-plaquette Hamiltonian gap certificate.
HAMILTONIAN_GAP_SCHEMA_VERSION = "verified-hamiltonian-gap-1"

#: The :class:`~omnibias.core.proof.Conjecture` kind this certificate answers.
HAMILTONIAN_GAP_KIND = "two_plaquette_hamiltonian_gap"

#: Kind for the three-plaquette chain (same schema, different observable).
THREE_PLAQUETTE_GAP_KIND = "three_plaquette_hamiltonian_gap"

_HAMILTONIAN_NOTES = {
    HAMILTONIAN_GAP_KIND: (
        "certified spectral gap of one finite two-plaquette SU(2) Kogut-Susskind "
        "Hamiltonian at one coupling and one spin truncation; NOT a continuum-limit, "
        "infinite-volume, or Yang-Mills mass-gap claim"
    ),
    THREE_PLAQUETTE_GAP_KIND: (
        "certified spectral gap of one finite three-plaquette SU(2) Kogut-Susskind "
        "Hamiltonian at one coupling and one spin truncation; NOT a continuum-limit, "
        "infinite-volume, or Yang-Mills mass-gap claim"
    ),
}
_HAMILTONIAN_NOTE = _HAMILTONIAN_NOTES[HAMILTONIAN_GAP_KIND]


def seal_hamiltonian_gap_certificate(
    result: Any,
    hamiltonian: Any,
    *,
    claim: str = "",
) -> Certificate:
    """Seal a :class:`~.hamiltonian.HamiltonianGapResult`.  Refuses an uncertified gap."""
    if not result.certified:
        raise ValueError(
            "refusing to seal an uncertified Hamiltonian gap "
            "(non-positive λ1-λ0 lower bound)"
        )
    model = getattr(hamiltonian, "model", "")
    observable = (
        THREE_PLAQUETTE_GAP_KIND
        if model == "su2_three_plaquette"
        else HAMILTONIAN_GAP_KIND
    )
    return seal_certificate(
        {
            "schema_version": HAMILTONIAN_GAP_SCHEMA_VERSION,
            "claim": claim or f"{result.model} Hamiltonian gap",
            "observable": observable,
            "model": result.model,
            "method": result.method,
            "dimension": int(result.dimension),
            "coupling": float(result.coupling),
            "j_max": int(result.j_max),
            "subdominant_ratio_upper": float(result.subdominant_ratio_upper),
            "spectral_gap_lower": float(result.spectral_gap_lower),
            "lambda0_upper": float(result.lambda0_upper),
            "lambda1_lower": float(result.lambda1_lower),
            "parameters": dict(hamiltonian.parameters),
            "trial_gram_condition": result.trial_gram_condition,
            "trial_flagged": bool(result.trial_flagged),
            "trial_remainder_width": float(result.trial_remainder_width),
            "continuum_claim": False,
            "honesty": {
                "unproven_claim": False,
                "continuum_claim": False,
                "fixed_spacing": True,
                "finite_truncation": True,
                "interval_verified": True,
                "yang_mills_claim": False,
                "note": _HAMILTONIAN_NOTES[observable],
            },
        }
    )


def hamiltonian_gap_schema_errors(cert: Certificate) -> list[str]:
    """Validate a ``verified-hamiltonian-gap-1`` certificate."""
    errors: list[str] = []
    required = (
        "schema_version",
        "observable",
        "model",
        "method",
        "dimension",
        "coupling",
        "j_max",
        "subdominant_ratio_upper",
        "spectral_gap_lower",
        "parameters",
        "trial_gram_condition",
        "trial_flagged",
        "trial_remainder_width",
        "continuum_claim",
        "honesty",
        "digest",
    )
    for key in required:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    if cert.get("schema_version") != HAMILTONIAN_GAP_SCHEMA_VERSION:
        errors.append(f"schema_version must be {HAMILTONIAN_GAP_SCHEMA_VERSION!r}")
    if cert.get("continuum_claim", True):
        errors.append("continuum_claim must be False")
    if cert.get("observable") not in (HAMILTONIAN_GAP_KIND, THREE_PLAQUETTE_GAP_KIND):
        errors.append(
            f"observable must be {HAMILTONIAN_GAP_KIND!r} or {THREE_PLAQUETTE_GAP_KIND!r}"
        )
    honesty = cert.get("honesty", {})
    if not isinstance(honesty, Mapping):
        errors.append("honesty must be a mapping")
        honesty = {}
    if honesty.get("unproven_claim", True):
        errors.append("honesty.unproven_claim must be False")
    if honesty.get("continuum_claim", True):
        errors.append("honesty.continuum_claim must be False")
    if honesty.get("yang_mills_claim", True):
        errors.append("honesty.yang_mills_claim must be False")
    ratio = _as_float(cert.get("subdominant_ratio_upper"))
    if ratio is None or not 0.0 <= ratio < 1.0:
        errors.append("subdominant_ratio_upper must lie in [0, 1)")
    gap = _as_float(cert.get("spectral_gap_lower"))
    if gap is None or gap <= 0.0:
        errors.append("spectral_gap_lower must be > 0")
    coupling = _as_float(cert.get("coupling"))
    if coupling is None or coupling <= 0.0:
        errors.append("coupling must be > 0")
    j_max = cert.get("j_max")
    if not isinstance(j_max, int) or isinstance(j_max, bool) or j_max < 1:
        errors.append("j_max must be an integer >= 1")
    if "digest" in cert and not verify_certificate_digest(cert):
        errors.append("digest does not match the certificate body (tampered/stale)")
    return errors


def replay_hamiltonian_gap(cert: Certificate) -> bool | None:
    """Independently re-derive the Hamiltonian gap from ``(coupling, j_max)``."""
    from omnibias.geometry.gauge.transfer.hamiltonian import (
        certified_hamiltonian_gap,
        plaquette_holonomy_trial_space,
        rebuild_hamiltonian,
    )

    parameters = cert.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        return None
    try:
        hamiltonian = rebuild_hamiltonian(parameters)
        trial = (
            plaquette_holonomy_trial_space(hamiltonian)
            if cert.get("trial_gram_condition") is not None
            else None
        )
        fresh = certified_hamiltonian_gap(hamiltonian, trial=trial)
    except (ValueError, TypeError, KeyError):
        return False
    if not fresh.certified:
        return False
    sealed_ratio = _as_float(cert.get("subdominant_ratio_upper"))
    sealed_gap = _as_float(cert.get("spectral_gap_lower"))
    if sealed_ratio is None or sealed_gap is None:
        return False
    tolerance = 1e-9
    if sealed_ratio < fresh.subdominant_ratio_upper - tolerance:
        return False
    return not sealed_gap > fresh.spectral_gap_lower + tolerance


STRIP_RP_SCHEMA_VERSION = "verified-strip-rp-1"
STRIP_RP_KIND = "strip_reflection_positivity"

_STRIP_RP_NOTE = (
    "reflection positivity of one finite spatial-strip transfer on a locked "
    "angle inversion and test vectors; NOT Osterwalder-Seiler reconstruction, "
    "NOT a continuum-limit, infinite-volume, or Yang-Mills mass-gap claim"
)


def seal_strip_rp_certificate(
    result: Any,
    transfer: TransferMatrix,
    *,
    claim: str = "",
) -> Certificate:
    """Seal a certified strip RP check.  Refuses an uncertified result."""
    if not result.certified:
        raise ValueError("refusing to seal an uncertified strip reflection-positivity check")
    return seal_certificate(
        {
            "schema_version": STRIP_RP_SCHEMA_VERSION,
            "claim": claim or "spatial-strip reflection positivity",
            "observable": STRIP_RP_KIND,
            "model": transfer.model,
            "method": result.method,
            "dimension": int(transfer.dimension),
            "forms_lo": [float(form.lo) for form in result.forms],
            "parameters": dict(transfer.parameters),
            "continuum_claim": False,
            "honesty": {
                "unproven_claim": False,
                "continuum_claim": False,
                "fixed_matrix": True,
                "finite_truncation": True,
                "interval_verified": True,
                "yang_mills_claim": False,
                "note": _STRIP_RP_NOTE,
            },
        }
    )


def strip_rp_schema_errors(cert: Certificate) -> list[str]:
    """Validate a ``verified-strip-rp-1`` certificate."""
    errors: list[str] = []
    required = (
        "schema_version",
        "observable",
        "model",
        "method",
        "dimension",
        "forms_lo",
        "parameters",
        "continuum_claim",
        "honesty",
        "digest",
    )
    for key in required:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    if cert.get("schema_version") != STRIP_RP_SCHEMA_VERSION:
        errors.append(f"schema_version must be {STRIP_RP_SCHEMA_VERSION!r}")
    if cert.get("continuum_claim", True):
        errors.append("continuum_claim must be False")
    if cert.get("observable") != STRIP_RP_KIND:
        errors.append(f"observable must be {STRIP_RP_KIND!r}")
    honesty = cert.get("honesty", {})
    if not isinstance(honesty, Mapping):
        errors.append("honesty must be a mapping")
        honesty = {}
    if honesty.get("unproven_claim", True):
        errors.append("honesty.unproven_claim must be False")
    if honesty.get("continuum_claim", True):
        errors.append("honesty.continuum_claim must be False")
    if honesty.get("yang_mills_claim", True):
        errors.append("honesty.yang_mills_claim must be False")
    forms = cert.get("forms_lo")
    if not isinstance(forms, list) or not forms:
        errors.append("forms_lo must be a non-empty list")
    elif any(not isinstance(item, int | float) or isinstance(item, bool) or item < 0.0 for item in forms):
        errors.append("forms_lo entries must be non-negative floats")
    if "digest" in cert and not verify_certificate_digest(cert):
        errors.append("digest does not match the certificate body (tampered/stale)")
    return errors


def replay_strip_rp(cert: Certificate) -> bool | None:
    """Independently re-derive strip RP from the recorded constructor inputs."""
    from omnibias.geometry.gauge.transfer.strip import certified_strip_reflection_positivity

    parameters = cert.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        return None
    try:
        transfer = rebuild(parameters)
        fresh = certified_strip_reflection_positivity(transfer)
    except (ValueError, TypeError, KeyError):
        return False
    if not fresh.certified:
        return False
    sealed = cert.get("forms_lo")
    if not isinstance(sealed, list) or len(sealed) != len(fresh.forms):
        return False
    tolerance = 1e-9
    for claimed, form in zip(sealed, fresh.forms, strict=True):
        value = _as_float(claimed)
        if value is None or value < form.lo - tolerance:
            return False
    return True


__all__ = [
    "HAMILTONIAN_GAP_KIND",
    "HAMILTONIAN_GAP_SCHEMA_VERSION",
    "STRIP_RP_KIND",
    "STRIP_RP_SCHEMA_VERSION",
    "THREE_PLAQUETTE_GAP_KIND",
    "STRONG_COUPLING_KIND",
    "STRONG_COUPLING_SCHEMA_VERSION",
    "TRANSFER_GAP_KIND",
    "TRANSFER_GAP_SCHEMA_VERSION",
    "hamiltonian_gap_schema_errors",
    "replay_hamiltonian_gap",
    "replay_strong_coupling_gap",
    "replay_strip_rp",
    "replay_transfer_matrix_gap",
    "seal_hamiltonian_gap_certificate",
    "seal_strong_coupling_certificate",
    "seal_strip_rp_certificate",
    "seal_transfer_gap_certificate",
    "strip_rp_schema_errors",
    "strong_coupling_schema_errors",
    "transfer_gap_schema_errors",
]
