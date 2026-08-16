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
    CLUSTER_POLYMER_METHOD,
    CRUDE_POLYMER_METHOD,
    POLYMER_BETA_DOMAIN_METHOD,
    POLYMER_METHOD,
    WILSON_CHARACTER_BETA_DOMAIN_METHOD,
    PolymerDomainResult,
    StrongCouplingGapResult,
    WilsonCharacterDomainResult,
    certified_polymer_beta_domain,
    certified_strong_coupling_glueball_bound,
    certified_wilson_character_beta_domain,
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
    "crude_polymer_count, or finite_polymer_cluster, NOT Osterwalder-Seiler, "
    "NOT a continuum-limit, "
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
            "n_keep": result.n_keep,
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
        CLUSTER_POLYMER_METHOD,
    ):
        errors.append(
            f"method must be {POLYMER_METHOD!r}, {BACKTRACK_POLYMER_METHOD!r}, "
            f"{CRUDE_POLYMER_METHOD!r}, or {CLUSTER_POLYMER_METHOD!r}"
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
    if counting not in ("two_scale", "backtrack", "crude", "cluster"):
        return False
    keep = cert.get("n_keep", 3)
    kwargs: dict[str, object] = {"spacetime_dim": dim, "counting": counting}
    if counting == "cluster":
        if not isinstance(keep, int) or isinstance(keep, bool) or keep < 2:
            return False
        kwargs["n_keep"] = keep
    try:
        fresh = certified_strong_coupling_glueball_bound(beta, **kwargs)  # type: ignore[arg-type]
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


POLYMER_DOMAIN_SCHEMA_VERSION = "verified-polymer-beta-domain-1"
POLYMER_DOMAIN_KIND = "polymer_beta_domain"

_POLYMER_DOMAIN_NOTE = (
    "majorant domain on a locked dyadic beta grid: largest certifying "
    "grid point and the next grid failure; NOT a physical critical "
    "coupling, NOT Osterwalder-Seiler, NOT a continuum-limit, "
    "infinite-volume, or Yang-Mills mass-gap claim"
)


def _fraction_pair(value: Any) -> list[int] | None:
    from fractions import Fraction

    if isinstance(value, Fraction):
        return [int(value.numerator), int(value.denominator)]
    return None


def _as_fraction(value: Any) -> Any:
    from fractions import Fraction

    if isinstance(value, Fraction):
        return value
    if isinstance(value, list | tuple) and len(value) == 2:
        num, den = value
        if (
            isinstance(num, int)
            and isinstance(den, int)
            and not isinstance(num, bool)
            and not isinstance(den, bool)
            and den > 0
        ):
            return Fraction(num, den)
    if isinstance(value, str) and "/" in value:
        return Fraction(value)
    return None


def seal_polymer_domain_certificate(
    result: PolymerDomainResult,
    *,
    claim: str = "",
) -> Certificate:
    """Seal a certified polymer-domain grid statement.  Refuses an uncertified result."""
    if not result.certified:
        raise ValueError(
            "refusing to seal an uncertified polymer-domain result "
            "(no certifying grid point, or no larger failure)"
        )
    certified_pair = _fraction_pair(result.beta_certified)
    outside_pair = _fraction_pair(result.beta_outside)
    if certified_pair is None or outside_pair is None:
        raise ValueError("beta_certified and beta_outside must be Fractions")
    return seal_certificate(
        {
            "schema_version": POLYMER_DOMAIN_SCHEMA_VERSION,
            "claim": claim or "SU(2) polymer majorant beta-domain",
            "observable": POLYMER_DOMAIN_KIND,
            "model": "su2_wilson_polymer_domain",
            "method": result.method,
            "counting": result.counting,
            "spacetime_dim": int(result.spacetime_dim),
            "n_keep": result.n_keep,
            "grid": [_fraction_pair(item) for item in result.grid],
            "beta_certified": certified_pair,
            "beta_outside": outside_pair,
            "subdominant_ratio_upper": float(result.certified_result.subdominant_ratio_upper),
            "spectral_gap_lower": float(result.certified_result.spectral_gap_lower),
            "in_convergence_domain": True,
            "continuum_claim": False,
            "honesty": {
                "unproven_claim": False,
                "continuum_claim": False,
                "fixed_spacing": True,
                "finite_truncation": True,
                "interval_verified": True,
                "yang_mills_claim": False,
                "note": _POLYMER_DOMAIN_NOTE,
            },
        }
    )


def polymer_domain_schema_errors(cert: Certificate) -> list[str]:
    """Validate a ``verified-polymer-beta-domain-1`` certificate."""
    errors: list[str] = []
    required = (
        "schema_version",
        "observable",
        "model",
        "method",
        "counting",
        "spacetime_dim",
        "grid",
        "beta_certified",
        "beta_outside",
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
    if cert.get("schema_version") != POLYMER_DOMAIN_SCHEMA_VERSION:
        errors.append(f"schema_version must be {POLYMER_DOMAIN_SCHEMA_VERSION!r}")
    if cert.get("observable") != POLYMER_DOMAIN_KIND:
        errors.append(f"observable must be {POLYMER_DOMAIN_KIND!r}")
    if cert.get("continuum_claim", True):
        errors.append("continuum_claim must be False")
    if cert.get("method") != POLYMER_BETA_DOMAIN_METHOD:
        errors.append(f"method must be {POLYMER_BETA_DOMAIN_METHOD!r}")
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
    beta_star = _as_fraction(cert.get("beta_certified"))
    beta_out = _as_fraction(cert.get("beta_outside"))
    if beta_star is None or beta_out is None or not (beta_star < beta_out):
        errors.append("beta_certified must be a Fraction pair strictly below beta_outside")
    ratio = _as_float(cert.get("subdominant_ratio_upper"))
    if ratio is None or not 0.0 <= ratio < 1.0:
        errors.append("subdominant_ratio_upper must lie in [0, 1)")
    gap = _as_float(cert.get("spectral_gap_lower"))
    if gap is None or gap <= 0.0:
        errors.append("spectral_gap_lower must be > 0")
    if "digest" in cert and not verify_certificate_digest(cert):
        errors.append("digest does not match the certificate body (tampered/stale)")
    return errors


def replay_polymer_domain(cert: Certificate) -> bool | None:
    """Independently re-derive the majorant domain from the recorded grid."""
    counting = cert.get("counting", "two_scale")
    if counting not in ("two_scale", "backtrack", "crude", "cluster"):
        return False
    dim = cert.get("spacetime_dim")
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 2:
        return None
    raw_grid = cert.get("grid")
    if not isinstance(raw_grid, list) or len(raw_grid) < 2:
        return None
    grid = []
    for item in raw_grid:
        parsed = _as_fraction(item)
        if parsed is None:
            return False
        grid.append(parsed)
    keep = cert.get("n_keep", 3)
    kwargs: dict[str, object] = {"counting": counting, "spacetime_dim": dim, "grid": grid}
    if counting == "cluster":
        if not isinstance(keep, int) or isinstance(keep, bool) or keep < 2:
            return False
        kwargs["n_keep"] = keep
    try:
        fresh = certified_polymer_beta_domain(**kwargs)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return False
    if not fresh.certified:
        return False
    sealed_star = _as_fraction(cert.get("beta_certified"))
    sealed_out = _as_fraction(cert.get("beta_outside"))
    if sealed_star is None or sealed_out is None:
        return False
    if sealed_star > fresh.beta_certified:
        return False
    if sealed_star == fresh.beta_certified and sealed_out != fresh.beta_outside:
        return False
    sealed_ratio = _as_float(cert.get("subdominant_ratio_upper"))
    sealed_gap = _as_float(cert.get("spectral_gap_lower"))
    if sealed_ratio is None or sealed_gap is None:
        return False
    tolerance = 1e-9
    if sealed_ratio < fresh.certified_result.subdominant_ratio_upper - tolerance:
        return False
    return not sealed_gap > fresh.certified_result.spectral_gap_lower + tolerance


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
TORUS_RP_SCHEMA_VERSION = "verified-torus-rp-1"
TORUS_RP_KIND = "torus_reflection_positivity"

_STRIP_RP_NOTE = (
    "reflection positivity of one finite spatial-strip transfer on a locked "
    "angle inversion and test vectors; NOT Osterwalder-Seiler reconstruction, "
    "NOT a continuum-limit, infinite-volume, or Yang-Mills mass-gap claim"
)
_TORUS_RP_NOTE = (
    "reflection positivity of one finite 2-D spatial-torus transfer on a "
    "locked angle inversion and test vectors; NOT Osterwalder-Seiler "
    "reconstruction, NOT a continuum-limit, infinite-volume, or Yang-Mills "
    "mass-gap claim"
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
    torus = transfer.model == "su2_spatial_torus"
    return seal_certificate(
        {
            "schema_version": TORUS_RP_SCHEMA_VERSION if torus else STRIP_RP_SCHEMA_VERSION,
            "claim": claim
            or (
                "spatial-torus reflection positivity"
                if torus
                else "spatial-strip reflection positivity"
            ),
            "observable": TORUS_RP_KIND if torus else STRIP_RP_KIND,
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
                "note": _TORUS_RP_NOTE if torus else _STRIP_RP_NOTE,
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
    schema = cert.get("schema_version")
    observable = cert.get("observable")
    if schema == TORUS_RP_SCHEMA_VERSION:
        if observable != TORUS_RP_KIND:
            errors.append(f"observable must be {TORUS_RP_KIND!r}")
    elif schema == STRIP_RP_SCHEMA_VERSION:
        if observable != STRIP_RP_KIND:
            errors.append(f"observable must be {STRIP_RP_KIND!r}")
    else:
        errors.append(
            f"schema_version must be {STRIP_RP_SCHEMA_VERSION!r} or "
            f"{TORUS_RP_SCHEMA_VERSION!r}"
        )
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


WILSON_CHARACTER_DOMAIN_SCHEMA_VERSION = "verified-wilson-character-beta-domain-1"
WILSON_CHARACTER_DOMAIN_KIND = "wilson_character_beta_domain"

_WILSON_CHARACTER_DOMAIN_NOTE = (
    "0+1-D infinite character-basis Wilson gap on a locked beta grid "
    "that includes the polymer two-scale failure 1/4; NOT 4-D Yang-Mills, "
    "NOT a physical critical coupling, NOT a continuum-limit, "
    "infinite-volume, or mass-gap claim"
)


def seal_wilson_character_domain_certificate(
    result: WilsonCharacterDomainResult,
    *,
    claim: str = "",
) -> Certificate:
    """Seal a certified Wilson-character domain.  Refuses an uncertified result."""
    if not result.certified:
        raise ValueError(
            "refusing to seal an uncertified Wilson-character domain "
            "(1/4 must certify, and at least one strictly larger grid point)"
        )
    certified_pair = _fraction_pair(result.beta_certified)
    if certified_pair is None:
        raise ValueError("beta_certified must be a Fraction")
    outside_pair = (
        None if result.beta_outside is None else _fraction_pair(result.beta_outside)
    )
    if result.beta_outside is not None and outside_pair is None:
        raise ValueError("beta_outside must be a Fraction when the grid is not exhausted")
    return seal_certificate(
        {
            "schema_version": WILSON_CHARACTER_DOMAIN_SCHEMA_VERSION,
            "claim": claim or "SU(2) Wilson character beta-domain",
            "observable": WILSON_CHARACTER_DOMAIN_KIND,
            "model": "su2_wilson_character_domain",
            "method": result.method,
            "grid": [_fraction_pair(item) for item in result.grid],
            "beta_certified": certified_pair,
            "beta_outside": outside_pair,
            "grid_exhausted": bool(result.grid_exhausted),
            "quarter_certified": bool(result.quarter_certified),
            "subdominant_ratio_upper": float(result.certified_result.subdominant_ratio_upper),
            "spectral_gap_lower": float(result.certified_result.spectral_gap_lower),
            "continuum_claim": False,
            "honesty": {
                "unproven_claim": False,
                "continuum_claim": False,
                "fixed_spacing": True,
                "finite_truncation": True,
                "interval_verified": True,
                "yang_mills_claim": False,
                "note": _WILSON_CHARACTER_DOMAIN_NOTE,
            },
        }
    )


def wilson_character_domain_schema_errors(cert: Certificate) -> list[str]:
    """Validate a ``verified-wilson-character-beta-domain-1`` certificate."""
    errors: list[str] = []
    required = (
        "schema_version",
        "observable",
        "model",
        "method",
        "grid",
        "beta_certified",
        "beta_outside",
        "grid_exhausted",
        "quarter_certified",
        "subdominant_ratio_upper",
        "spectral_gap_lower",
        "continuum_claim",
        "honesty",
        "digest",
    )
    for key in required:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    if cert.get("schema_version") != WILSON_CHARACTER_DOMAIN_SCHEMA_VERSION:
        errors.append(f"schema_version must be {WILSON_CHARACTER_DOMAIN_SCHEMA_VERSION!r}")
    if cert.get("observable") != WILSON_CHARACTER_DOMAIN_KIND:
        errors.append(f"observable must be {WILSON_CHARACTER_DOMAIN_KIND!r}")
    if cert.get("continuum_claim", True):
        errors.append("continuum_claim must be False")
    if cert.get("method") != WILSON_CHARACTER_BETA_DOMAIN_METHOD:
        errors.append(f"method must be {WILSON_CHARACTER_BETA_DOMAIN_METHOD!r}")
    if cert.get("quarter_certified") is not True:
        errors.append("quarter_certified must be True")
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
    beta_star = _as_fraction(cert.get("beta_certified"))
    if beta_star is None:
        errors.append("beta_certified must be a Fraction pair")
    exhausted = cert.get("grid_exhausted")
    if exhausted is True:
        if cert.get("beta_outside") is not None:
            errors.append("beta_outside must be null when grid_exhausted is True")
    elif exhausted is False:
        beta_out = _as_fraction(cert.get("beta_outside"))
        if beta_star is None or beta_out is None or not (beta_star < beta_out):
            errors.append(
                "beta_certified must be a Fraction pair strictly below beta_outside"
            )
    else:
        errors.append("grid_exhausted must be a bool")
    ratio = _as_float(cert.get("subdominant_ratio_upper"))
    if ratio is None or not 0.0 <= ratio < 1.0:
        errors.append("subdominant_ratio_upper must lie in [0, 1)")
    gap = _as_float(cert.get("spectral_gap_lower"))
    if gap is None or gap <= 0.0:
        errors.append("spectral_gap_lower must be > 0")
    if "digest" in cert and not verify_certificate_digest(cert):
        errors.append("digest does not match the certificate body (tampered/stale)")
    return errors


def replay_wilson_character_domain(cert: Certificate) -> bool | None:
    """Independently re-derive the Wilson-character domain from the recorded grid."""
    raw_grid = cert.get("grid")
    if not isinstance(raw_grid, list) or len(raw_grid) < 2:
        return None
    grid = []
    for item in raw_grid:
        parsed = _as_fraction(item)
        if parsed is None:
            return False
        grid.append(parsed)
    try:
        fresh = certified_wilson_character_beta_domain(grid=grid)
    except (ValueError, TypeError):
        return False
    if not fresh.certified:
        return False
    sealed_star = _as_fraction(cert.get("beta_certified"))
    if sealed_star is None or sealed_star > fresh.beta_certified:
        return False
    if cert.get("quarter_certified") is True and not fresh.quarter_certified:
        return False
    if cert.get("grid_exhausted") is True and not fresh.grid_exhausted:
        return False
    if fresh.beta_outside is not None:
        sealed_out = _as_fraction(cert.get("beta_outside"))
        if sealed_star == fresh.beta_certified and sealed_out != fresh.beta_outside:
            return False
    sealed_ratio = _as_float(cert.get("subdominant_ratio_upper"))
    sealed_gap = _as_float(cert.get("spectral_gap_lower"))
    if sealed_ratio is None or sealed_gap is None:
        return False
    tolerance = 1e-9
    if sealed_ratio < fresh.certified_result.subdominant_ratio_upper - tolerance:
        return False
    return not sealed_gap > fresh.certified_result.spectral_gap_lower + tolerance


FINITE_GAUGE_REPORT_SCHEMA_VERSION = "verified-finite-gauge-report-1"
FINITE_GAUGE_REPORT_KIND = "finite_gauge_report"

_FINITE_GAUGE_REPORT_NOTE = (
    "bundle of finite certificates on one named spec; NOT a continuum-limit, "
    "infinite-volume, Osterwalder-Seiler, or Yang-Mills mass-gap claim, "
    "and not a staircase to Clay existence"
)


def seal_finite_gauge_report_certificate(
    result: Any,
    *,
    claim: str = "",
) -> Certificate:
    """Seal a certified finite-gauge report.  Refuses an uncertified bundle."""
    from omnibias.geometry.gauge.transfer.report import finite_gauge_spec_to_mapping

    if not result.certified:
        raise ValueError(
            "refusing to seal an uncertified finite-gauge report "
            "(a required engine failed or honesty flags were raised)"
        )
    polymer_gaps = [float(item.spectral_gap_lower) for item in result.polymer]
    polymer_ratios = [float(item.subdominant_ratio_upper) for item in result.polymer]
    return seal_certificate(
        {
            "schema_version": FINITE_GAUGE_REPORT_SCHEMA_VERSION,
            "claim": claim or f"{result.spec.name} finite gauge report",
            "observable": FINITE_GAUGE_REPORT_KIND,
            "model": result.spec.name,
            "method": "finite_gauge_report",
            "spec": finite_gauge_spec_to_mapping(result.spec),
            "polymer_gaps": polymer_gaps,
            "polymer_ratios": polymer_ratios,
            "wilson_gap": float(result.wilson_character.spectral_gap_lower),
            "domain_beta_certified": [
                int(result.polymer_domain.beta_certified.numerator),
                int(result.polymer_domain.beta_certified.denominator),
            ],
            "domain_beta_outside": [
                int(result.polymer_domain.beta_outside.numerator),
                int(result.polymer_domain.beta_outside.denominator),
            ],
            "wilson_domain_beta_certified": [
                int(result.wilson_character_domain.beta_certified.numerator),
                int(result.wilson_character_domain.beta_certified.denominator),
            ],
            "wilson_domain_beta_outside": (
                None
                if result.wilson_character_domain.beta_outside is None
                else [
                    int(result.wilson_character_domain.beta_outside.numerator),
                    int(result.wilson_character_domain.beta_outside.denominator),
                ]
            ),
            "wilson_domain_grid_exhausted": bool(
                result.wilson_character_domain.grid_exhausted
            ),
            "haar_weyl_prefactor_24": result.haar.weyl_prefactor_24,
            "haar_su3_dim_3_0": result.haar.su3_dim_3_0,
            "su3_dimension": int(result.su3_gap.dimension),
            "su3_gap": float(result.su3_gap.spectral_gap_lower),
            "hamiltonian_gap": float(result.hamiltonian.spectral_gap_lower),
            "three_plaquette_gap": float(result.three_plaquette.spectral_gap_lower),
            "g1_factor": float(result.g1.factor),
            "g1_ge_generic": bool(result.g1.ge_generic),
            "g1_target_5x": False,
            "strip_rp": bool(result.strip_rp.certified),
            "include_torus": bool(result.spec.include_torus),
            "scaling_gaps": [float(point.spectral_gap_lower) for point in result.scaling.points],
            "continuum_claim": False,
            "honesty": {
                "unproven_claim": False,
                "continuum_claim": False,
                "fixed_matrix": True,
                "finite_truncation": True,
                "interval_verified": True,
                "yang_mills_claim": False,
                "note": _FINITE_GAUGE_REPORT_NOTE,
            },
        }
    )


def finite_gauge_report_schema_errors(cert: Certificate) -> list[str]:
    """Validate a ``verified-finite-gauge-report-1`` certificate."""
    errors: list[str] = []
    required = (
        "schema_version",
        "observable",
        "model",
        "method",
        "spec",
        "polymer_gaps",
        "wilson_gap",
        "domain_beta_certified",
        "domain_beta_outside",
        "wilson_domain_beta_certified",
        "wilson_domain_beta_outside",
        "wilson_domain_grid_exhausted",
        "haar_weyl_prefactor_24",
        "haar_su3_dim_3_0",
        "su3_dimension",
        "su3_gap",
        "hamiltonian_gap",
        "three_plaquette_gap",
        "g1_factor",
        "g1_ge_generic",
        "g1_target_5x",
        "strip_rp",
        "scaling_gaps",
        "continuum_claim",
        "honesty",
        "digest",
    )
    for key in required:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    if cert.get("schema_version") != FINITE_GAUGE_REPORT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {FINITE_GAUGE_REPORT_SCHEMA_VERSION!r}")
    if cert.get("observable") != FINITE_GAUGE_REPORT_KIND:
        errors.append(f"observable must be {FINITE_GAUGE_REPORT_KIND!r}")
    if cert.get("continuum_claim", True):
        errors.append("continuum_claim must be False")
    if cert.get("g1_target_5x") is not False:
        errors.append("g1_target_5x must be False")
    if cert.get("g1_ge_generic") is not True:
        errors.append("g1_ge_generic must be True")
    if cert.get("haar_weyl_prefactor_24") is not True:
        errors.append("haar_weyl_prefactor_24 must be True")
    if cert.get("haar_su3_dim_3_0") is not True:
        errors.append("haar_su3_dim_3_0 must be True")
    if cert.get("strip_rp") is not True:
        errors.append("strip_rp must be True")
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
    for key in ("wilson_gap", "hamiltonian_gap", "three_plaquette_gap", "g1_factor"):
        value = _as_float(cert.get(key))
        if value is None or value <= 0.0:
            errors.append(f"{key} must be > 0")
    su3_gap = _as_float(cert.get("su3_gap"))
    if su3_gap is None or su3_gap <= 0.0:
        errors.append("su3_gap must be > 0")
    su3_dim = cert.get("su3_dimension")
    if not isinstance(su3_dim, int) or isinstance(su3_dim, bool) or su3_dim < 4:
        errors.append("su3_dimension must be an integer >= 4")
    gaps = cert.get("polymer_gaps")
    if not isinstance(gaps, list) or not gaps or any(
        not isinstance(item, int | float) or isinstance(item, bool) or item <= 0.0
        for item in gaps
    ):
        errors.append("polymer_gaps must be a non-empty list of positive floats")
    scaling = cert.get("scaling_gaps")
    if not isinstance(scaling, list) or len(scaling) < 3 or any(
        not isinstance(item, int | float) or isinstance(item, bool) or item <= 0.0
        for item in scaling
    ):
        errors.append("scaling_gaps must list at least three positive floats")
    star = _as_fraction(cert.get("domain_beta_certified"))
    outside = _as_fraction(cert.get("domain_beta_outside"))
    if star is None or outside is None or not (star < outside):
        errors.append("domain_beta_certified must be a Fraction pair strictly below domain_beta_outside")
    wilson_star = _as_fraction(cert.get("wilson_domain_beta_certified"))
    if wilson_star is None:
        errors.append("wilson_domain_beta_certified must be a Fraction pair")
    if cert.get("wilson_domain_grid_exhausted") is True:
        if cert.get("wilson_domain_beta_outside") is not None:
            errors.append("wilson_domain_beta_outside must be null when the Wilson grid is exhausted")
    elif cert.get("wilson_domain_grid_exhausted") is False:
        wilson_out = _as_fraction(cert.get("wilson_domain_beta_outside"))
        if wilson_star is None or wilson_out is None or not (wilson_star < wilson_out):
            errors.append(
                "wilson_domain_beta_certified must sit strictly below wilson_domain_beta_outside"
            )
    else:
        errors.append("wilson_domain_grid_exhausted must be a bool")
    if "digest" in cert and not verify_certificate_digest(cert):
        errors.append("digest does not match the certificate body (tampered/stale)")
    return errors


def replay_finite_gauge_report(cert: Certificate) -> bool | None:
    """Independently re-run the named spec and refuse a tighter sealed bundle."""
    from omnibias.geometry.gauge.transfer.report import (
        finite_gauge_report,
        finite_gauge_spec_from_mapping,
    )

    raw_spec = cert.get("spec")
    if not isinstance(raw_spec, Mapping):
        return None
    try:
        spec = finite_gauge_spec_from_mapping(raw_spec)
        fresh = finite_gauge_report(spec)
    except (ValueError, TypeError, KeyError):
        return False
    if not fresh.certified:
        return False
    tolerance = 1e-9

    def _not_tighter(sealed: Any, actual: float) -> bool:
        value = _as_float(sealed)
        return value is not None and not value > actual + tolerance

    if not _not_tighter(cert.get("wilson_gap"), fresh.wilson_character.spectral_gap_lower):
        return False
    if cert.get("su3_dimension") != fresh.su3_gap.dimension:
        return False
    if not _not_tighter(cert.get("su3_gap"), fresh.su3_gap.spectral_gap_lower):
        return False
    if not _not_tighter(cert.get("hamiltonian_gap"), fresh.hamiltonian.spectral_gap_lower):
        return False
    if not _not_tighter(
        cert.get("three_plaquette_gap"), fresh.three_plaquette.spectral_gap_lower
    ):
        return False
    sealed_gaps = cert.get("polymer_gaps")
    if not isinstance(sealed_gaps, list) or len(sealed_gaps) != len(fresh.polymer):
        return False
    for claimed, item in zip(sealed_gaps, fresh.polymer, strict=True):
        if not _not_tighter(claimed, item.spectral_gap_lower):
            return False
    sealed_scaling = cert.get("scaling_gaps")
    if not isinstance(sealed_scaling, list) or len(sealed_scaling) != len(fresh.scaling.points):
        return False
    for claimed, point in zip(sealed_scaling, fresh.scaling.points, strict=True):
        if not _not_tighter(claimed, point.spectral_gap_lower):
            return False
    sealed_star = _as_fraction(cert.get("domain_beta_certified"))
    if sealed_star is None or sealed_star > fresh.polymer_domain.beta_certified:
        return False
    wilson_star = _as_fraction(cert.get("wilson_domain_beta_certified"))
    if wilson_star is None or wilson_star > fresh.wilson_character_domain.beta_certified:
        return False
    if cert.get("g1_ge_generic") is True and not fresh.g1.ge_generic:
        return False
    if cert.get("haar_weyl_prefactor_24") is True and not fresh.haar.weyl_prefactor_24:
        return False
    if cert.get("haar_su3_dim_3_0") is True and not fresh.haar.su3_dim_3_0:
        return False
    return True


__all__ = [
    "FINITE_GAUGE_REPORT_KIND",
    "FINITE_GAUGE_REPORT_SCHEMA_VERSION",
    "HAMILTONIAN_GAP_KIND",
    "HAMILTONIAN_GAP_SCHEMA_VERSION",
    "POLYMER_DOMAIN_KIND",
    "POLYMER_DOMAIN_SCHEMA_VERSION",
    "STRIP_RP_KIND",
    "STRIP_RP_SCHEMA_VERSION",
    "TORUS_RP_KIND",
    "TORUS_RP_SCHEMA_VERSION",
    "THREE_PLAQUETTE_GAP_KIND",
    "STRONG_COUPLING_KIND",
    "STRONG_COUPLING_SCHEMA_VERSION",
    "TRANSFER_GAP_KIND",
    "TRANSFER_GAP_SCHEMA_VERSION",
    "WILSON_CHARACTER_DOMAIN_KIND",
    "WILSON_CHARACTER_DOMAIN_SCHEMA_VERSION",
    "finite_gauge_report_schema_errors",
    "hamiltonian_gap_schema_errors",
    "polymer_domain_schema_errors",
    "replay_finite_gauge_report",
    "replay_hamiltonian_gap",
    "replay_polymer_domain",
    "replay_strong_coupling_gap",
    "replay_strip_rp",
    "replay_transfer_matrix_gap",
    "replay_wilson_character_domain",
    "seal_finite_gauge_report_certificate",
    "seal_hamiltonian_gap_certificate",
    "seal_polymer_domain_certificate",
    "seal_strong_coupling_certificate",
    "seal_strip_rp_certificate",
    "seal_transfer_gap_certificate",
    "seal_wilson_character_domain_certificate",
    "strip_rp_schema_errors",
    "strong_coupling_schema_errors",
    "transfer_gap_schema_errors",
    "wilson_character_domain_schema_errors",
]
