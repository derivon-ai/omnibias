# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The default omnibias prove/disprove machine.

Registers the concrete certificate generators of this repository behind the
backend-agnostic :class:`omnibias.core.proof.ProofMachine` front door.  Because
:mod:`omnibias.core` may not import a downstream package, the wiring lives here
(in ``omnibias-pinn``, which *can* import the core).

Each prover maps a :class:`~omnibias.core.proof.Conjecture` of a given ``kind``
to a :class:`~omnibias.core.proof.ProofAttempt`:

============================  ===================================================
kind                          certificate
============================  ===================================================
``clm_blowup``                :func:`.certified_clm_blowup` (origin only)
``clm_multizero_blowup``      :func:`.certified_clm_multizero_first_blowup`
``ccf_selfsimilar_blowup``    :func:`.certified_ccf_selfsimilar_blowup_attempt`
``ccf_hardy_wholeline_blowup`` :func:`omnibias.pinn.certified.ccf_hardy.certified_ccf_hardy_wholeline_blowup_attempt`
``ccf_fractional_dissipation`` :func:`omnibias.pinn.certified.dissipation_threshold.certified_fractional_dissipation_threshold`
``gclm_selfsimilar_blowup``   :func:`.certified_gclm_selfsimilar_blowup`
``gclm_gradient_amplification`` :func:`.certified_gclm_gradient_amplification`
``perron_spectral_gap``       :func:`omnibias.core.verified.eig.certified_perron_spectral_gap`
``pinn_aposteriori_error``    :func:`omnibias.core.verified.pde_certificate.aposteriori_error_certificate`
``navier_stokes_periodic_residual`` :func:`.certified_periodic_flow_residual`
``navier_stokes_streamfunction_residual`` :func:`omnibias.pinn.certified.fluid_rigorous.certified_streamfunction_residual`
``navier_stokes_rollout_diagnostics`` :func:`omnibias.pinn.certified.fluid_rollout.certified_rollout_diagnostics`
============================  ===================================================

The machine still applies the schema gate, the independent replay (where a twin
exists), and the honesty gate, so a forged ``unproven_claim=True`` conjecture is
always downgraded to ``BLOCKED``.

Formal (Lean-kernel) gate
-------------------------
The Birkhoff-Hopf / Perron certificate carries a *rational* subdominant-ratio
upper bound, i.e. a **finite, kernel-checkable** obligation.  It is therefore
sealed (:func:`omnibias.core.proof.seal_certificate`): ``check_certificate``
refuses an unsealed payload before emitting any Lean, so a certificate that
carries an obligation but no digest could never reach the kernel at all.
Evaluating such a conjecture with ``machine.evaluate(conj, lean_check=True)`` --
or asserting the ``theorem_prover_verified`` claim -- routes the certificate
through :func:`omnibias.core.proof.lean_check.check_certificate`, which emits a
Lean proof and runs the ``formal/omnibias-verified-kernel`` Lake build.  The
:attr:`~omnibias.core.proof.Verdict.theorem_prover_verified` flag is set only on
a genuine kernel pass; asserting the claim without one downgrades ``PROVED`` to
``BLOCKED``.  When no Lean toolchain is present the check degrades gracefully
(the flag stays ``False`` and an unclaimed verdict is unaffected).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from omnibias.core.proof import (
    Certificate,
    Conjecture,
    FunctionProver,
    ProofAttempt,
    ProofMachine,
    seal_certificate,
    verify_certificate_digest,
)
from omnibias.core.verified.eig import certified_perron_spectral_gap
from omnibias.pinn.certified.ccf_hardy import (
    certified_ccf_hardy_wholeline_blowup_attempt,
    certified_ccf_hardy_wholeline_blowup_attempt_schema_errors,
)
from omnibias.pinn.certified.dissipation_threshold import (
    certified_fractional_dissipation_threshold,
    verify_fractional_dissipation_threshold,
)
from omnibias.pinn.certified.fluid import (
    navier_stokes_periodic_residual_schema_errors,
    prove_navier_stokes_periodic_residual,
    replay_navier_stokes_periodic_residual,
)
from omnibias.pinn.certified.fluid_rigorous import (
    prove_streamfunction_residual,
    replay_streamfunction_residual,
    streamfunction_residual_proof_schema_errors,
)
from omnibias.pinn.certified.fluid_rollout import (
    prove_rollout_diagnostics,
    replay_rollout_diagnostics,
    rollout_diagnostics_schema_errors,
)
from omnibias.pinn.certified.navier_stokes import (
    certified_ccf_selfsimilar_blowup_attempt,
    certified_ccf_selfsimilar_blowup_attempt_schema_errors,
    certified_clm_blowup,
    certified_clm_blowup_schema_errors,
    certified_clm_multizero_first_blowup,
    certified_clm_multizero_first_blowup_schema_errors,
    certified_gclm_gradient_amplification,
    certified_gclm_gradient_amplification_schema_errors,
    certified_gclm_selfsimilar_blowup,
    certified_gclm_selfsimilar_blowup_schema_errors,
    refine_ccf_selfsimilar_profile,
)
from omnibias.pinn.certified.pde import (
    pinn_aposteriori_proof_schema_errors,
    prove_pinn_aposteriori,
    replay_pinn_aposteriori,
)

PERRON_GAP_SCHEMA_VERSION = "verified-perron-spectral-gap-1"


def _blocked(detail: str) -> ProofAttempt:
    return ProofAttempt(status="BLOCKED", certificate=None, obligations=(detail,), detail=detail)


# --------------------------------------------------------------------------- #
# CLM single-point (origin) blow-up
# --------------------------------------------------------------------------- #
def _prove_clm_blowup(conjecture: Conjecture) -> ProofAttempt:
    data = conjecture.data
    try:
        cert = certified_clm_blowup(
            coeffs=list(data["coeffs"]),
            scales=list(data["scales"]),
            nodes=data.get("nodes"),
        )
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        return _blocked(f"could not build CLM certificate: {exc}")
    if cert["singularity_certified"]:
        return ProofAttempt(status="PROVED", certificate=cert,
                            detail="origin satisfies the CLM criterion H omega0(0) > 0")
    return ProofAttempt(
        status="BLOCKED",
        certificate=cert,
        obligations=("H omega0(0) <= 0 at the origin; the single-point cert cannot "
                     "disprove blow-up (other zeros unchecked -- use clm_multizero_blowup)",),
        detail="origin criterion not met",
    )


# --------------------------------------------------------------------------- #
# CLM multi-zero earliest blow-up (can PROVE or DISPROVE)
# --------------------------------------------------------------------------- #
def _prove_clm_multizero(conjecture: Conjecture) -> ProofAttempt:
    data = conjecture.data
    try:
        cert = certified_clm_multizero_first_blowup(
            coeffs=list(data["coeffs"]),
            scales=list(data["scales"]),
            newton_iters=int(data.get("newton_iters", 80)),
        )
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        return _blocked(f"could not build CLM multi-zero certificate: {exc}")
    if cert["singularity_certified"]:
        return ProofAttempt(status="PROVED", certificate=cert,
                            detail="max H omega0 over all zeros is certified positive")
    # All zeros enumerated (completeness) AND the certified max Hilbert value is
    # strictly negative => no zero can satisfy H omega0 > 0 => no blow-up.
    if cert["completeness_certified"] and float(cert["hilbert_omega0_max"]["upper"]) < 0.0:
        return ProofAttempt(status="DISPROVED", certificate=cert,
                            detail="all zeros enumerated; max H omega0 < 0, so no "
                                   "finite-time CLM singularity exists for this datum")
    return ProofAttempt(
        status="BLOCKED",
        certificate=cert,
        obligations=("neither a positive H omega0 nor a certified-negative maximum; "
                     "completeness or sign could not be certified",),
        detail="blow-up neither certified nor excluded",
    )


def _replay_clm_multizero(certificate: Certificate) -> bool | None:
    """Independent numpy twin (lazy import; ``None`` if omnibias-symbolic absent)."""
    try:
        from omnibias.symbolic.navier_stokes import verify_clm_multizero_first_blowup
    except ImportError:
        return None
    report = verify_clm_multizero_first_blowup(certificate)
    return bool(report["replay_match"])


def _prove_viscous_perturbation_enclosure(conjecture: Conjecture) -> ProofAttempt:
    data = conjecture.data
    try:
        from omnibias.pinn.certified.viscous_perturbation import (
            verify_viscous_perturbation_enclosure,
            viscous_perturbation_enclosure,
        )
        cert = viscous_perturbation_enclosure(
            inviscid_residual_sup=float(data["inviscid_residual_sup"]),
            viscosity=float(data["viscosity"]),
            enstrophy_bound=float(data["enstrophy_bound"]),
            window_length=float(data["window_length"]),
            tol=float(data.get("tol", 1e-2)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        return _blocked(f"could not build viscous perturbation enclosure: {exc}")
    report = verify_viscous_perturbation_enclosure(cert)
    if not report["replay_match"]:
        return _blocked("viscous enclosure independent recomputation failed")
    if cert["enclosure_closed"]:
        return ProofAttempt(
            status="PROVED",
            certificate=cert,
            detail=(
                "compact-window viscous residual upper bound closed "
                "(not a continuum Navier-Stokes theorem)"
            ),
        )
    return ProofAttempt(
        status="BLOCKED",
        certificate=cert,
        obligations=("ns_residual_upper_bound exceeds tol on the compact window",),
        detail="viscous perturbation enclosure did not close",
    )


def _prove_ccf_line_compactified_cap(conjecture: Conjecture) -> ProofAttempt:
    """Schema + honesty gate for a float line/compactified CCF CAP bundle.

    This does **not** claim radii-polynomial closure or Navier-Stokes. It only
    admits a well-formed CAP bundle whose honesty flags forbid continuum claims
    and whose residual samples agree with the independent symbolic twin.
    """
    data = conjecture.data
    try:
        from omnibias.pinn.jax.discovery.cap import cap_schema_errors
        from omnibias.symbolic.ccf import verify_cap_bundle
    except ImportError as exc:
        return _blocked(f"line CAP dependencies unavailable: {exc}")

    bundle = data.get("certificate") or data.get("bundle")
    if not isinstance(bundle, dict):
        return _blocked("conjecture.data must include 'certificate' or 'bundle' dict")
    errors = cap_schema_errors(bundle)
    if errors:
        return _blocked("CAP schema errors: " + "; ".join(errors))
    honesty = bundle.get("honesty", {})
    if honesty.get("navier_stokes_proof_claim", True) is not False:
        return _blocked("honesty.navier_stokes_proof_claim must be False")
    if honesty.get("continuum_navier_stokes_claim", False) is True:
        return _blocked("honesty.continuum_navier_stokes_claim must not be True")
    if bundle.get("domain", {}).get("type") != "line_compactified":
        return _blocked("domain.type must be 'line_compactified'")
    report = verify_cap_bundle(bundle)
    if not report.get("residual_samples_match"):
        return ProofAttempt(
            status="BLOCKED",
            certificate=bundle,
            obligations=("independent symbolic residual replay failed",),
            detail=f"agreement_max_abs_diff={report.get('agreement_max_abs_diff')}",
        )
    return ProofAttempt(
        status="PROVED",
        certificate=bundle,
        detail=(
            "line_compactified CAP schema-ok + symbolic residual replay match "
            "(float evidence; not a continuum NS theorem)"
        ),
    )


def _replay_ccf_line_compactified_cap(certificate: Certificate) -> bool | None:
    try:
        from omnibias.symbolic.ccf import verify_cap_bundle
    except ImportError:
        return None
    report = verify_cap_bundle(certificate if isinstance(certificate, dict) else dict(certificate))
    return bool(report.get("residual_samples_match"))


# --------------------------------------------------------------------------- #
# Cordoba-Cordoba-Fontelos self-similar profile (radii-polynomial; may BLOCK)
# --------------------------------------------------------------------------- #
def _prove_ccf_selfsimilar(conjecture: Conjecture) -> ProofAttempt:
    data = conjecture.data
    try:
        coeffs = list(data["coeffs"])
        scales = list(data["scales"])
        lam = float(data["lam"])
        nodes = data.get("nodes")
        form = str(data.get("form", "transport"))
        velocity_sign = float(data.get("velocity_sign", 1.0))
        if data.get("refine", False):
            refined = refine_ccf_selfsimilar_profile(
                coeffs=coeffs, scales=scales, lam=lam, nodes=nodes,
                form=form, velocity_sign=velocity_sign,
            )
            coeffs, scales, lam, nodes = (
                refined["coeffs"], refined["scales"], refined["lam"], refined["nodes"],
            )
        cert = certified_ccf_selfsimilar_blowup_attempt(
            coeffs=coeffs, scales=scales, lam=lam, nodes=nodes,
            form=form, velocity_sign=velocity_sign,
        )
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        return _blocked(f"could not build CCF self-similar certificate: {exc}")
    if cert["closure_certified"]:
        return ProofAttempt(
            status="PROVED", certificate=cert,
            detail="radii-polynomial closed: a self-similar CCF profile exists near "
                   "the candidate (collocation-level; lambda enclosed two-sided)",
        )
    failed = cert["closure_report"].get("failed_inequality") or "closure inequality not met"
    return ProofAttempt(
        status="BLOCKED",
        certificate=cert,
        obligations=(f"radii-polynomial closure failed: {failed}",
                     "between-node and whole-line residual sup-norm remain open"),
        detail="self-similar profile not certified (closure gap quantified in closure_report)",
    )


def _replay_ccf_selfsimilar(certificate: Certificate) -> bool | None:
    """Independent numpy twin (lazy import; ``None`` if omnibias-symbolic absent)."""
    try:
        from omnibias.symbolic.ccf import verify_ccf_selfsimilar_blowup_attempt
    except ImportError:
        return None
    report = verify_ccf_selfsimilar_blowup_attempt(certificate)
    return bool(report["replay_match"])


def _prove_ccf_hardy_wholeline(conjecture: Conjecture) -> ProofAttempt:
    data = conjecture.data
    try:
        cert = certified_ccf_hardy_wholeline_blowup_attempt(
            coeffs=list(data["coeffs"]),
            scales=list(data["scales"]),
            lam=float(data["lam"]),
            nodes=data.get("nodes"),
            form=str(data.get("form", "transport")),
            velocity_sign=float(data.get("velocity_sign", 1.0)),
            residual_gate=float(data.get("residual_gate", 1e-11)),
        )
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        return _blocked(f"could not build Hardy whole-line CCF certificate: {exc}")
    # PROVED only when whole_line_certified (collocation close alone is not enough).
    if cert["closure_certified"] and cert["honesty"].get("whole_line_certified"):
        return ProofAttempt(
            status="PROVED",
            certificate=cert,
            detail="Hardy whole-line CAP closed (residual gate + ell1_nu radii)",
        )
    gap = cert["closure_report"].get("quantified_gap", {})
    return ProofAttempt(
        status="BLOCKED",
        certificate=cert,
        obligations=(
            f"whole-line CAP gap: residual_gap={gap.get('residual_gap')}",
            "collocation/sequence closure incomplete — see closure_report",
        ),
        detail="Hardy whole-line CAP not closed (gap quantified)",
    )


def _prove_ccf_fractional_dissipation(conjecture: Conjecture) -> ProofAttempt:
    data = conjecture.data
    try:
        cert = certified_fractional_dissipation_threshold(
            lambda_lo=float(data["lambda_lo"]),
            lambda_hi=float(data["lambda_hi"]),
            alpha_claimed=data.get("alpha_claimed"),
        )
    except (ValueError, KeyError) as exc:
        return _blocked(f"could not build dissipation threshold: {exc}")
    replay = verify_fractional_dissipation_threshold(cert)
    if cert["threshold_closed"] and replay["replay_match"]:
        return ProofAttempt(
            status="PROVED",
            certificate=cert,
            detail="alpha_crit >= 1/(1+lambda_hi) margin sealed",
        )
    return ProofAttempt(
        status="BLOCKED",
        certificate=cert,
        obligations=("dissipation threshold margin not non-negative",),
        detail="fractional dissipation threshold not closed",
    )


# --------------------------------------------------------------------------- #
# generalized CLM (OSW) self-similar blow-up
# --------------------------------------------------------------------------- #
def _prove_gclm_selfsimilar(conjecture: Conjecture) -> ProofAttempt:
    data = conjecture.data
    try:
        cert = certified_gclm_selfsimilar_blowup(
            a=float(data.get("a", 0.5)),
            nodes=data.get("nodes"),
        )
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        return _blocked(f"could not build gCLM self-similar certificate: {exc}")
    if cert["blowup_certified"]:
        return ProofAttempt(status="PROVED", certificate=cert,
                            detail="exact self-similar profile identity satisfied")
    return ProofAttempt(
        status="BLOCKED",
        certificate=cert,
        obligations=("self-similar profile identity not exactly satisfied for this a",),
        detail="profile identity not certified",
    )


# --------------------------------------------------------------------------- #
# generalized CLM stagnation-point gradient amplification
# --------------------------------------------------------------------------- #
def _prove_gclm_gradient_amplification(conjecture: Conjecture) -> ProofAttempt:
    data = conjecture.data
    try:
        cert = certified_gclm_gradient_amplification(
            a=float(data["a"]),
            coeffs=list(data["coeffs"]),
            scales=list(data["scales"]),
        )
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        return _blocked(f"could not build gCLM gradient-amplification certificate: {exc}")
    if cert["instantaneous_amplification_certified"]:
        return ProofAttempt(status="PROVED", certificate=cert,
                            detail="instantaneous gradient amplification rate certified positive")
    return ProofAttempt(
        status="BLOCKED",
        certificate=cert,
        obligations=("amplification rate not certified positive at the stagnation point",),
        detail="amplification not certified",
    )


# --------------------------------------------------------------------------- #
# Birkhoff-Hopf certified spectral gap (omnibias.core.verified.eig)
# --------------------------------------------------------------------------- #
def _perron_certificate(matrix: Sequence[Sequence[Any]], lattice_spacing: float) -> Certificate:
    result = certified_perron_spectral_gap(matrix, lattice_spacing=lattice_spacing)
    # Sealed, because the rational ``subdominant_ratio_upper`` below is a finite
    # Lean-checkable obligation and ``check_certificate`` refuses an unsealed
    # payload before emitting any Lean.  Without the seal the kernel could never
    # be reached and ``theorem_prover_verified`` would be unreachable by
    # construction.  ``seal_certificate`` only ``setdefault``s ``schema_version``,
    # so the domain version below survives.
    return seal_certificate(
        {
            "schema_version": PERRON_GAP_SCHEMA_VERSION,
            "observable": "birkhoff_hopf_spectral_gap",
            "model": "fixed_positive_transfer_matrix",
            "method": "birkhoff_hopf_projective_contraction",
            "dimension": int(result.dimension),
            "min_entry": float(result.min_entry),
            "kappa_upper": float(result.kappa_upper),
            "subdominant_ratio_upper": float(result.subdominant_ratio_upper),
            "spectral_gap_lower": float(result.spectral_gap_lower),
            "spectral_gap_lower_per_unit": float(result.spectral_gap_lower_per_unit),
            "lattice_spacing": float(lattice_spacing),
            "continuum_claim": False,
            "honesty": {
                "unproven_claim": False,
                "continuum_claim": False,
                "fixed_matrix": True,
                "interval_verified": True,
                "note": (
                    "fixed-matrix (fixed spacing / finite volume) Birkhoff-Hopf gap; "
                    "NOT a continuum / uniform-limit or global-regularity claim"
                ),
            },
        }
    )


def perron_spectral_gap_schema_errors(cert: Certificate) -> list[str]:
    """Validate a ``verified-perron-spectral-gap-1`` certificate."""
    errors: list[str] = []
    required = (
        "schema_version",
        "observable",
        "model",
        "dimension",
        "subdominant_ratio_upper",
        "spectral_gap_lower",
        "lattice_spacing",
        "continuum_claim",
        "honesty",
        "digest",
    )
    for key in required:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    if cert.get("schema_version") != PERRON_GAP_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PERRON_GAP_SCHEMA_VERSION!r}")
    if cert.get("continuum_claim", True):
        errors.append("continuum_claim must be False")
    honesty = cert.get("honesty", {})
    if honesty.get("unproven_claim", False):
        errors.append("honesty.unproven_claim must be False")
    if float(cert.get("spectral_gap_lower", -1.0)) < 0.0:
        errors.append("spectral_gap_lower must be >= 0")
    if "digest" in cert and not verify_certificate_digest(cert):
        errors.append("digest does not match the certificate body (tampered/stale)")
    return errors


def _prove_perron_spectral_gap(conjecture: Conjecture) -> ProofAttempt:
    data = conjecture.data
    try:
        cert = _perron_certificate(
            data["matrix"], float(data.get("lattice_spacing", 1.0))
        )
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        return _blocked(f"could not build Perron spectral-gap certificate: {exc}")
    if float(cert["spectral_gap_lower"]) > 0.0:
        return ProofAttempt(status="PROVED", certificate=cert,
                            detail="Birkhoff-Hopf certifies a strictly positive spectral gap")
    return ProofAttempt(
        status="BLOCKED",
        certificate=cert,
        obligations=("no strictly positive spectral gap could be certified",),
        detail="gap not certified positive",
    )


def default_provers() -> list[FunctionProver]:
    """The built-in provers registered by :func:`build_default_machine`."""
    return [
        FunctionProver(
            name="clm_blowup",
            kinds=frozenset({"clm_blowup"}),
            prove_fn=_prove_clm_blowup,
            schema_fn=certified_clm_blowup_schema_errors,
        ),
        FunctionProver(
            name="clm_multizero_blowup",
            kinds=frozenset({"clm_multizero_blowup"}),
            prove_fn=_prove_clm_multizero,
            schema_fn=certified_clm_multizero_first_blowup_schema_errors,
            replay_fn=_replay_clm_multizero,
        ),
        FunctionProver(
            name="ccf_selfsimilar_blowup",
            kinds=frozenset({"ccf_selfsimilar_blowup"}),
            prove_fn=_prove_ccf_selfsimilar,
            schema_fn=certified_ccf_selfsimilar_blowup_attempt_schema_errors,
            replay_fn=_replay_ccf_selfsimilar,
        ),
        FunctionProver(
            name="ccf_hardy_wholeline_blowup",
            kinds=frozenset({"ccf_hardy_wholeline_blowup"}),
            prove_fn=_prove_ccf_hardy_wholeline,
            schema_fn=certified_ccf_hardy_wholeline_blowup_attempt_schema_errors,
        ),
        FunctionProver(
            name="ccf_fractional_dissipation",
            kinds=frozenset({"ccf_fractional_dissipation"}),
            prove_fn=_prove_ccf_fractional_dissipation,
        ),
        FunctionProver(
            name="ccf_line_compactified_cap",
            kinds=frozenset({"ccf_line_compactified_cap"}),
            prove_fn=_prove_ccf_line_compactified_cap,
            schema_fn=lambda cert: __import__(
                "omnibias.pinn.jax.discovery.cap", fromlist=["cap_schema_errors"]
            ).cap_schema_errors(cert),
            replay_fn=_replay_ccf_line_compactified_cap,
        ),
        FunctionProver(
            name="viscous_perturbation_enclosure",
            kinds=frozenset({"viscous_perturbation_enclosure"}),
            prove_fn=_prove_viscous_perturbation_enclosure,
        ),
        FunctionProver(
            name="gclm_selfsimilar_blowup",
            kinds=frozenset({"gclm_selfsimilar_blowup"}),
            prove_fn=_prove_gclm_selfsimilar,
            schema_fn=certified_gclm_selfsimilar_blowup_schema_errors,
        ),
        FunctionProver(
            name="gclm_gradient_amplification",
            kinds=frozenset({"gclm_gradient_amplification"}),
            prove_fn=_prove_gclm_gradient_amplification,
            schema_fn=certified_gclm_gradient_amplification_schema_errors,
        ),
        FunctionProver(
            name="perron_spectral_gap",
            kinds=frozenset({"perron_spectral_gap"}),
            prove_fn=_prove_perron_spectral_gap,
            schema_fn=perron_spectral_gap_schema_errors,
        ),
        FunctionProver(
            name="pinn_aposteriori_error",
            kinds=frozenset({"pinn_aposteriori_error"}),
            prove_fn=prove_pinn_aposteriori,
            schema_fn=pinn_aposteriori_proof_schema_errors,
            replay_fn=replay_pinn_aposteriori,
        ),
        FunctionProver(
            name="navier_stokes_periodic_residual",
            kinds=frozenset({"navier_stokes_periodic_residual"}),
            prove_fn=prove_navier_stokes_periodic_residual,
            schema_fn=navier_stokes_periodic_residual_schema_errors,
            replay_fn=replay_navier_stokes_periodic_residual,
        ),
        FunctionProver(
            name="navier_stokes_streamfunction_residual",
            kinds=frozenset({"navier_stokes_streamfunction_residual"}),
            prove_fn=prove_streamfunction_residual,
            schema_fn=streamfunction_residual_proof_schema_errors,
            replay_fn=replay_streamfunction_residual,
        ),
        FunctionProver(
            name="navier_stokes_rollout_diagnostics",
            kinds=frozenset({"navier_stokes_rollout_diagnostics"}),
            prove_fn=prove_rollout_diagnostics,
            schema_fn=rollout_diagnostics_schema_errors,
            replay_fn=replay_rollout_diagnostics,
        ),
    ]


def build_default_machine() -> ProofMachine:
    """A :class:`ProofMachine` preloaded with every built-in omnibias prover."""
    machine = ProofMachine()
    for prover in default_provers():
        machine.register(prover)
    return machine


__all__ = [
    "PERRON_GAP_SCHEMA_VERSION",
    "build_default_machine",
    "default_provers",
    "navier_stokes_periodic_residual_schema_errors",
    "perron_spectral_gap_schema_errors",
    "pinn_aposteriori_proof_schema_errors",
    "streamfunction_residual_proof_schema_errors",
]
