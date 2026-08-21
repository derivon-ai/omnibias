# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Register the SOS provers behind the :class:`omnibias.core.proof.ProofMachine`.

The machine is the repo's common *prove / disprove* front door: a
:class:`~omnibias.core.proof.Conjecture` goes in and a
:class:`~omnibias.core.proof.Verdict` (``PROVED | BLOCKED``) comes out, with the
schema gate, an independent replay, the honesty gate, and the optional Lean-kernel
gate all applied uniformly.  This module wires the three SOS provers into it, so an
SOS / Positivstellensatz / auxiliary-functional result flows through exactly the
same gates as every other omnibias certificate:

===============================  ====================================================
kind                             prover
===============================  ====================================================
``sos_global_nonneg``            :func:`omnibias.sos.certify.certify_sos`
``sos_nonneg_on_set``            :func:`omnibias.sos.positivstellensatz.certify_nonneg_on_set`
``sos_time_average_bound``       :func:`omnibias.sos.auxiliary.certify_time_average_bound`
===============================  ====================================================

Each prover attaches the sealed v1 ``positive_definite`` certificate, so:

* the **schema / digest gate** validates the sealed envelope and its tamper-evident
  digest;
* the **replay** independently re-runs the rigorous interval ``LDL^T`` on the exact
  rational Gram stored in the certificate and confirms both positive definiteness
  and agreement with the sealed pivots;
* the **honesty gate** downgrades any conjecture that asserts an unsupported claim
  (every SOS certificate hard-wires ``unproven_claim = False``, so a forged unproven-result
  assertion is always ``BLOCKED``);
* the **formal gate** (``lean_check=True`` or an asserted
  ``theorem_prover_verified``) routes the ``allPivotsPos`` obligation to the
  Mathlib-free Lean kernel.

Because :mod:`omnibias.core` must not import a downstream package, this wiring lives
here in ``omnibias-sos`` (which may import the core), mirroring
``omnibias.pinn.certified.machine``.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from omnibias.core.proof import (
    CatalogEntry,
    Certificate,
    Conjecture,
    FunctionProver,
    ProofAttempt,
    ProofMachine,
    register_catalog,
    run_discovery,
    schema_errors_v1,
)
from omnibias.core.proof.certificate import decode_interval
from omnibias.core.verified.eig_operator import interval_ldlt_pivots
from omnibias.sos.auxiliary import certify_time_average_bound, seal_auxiliary_bound
from omnibias.sos.certify import certify_sos
from omnibias.sos.honesty import SOSScope, seal_sos_certificate
from omnibias.sos.positivstellensatz import (
    certify_nonneg_on_set,
    seal_positivstellensatz_certificate,
)

SOS_GLOBAL_NONNEG = "sos_global_nonneg"
SOS_NONNEG_ON_SET = "sos_nonneg_on_set"
SOS_TIME_AVERAGE_BOUND = "sos_time_average_bound"
SOS_DEGREE_SEARCH = "sos_degree_search"
CONDITION_SOS_TEMPLATE = "condition_sos_template"
CONDITION_SOS_ONSET = "condition_sos_onset"


def _blocked(detail: str) -> ProofAttempt:
    return ProofAttempt(status="BLOCKED", certificate=None, obligations=(detail,), detail=detail)


def _scope(data: Mapping[str, Any]) -> SOSScope | None:
    scope = data.get("scope")
    return scope if isinstance(scope, SOSScope) else None


# --------------------------------------------------------------------------- #
# provers
# --------------------------------------------------------------------------- #
def _prove_sos_global(conjecture: Conjecture) -> ProofAttempt:
    data = conjecture.data
    try:
        cert = certify_sos(data["polynomial"], half_degree=data.get("half_degree"))
    except (KeyError, ValueError) as exc:
        return _blocked(f"could not build SOS certificate: {exc}")
    if not cert.certified:
        return _blocked(f"SOS not certified: {cert.detail}")
    sealed = seal_sos_certificate(cert, claim=conjecture.name, scope=_scope(data))
    return ProofAttempt(status="PROVED", certificate=sealed, detail=cert.detail)


def _prove_nonneg_on_set(conjecture: Conjecture) -> ProofAttempt:
    data = conjecture.data
    try:
        cert = certify_nonneg_on_set(
            data["polynomial"], list(data.get("constraints", ())),
            half_degree=data.get("half_degree"),
        )
    except (KeyError, ValueError) as exc:
        return _blocked(f"could not build Positivstellensatz certificate: {exc}")
    if not cert.certified:
        return _blocked(f"constrained positivity not certified: {cert.detail}")
    sealed = seal_positivstellensatz_certificate(cert, claim=conjecture.name, scope=_scope(data))
    return ProofAttempt(status="PROVED", certificate=sealed, detail=cert.detail)


def _prove_time_average_bound(conjecture: Conjecture) -> ProofAttempt:
    data = conjecture.data
    kwargs: dict[str, Any] = {}
    for key in ("auxiliary_degree", "gram_half_degree", "free_ridge"):
        if key in data:
            kwargs[key] = data[key]
    try:
        cert = certify_time_average_bound(data["system"], data["observable"], **kwargs)
    except (KeyError, ValueError) as exc:
        return _blocked(f"could not build auxiliary-functional bound: {exc}")
    if not cert.certified:
        return _blocked(f"time-average bound not certified: {cert.detail}")
    sealed = seal_auxiliary_bound(cert, claim=conjecture.name, scope=_scope(data))
    return ProofAttempt(status="PROVED", certificate=sealed, detail=cert.detail)


# --------------------------------------------------------------------------- #
# schema + independent replay (shared by all three SOS certificate kinds)
# --------------------------------------------------------------------------- #
def sos_certificate_schema_errors(certificate: Certificate) -> list[str]:
    """Validate a sealed SOS ``positive_definite`` certificate (empty list == valid)."""
    errors: list[str] = [str(e) for e in schema_errors_v1(certificate)]
    payload = certificate.get("payload", {})
    if not isinstance(payload, Mapping) or payload.get("type") != "positive_definite":
        errors.append("payload.type must be 'positive_definite'")
    honesty = certificate.get("honesty", {})
    if not isinstance(honesty, Mapping) or honesty.get("unproven_claim", True):
        errors.append("honesty.unproven_claim must be False")
    return errors


def replay_sos_certificate(certificate: Certificate) -> bool | None:
    r"""Independently re-derive the PD fact from the exact rational Gram.

    Re-runs the rigorous interval ``LDL^T`` on the Gram stored in ``meta.sos.gram``
    and confirms (a) every recomputed pivot's lower endpoint is strictly positive
    (an *independent* proof of positive definiteness), (b) the sealed payload pivots
    are likewise all positive, and (c) each recomputed enclosure is **consistent**
    with the sealed one (they overlap -- two sound interval factorisations of the
    same matrix may differ in width but must enclose the same true pivot).  A
    tampered Gram breaks positive definiteness or consistency and is rejected.
    Returns ``None`` when no Gram is present (nothing to replay), never raising.
    """
    meta = certificate.get("meta", {})
    sos = meta.get("sos", {}) if isinstance(meta, Mapping) else {}
    gram_str = sos.get("gram") if isinstance(sos, Mapping) else None
    if not gram_str:
        return None
    try:
        gram = [[Fraction(entry) for entry in row] for row in gram_str]
    except (ValueError, TypeError):
        return False
    pivots = interval_ldlt_pivots(gram)
    if pivots is None or not all(p.lo > 0.0 for p in pivots):
        return False

    payload = certificate.get("payload", {})
    encoded = payload.get("pivots", []) if isinstance(payload, Mapping) else []
    if len(encoded) != len(pivots):
        return False
    for enc, piv in zip(encoded, pivots, strict=False):
        sealed = decode_interval(enc)
        if sealed.lo <= 0.0:
            return False
        # The two independent enclosures of the same pivot must overlap.
        if max(sealed.lo, piv.lo) > min(sealed.hi, piv.hi):
            return False
    return True


def _prove_degree_search(conjecture: Conjecture) -> ProofAttempt:
    from omnibias.sos.families import SosDegreeFamily

    family = SosDegreeFamily(max_half_degree=int(conjecture.data.get("max_half_degree", 2)))
    proposer = str(conjecture.data.get("proposer", "score_guided"))
    result = run_discovery(family.statement, family, proposer, budget=int(conjecture.data.get("budget", 4)))
    payload = result.as_dict()
    if result.status != "PROVED":
        return _blocked(result.detail)
    return ProofAttempt(status="PROVED", certificate=payload, detail=result.detail)


def _degree_schema(certificate: Certificate) -> list[str]:
    honesty = certificate.get("honesty", {})
    if not isinstance(honesty, Mapping):
        return ["honesty must be a mapping"]
    if honesty.get("unproven_claim"):
        return ["unproven_claim must be False"]
    return []


def sos_provers() -> list[FunctionProver]:
    """The SOS provers registered by :func:`build_sos_machine`."""
    return [
        FunctionProver(
            name="sos_global_nonneg",
            kinds=frozenset({SOS_GLOBAL_NONNEG}),
            prove_fn=_prove_sos_global,
            schema_fn=sos_certificate_schema_errors,
            replay_fn=replay_sos_certificate,
        ),
        FunctionProver(
            name="sos_nonneg_on_set",
            kinds=frozenset({SOS_NONNEG_ON_SET}),
            prove_fn=_prove_nonneg_on_set,
            schema_fn=sos_certificate_schema_errors,
            replay_fn=replay_sos_certificate,
        ),
        FunctionProver(
            name="sos_time_average_bound",
            kinds=frozenset({SOS_TIME_AVERAGE_BOUND}),
            prove_fn=_prove_time_average_bound,
            schema_fn=sos_certificate_schema_errors,
            replay_fn=replay_sos_certificate,
        ),
        FunctionProver(
            name=SOS_DEGREE_SEARCH,
            kinds=frozenset({SOS_DEGREE_SEARCH}),
            prove_fn=_prove_degree_search,
            schema_fn=_degree_schema,
            replay_fn=lambda c: bool(c.get("replay_ok")),
        ),
    ]


def build_sos_machine() -> ProofMachine:
    """A :class:`~omnibias.core.proof.ProofMachine` preloaded with the SOS provers."""
    machine = ProofMachine()
    for prover in sos_provers():
        machine.register(prover)
    return machine


def _register() -> None:
    from omnibias.sos.families import SosDegreeFamily

    for kind, obligation in (
        (SOS_GLOBAL_NONNEG, "a global SOS certificate for a named polynomial"),
        (SOS_NONNEG_ON_SET, "a Positivstellensatz certificate on a named set"),
        (SOS_TIME_AVERAGE_BOUND, "a time-average auxiliary-functional bound"),
    ):
        register_catalog(
            CatalogEntry(
                kind=kind,
                obligation=obligation,
                parent="sum-of-squares positivity",
                parent_status="already_true",
                package="omnibias.sos",
                mode="enclosure",
                complete=True,
            ),
            lambda kind=kind, **_k: {"kind": kind, "mode": "enclosure"},
        )
    register_catalog(
        CatalogEntry(
            kind=SOS_DEGREE_SEARCH,
            obligation="the planted polynomial is SOS at some half-degree in the box",
            parent="sum-of-squares positivity",
            parent_status="already_true",
            package="omnibias.sos",
            mode="exact_search",
            complete=True,
        ),
        lambda **kwargs: run_discovery(
            SosDegreeFamily().statement,
            SosDegreeFamily(),
            "score_guided",
            budget=int(kwargs.get("budget", 4)),
        ),
    )
    from omnibias.sos.conditions import SosOnsetFamily, SosTemplateFamily

    register_catalog(
        CatalogEntry(
            kind=CONDITION_SOS_TEMPLATE,
            obligation="the planted polynomial is SOS at some half-degree in this grammar",
            parent="sum-of-squares positivity",
            parent_status="already_true",
            package="omnibias.sos",
            mode="exact_search",
            complete=False,
        ),
        lambda **kwargs: run_discovery(
            SosTemplateFamily().statement,
            SosTemplateFamily(),
            "score_guided",
            budget=int(kwargs.get("budget", 4)),
        ),
    )
    register_catalog(
        CatalogEntry(
            kind=CONDITION_SOS_ONSET,
            obligation="the observed polynomial is nonnegative on the packed constraint set",
            parent="on-set nonnegativity",
            parent_status="already_true",
            package="omnibias.sos",
            mode="enclosure",
            complete=False,
        ),
        lambda **kwargs: run_discovery(
            SosOnsetFamily().statement,
            SosOnsetFamily(),
            "score_guided",
            budget=int(kwargs.get("budget", 4)),
        ),
    )


_register()


__all__ = [
    "CONDITION_SOS_ONSET",
    "CONDITION_SOS_TEMPLATE",
    "SOS_DEGREE_SEARCH",
    "SOS_GLOBAL_NONNEG",
    "SOS_NONNEG_ON_SET",
    "SOS_TIME_AVERAGE_BOUND",
    "build_sos_machine",
    "replay_sos_certificate",
    "sos_certificate_schema_errors",
    "sos_provers",
]
