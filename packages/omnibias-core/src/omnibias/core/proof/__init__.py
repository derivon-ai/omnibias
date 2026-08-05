# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""A backend-agnostic *prove / disprove* orchestration engine.

omnibias already produces many independently checkable **certificates** -- the
Constantin-Lax-Majda / generalised-CLM blow-up certificates, the multi-zero
earliest-blow-up theorem, the Birkhoff-Hopf spectral-gap certificate, and so on.
Each lives in its own module with its own schema validator and, often, an
independent numpy/mpmath replay twin.  This subpackage gives them **one common
front door**: a :class:`Conjecture` goes in, a :class:`Verdict`
(``PROVED | DISPROVED | BLOCKED``) comes out, with the schema gate, the
independent replay, and an *honesty gate* all enforced uniformly.

The engine is intentionally pure-Python and dependency-free (no ``torch`` /
``jax`` / ``numpy`` / ``omnibias.pinn`` imports): concrete provers register
themselves from their own packages, so :mod:`omnibias.core` never imports a
downstream package.  The default registry of provers is assembled in
:func:`omnibias.pinn.certified.build_default_machine` (which *can* import the
core).

Honesty gate
------------
A :class:`Conjecture` may *assert* honesty claims (e.g. ``unproven_claim=True``).
The machine only lets a ``PROVED`` verdict stand if every asserted claim is
actually backed by the certificate's own ``honesty`` flags.  Because every
omnibias model certificate hard-wires ``unproven_claim=False``, asserting an unproven
claim is **always** downgraded to ``BLOCKED`` unless an external verifier has
attached genuine evidence -- exactly mirroring the per-certificate schema gates
(e.g. :func:`certified_clm_blowup_schema_errors`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from omnibias.core.proof.certificate import (
    CERTIFICATE_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    THEOREM_PROVER_VERIFIED_KEY,
    canonical_json,
    certificate_digest,
    decode_interval,
    decode_interval_matrix,
    decode_taylor_model,
    encode_interval,
    encode_interval_matrix,
    encode_taylor_model,
    interval_certificate,
    make_certificate,
    positive_definite_certificate,
    schema_errors_v1,
    seal_certificate,
    taylor_model_certificate,
    verify_certificate_digest,
)
from omnibias.core.proof.lean_check import (
    LeanCheckResult,
    check_certificate,
    generate_obligation,
    kernel_root,
    lean_check_available,
)

#: A certificate is a plain JSON-serialisable mapping (the omnibias convention).
Certificate = dict[str, Any]

#: The three terminal verdicts.
VerdictStatus = Literal["PROVED", "DISPROVED", "BLOCKED"]

#: A reserved honesty claim that is *earned by the Lean kernel*, not asserted by a
#: certificate's own ``honesty`` flags. The generic :func:`honesty_gate` ignores
#: it; :meth:`ProofMachine.evaluate` adjudicates it against an actual kernel pass.
#: Alias of :data:`~omnibias.core.proof.certificate.THEOREM_PROVER_VERIFIED_KEY`.
FORMAL_CLAIM_KEY = THEOREM_PROVER_VERIFIED_KEY


@dataclass(frozen=True)
class Conjecture:
    """A statement to be adjudicated by the :class:`ProofMachine`.

    Parameters
    ----------
    name:
        Human-readable label (free text), e.g. ``"CLM blow-up for omega0=-q_1"``.
    kind:
        The dispatch key a prover matches on, e.g. ``"clm_multizero_blowup"``.
    data:
        Inputs forwarded to the prover (e.g. ``{"coeffs": [...], "scales": [...]}``).
    claims:
        Honesty claims the conjecture *asserts*.  Each ``True`` entry must be
        supported by the produced certificate's ``honesty`` flags or the verdict
        is downgraded to ``BLOCKED`` (the honesty gate).  Leave empty for the
        normal "is this model statement certified?" question.
    """

    name: str
    kind: str
    data: Mapping[str, Any] = field(default_factory=dict)
    claims: Mapping[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ProofAttempt:
    """What a prover returns: its intrinsic verdict plus any artifact.

    The machine still applies the schema gate, the replay check, and the honesty
    gate on top of this, so a prover may return ``PROVED`` and still be
    downgraded to ``BLOCKED`` by the machine if those checks fail.
    """

    status: VerdictStatus
    certificate: Certificate | None = None
    obligations: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class Verdict:
    """The machine's adjudication of a :class:`Conjecture`."""

    status: VerdictStatus
    conjecture: str
    kind: str
    prover: str
    certificate: Certificate | None
    obligations: tuple[str, ...]
    schema_ok: bool
    replay_ok: bool | None
    honesty_ok: bool
    detail: str = ""
    #: The ``schema_version`` carried by the certificate.
    certificate_schema_version: str | None = None
    #: ``True`` only when a formal theorem-prover kernel re-checked the
    #: certificate (set by the Lean bridge); ``False`` otherwise.
    theorem_prover_verified: bool = False

    @property
    def proved(self) -> bool:
        return self.status == "PROVED"

    @property
    def disproved(self) -> bool:
        return self.status == "DISPROVED"

    @property
    def blocked(self) -> bool:
        return self.status == "BLOCKED"

    def summary(self) -> dict[str, Any]:
        """A small JSON-serialisable digest (omits the full certificate)."""
        return {
            "status": self.status,
            "conjecture": self.conjecture,
            "kind": self.kind,
            "prover": self.prover,
            "schema_ok": self.schema_ok,
            "replay_ok": self.replay_ok,
            "honesty_ok": self.honesty_ok,
            "obligations": list(self.obligations),
            "detail": self.detail,
            "certificate_schema_version": self.certificate_schema_version,
            "theorem_prover_verified": self.theorem_prover_verified,
        }


class Prover(Protocol):
    """Structural protocol every prover satisfies.

    ``handles`` selects on :attr:`Conjecture.kind`; ``attempt`` produces a
    :class:`ProofAttempt`; ``schema_errors`` validates a certificate (empty list
    == valid); ``replay`` runs an *independent* recomputation and returns
    ``True``/``False`` (or ``None`` when no twin exists).
    """

    name: str

    def handles(self, conjecture: Conjecture) -> bool: ...

    def attempt(self, conjecture: Conjecture) -> ProofAttempt: ...

    def schema_errors(self, certificate: Certificate) -> list[str]: ...

    def replay(self, certificate: Certificate) -> bool | None: ...


@dataclass(frozen=True)
class FunctionProver:
    """Adapter that builds a :class:`Prover` from plain callables.

    Lets a downstream package register an existing certificate generator without
    writing a class: supply the kinds it handles, a function mapping a
    :class:`Conjecture` to a :class:`ProofAttempt`, and (optionally) the schema
    validator and the independent replay twin.
    """

    name: str
    kinds: frozenset[str]
    prove_fn: Callable[[Conjecture], ProofAttempt]
    schema_fn: Callable[[Certificate], list[str]] | None = None
    replay_fn: Callable[[Certificate], bool | None] | None = None

    def handles(self, conjecture: Conjecture) -> bool:
        return conjecture.kind in self.kinds

    def attempt(self, conjecture: Conjecture) -> ProofAttempt:
        return self.prove_fn(conjecture)

    def schema_errors(self, certificate: Certificate) -> list[str]:
        return [] if self.schema_fn is None else list(self.schema_fn(certificate))

    def replay(self, certificate: Certificate) -> bool | None:
        return None if self.replay_fn is None else self.replay_fn(certificate)


class ProverRegistry:
    """An ordered collection of provers with kind-based selection."""

    def __init__(self, provers: Iterable[Prover] | None = None) -> None:
        self._provers: list[Prover] = list(provers) if provers is not None else []

    def register(self, prover: Prover) -> ProverRegistry:
        """Append ``prover`` (returns ``self`` for chaining)."""
        self._provers.append(prover)
        return self

    def extend(self, provers: Iterable[Prover]) -> ProverRegistry:
        for prover in provers:
            self._provers.append(prover)
        return self

    @property
    def provers(self) -> tuple[Prover, ...]:
        return tuple(self._provers)

    def kinds(self) -> tuple[str, ...]:
        """All conjecture kinds handled by at least one registered prover."""
        seen: dict[str, None] = {}
        for prover in self._provers:
            kinds = getattr(prover, "kinds", None)
            if kinds is not None:
                for kind in kinds:
                    seen[str(kind)] = None
        return tuple(seen)

    def select(self, conjecture: Conjecture) -> Prover | None:
        """The first registered prover that handles ``conjecture`` (or ``None``)."""
        for prover in self._provers:
            if prover.handles(conjecture):
                return prover
        return None


def honesty_gate(conjecture: Conjecture, certificate: Certificate | None) -> bool:
    """``True`` iff every asserted honesty claim is backed by the certificate.

    An asserted ``claims[key] == True`` requires ``certificate["honesty"][key]``
    to be truthy.  No assertions (the default) always passes.
    """
    if not conjecture.claims:
        return True
    honesty: Mapping[str, Any] = {}
    if certificate is not None:
        candidate = certificate.get("honesty", {})
        if isinstance(candidate, Mapping):
            honesty = candidate
    for key, asserted in conjecture.claims.items():
        if key == FORMAL_CLAIM_KEY:
            continue  # earned by the Lean kernel, not a certificate honesty flag
        if asserted and not bool(honesty.get(key, False)):
            return False
    return True


def formal_claim_forgery_errors(
    honesty: Mapping[str, Any] | None,
    *,
    certificate: Certificate | None = None,
    context: str = "honesty",
) -> list[str]:
    """Reject a *self-declared* ``theorem_prover_verified``.

    :data:`FORMAL_CLAIM_KEY` is **earned** by a genuine Lean kernel pass, never
    asserted by the artifact that benefits from it. :meth:`ProofMachine.evaluate`
    already enforces that for conjectures, but any consumer that reads the flag
    straight out of a *stored* bundle (a JSON file on disk, which anyone can edit)
    would trust the artifact's own word for it. That is the one shape in which the
    claim can be forged, so validators call this helper and treat a non-empty
    result as a schema error.

    A truthy declaration is accepted only when ``certificate`` is supplied *and*
    the kernel genuinely verifies its obligation. With no Lean toolchain present
    the bridge reports unverified, so the flag stays refused -- graceful
    degradation must never become silent acceptance.

    Parameters
    ----------
    honesty:
        The artifact's ``honesty`` mapping (``None`` and non-mappings are treated
        as declaring nothing).
    certificate:
        The certificate whose finite obligation the kernel should re-check. Omit
        it to refuse every truthy declaration outright.
    context:
        Dotted prefix used in the message, so the caller's schema report points at
        the offending field.

    Returns
    -------
    list[str]
        Empty when nothing is declared or the declaration is kernel-backed;
        otherwise a single explanatory error.
    """
    if not isinstance(honesty, Mapping) or not bool(honesty.get(FORMAL_CLAIM_KEY, False)):
        return []
    if certificate is not None:
        from omnibias.core.proof.lean_check import check_certificate

        if check_certificate(certificate).verified:
            return []
    return [
        f"{context}.{FORMAL_CLAIM_KEY} is declared true but no Lean kernel pass "
        "supports it; the flag is earned by a genuine `lake build`, never asserted "
        "by the artifact claiming it"
    ]


def kernel_earned_theorem_prover_verified(bundle: Mapping[str, Any]) -> bool:
    """Whether a Lean kernel pass backs ``bundle`` -- never trust honesty self-claims.

    Looks for a sealed v1 certificate under ``bundle["certificate"]``, or treats
    ``bundle`` itself as the certificate when it already carries a digest+payload.
    Calls :func:`~omnibias.core.proof.lean_check.check_certificate` in either case
    (degrades to ``False`` when Lean is absent or the digest is missing/mismatched).
    """
    cert: Mapping[str, Any] | None = None
    nested = bundle.get("certificate")
    if isinstance(nested, Mapping):
        cert = nested
    elif isinstance(bundle.get("digest"), str) and isinstance(bundle.get("payload"), Mapping):
        cert = bundle
    if cert is None:
        return False
    return bool(check_certificate(cert).verified)


class ProofMachine:
    """Dispatch a :class:`Conjecture` to a prover and adjudicate a :class:`Verdict`.

    Pipeline: select prover -> ``attempt`` -> schema gate -> independent replay
    -> honesty gate.  A prover's ``PROVED`` (or ``DISPROVED``) is downgraded to
    ``BLOCKED`` whenever the schema is invalid, an independent replay disagrees,
    or (for ``PROVED``) an asserted honesty claim is unsupported.
    """

    def __init__(self, registry: ProverRegistry | None = None) -> None:
        self.registry: ProverRegistry = registry if registry is not None else ProverRegistry()

    def register(self, prover: Prover) -> ProofMachine:
        self.registry.register(prover)
        return self

    @property
    def provers(self) -> tuple[Prover, ...]:
        return self.registry.provers

    def kinds(self) -> tuple[str, ...]:
        return self.registry.kinds()

    def evaluate(
        self,
        conjecture: Conjecture,
        *,
        replay: bool = True,
        lean_check: bool = False,
        strict: bool = False,
    ) -> Verdict:
        """Adjudicate a single conjecture.

        Two always-on gates run on any certificate (both non-breaking, because they
        only fire on certificates that opt into the relevant format):

        * **digest gate** -- if the certificate carries a ``digest`` it must match
          its body (:func:`~omnibias.core.proof.certificate.verify_certificate_digest`),
          so a tampered or stale sealed certificate is downgraded to ``BLOCKED``;
        * **v1 schema gate** -- a certificate whose ``schema_version`` is a
          supported v1 envelope is additionally validated with
          :func:`~omnibias.core.proof.certificate.schema_errors_v1`.

        When ``strict`` is true a ``PROVED``/``DISPROVED`` verdict additionally
        requires (i) a certificate, (ii) a present *and* valid tamper-evident
        digest, and (iii) an independent replay twin that **agrees**
        (``replay_ok is True``; ``None`` or ``False`` blocks).  This is the
        "treat the verdict as tamper-evident *and* independently replayed" mode;
        the default (non-strict) preserves graceful degradation when a replay twin
        or a digest is absent.

        When ``lean_check`` is true (or the conjecture asserts the
        ``theorem_prover_verified`` claim) and a Lean toolchain is available, the
        produced certificate's finite obligation is re-checked by the Lean kernel
        (:mod:`omnibias.core.proof.lean_check`); the resulting
        :attr:`Verdict.theorem_prover_verified` flag is set only on a genuine
        kernel pass.  Asserting the claim without a passing kernel downgrades a
        ``PROVED`` verdict to ``BLOCKED`` (the formal honesty gate).
        """
        prover = self.registry.select(conjecture)
        if prover is None:
            return Verdict(
                status="BLOCKED",
                conjecture=conjecture.name,
                kind=conjecture.kind,
                prover="<none>",
                certificate=None,
                obligations=(
                    f"no registered prover handles kind {conjecture.kind!r}",
                ),
                schema_ok=False,
                replay_ok=None,
                honesty_ok=honesty_gate(conjecture, None),
                detail="no prover registered for this kind",
            )

        attempt = prover.attempt(conjecture)
        certificate = attempt.certificate
        obligations: list[str] = list(attempt.obligations)

        schema_ok = True
        if certificate is not None:
            errors = prover.schema_errors(certificate)
            if errors:
                schema_ok = False
                obligations.extend(f"schema: {message}" for message in errors)
            # Tamper-evidence (always on): a certificate that carries a digest must
            # match its body (the same check the Lean bridge runs). Only fires on
            # sealed certs, so the many unsealed certified-evidence certs are unaffected.
            if "digest" in certificate and not verify_certificate_digest(certificate):
                schema_ok = False
                obligations.append(
                    "schema: certificate digest mismatch (tampered or stale)"
                )

        honesty_ok = honesty_gate(conjecture, certificate)

        replay_ok: bool | None = None
        if (replay or strict) and certificate is not None:
            replay_ok = prover.replay(certificate)

        status: VerdictStatus = attempt.status
        if status in ("PROVED", "DISPROVED"):
            if not schema_ok:
                status = "BLOCKED"
                obligations.append("schema validation failed")
            if replay_ok is False:
                status = "BLOCKED"
                obligations.append("independent replay disagreed with the certificate")
            if strict:
                if certificate is None:
                    status = "BLOCKED"
                    obligations.append("strict mode: no certificate to adjudicate")
                else:
                    if "digest" not in certificate:
                        status = "BLOCKED"
                        obligations.append(
                            "strict mode: an unsealed certificate (no tamper-evident "
                            "digest) cannot back a strict verdict"
                        )
                    # Require a well-formed v1 envelope (claim/payload/honesty/digest
                    # present and the digest valid).
                    v1_errors = schema_errors_v1(certificate)
                    if v1_errors:
                        status = "BLOCKED"
                        obligations.extend(
                            f"strict mode schema(v1): {message}" for message in v1_errors
                        )
                if replay_ok is not True:
                    status = "BLOCKED"
                    obligations.append(
                        "strict mode: requires an independent replay twin that "
                        "agrees (replay_ok must be True)"
                    )
        if status == "PROVED" and not honesty_ok:
            status = "BLOCKED"
            obligations.append(
                "honesty gate: an asserted claim is not supported by the "
                "certificate's honesty flags (no external evidence attached)"
            )

        schema_version: str | None = None
        if certificate is not None:
            candidate = certificate.get("schema_version")
            if isinstance(candidate, str):
                schema_version = candidate

        # Formal (Lean-kernel) gate: set theorem_prover_verified only on a genuine
        # kernel pass; an asserted-but-unbacked formal claim blocks a PROVED verdict.
        claims_formal = bool(conjecture.claims.get(FORMAL_CLAIM_KEY, False))
        theorem_prover_verified = False
        if (lean_check or claims_formal) and certificate is not None and status in (
            "PROVED",
            "DISPROVED",
        ):
            from omnibias.core.proof.lean_check import check_certificate

            theorem_prover_verified = check_certificate(certificate).verified
        if status == "PROVED" and claims_formal and not theorem_prover_verified:
            status = "BLOCKED"
            honesty_ok = False
            obligations.append(
                "honesty gate: 'theorem_prover_verified' asserted but the Lean "
                "kernel did not verify the certificate's obligation"
            )

        return Verdict(
            status=status,
            conjecture=conjecture.name,
            kind=conjecture.kind,
            prover=prover.name,
            certificate=certificate,
            obligations=tuple(obligations),
            schema_ok=schema_ok,
            replay_ok=replay_ok,
            honesty_ok=honesty_ok,
            detail=attempt.detail,
            certificate_schema_version=schema_version,
            theorem_prover_verified=theorem_prover_verified,
        )

    def evaluate_all(
        self,
        conjectures: Iterable[Conjecture],
        *,
        replay: bool = True,
        lean_check: bool = False,
        strict: bool = False,
    ) -> list[Verdict]:
        return [
            self.evaluate(conjecture, replay=replay, lean_check=lean_check, strict=strict)
            for conjecture in conjectures
        ]


__all__ = [
    "CERTIFICATE_SCHEMA_VERSION",
    "Certificate",
    "Conjecture",
    "FORMAL_CLAIM_KEY",
    "FunctionProver",
    "LeanCheckResult",
    "ProofAttempt",
    "ProofMachine",
    "Prover",
    "ProverRegistry",
    "SUPPORTED_SCHEMA_VERSIONS",
    "Verdict",
    "VerdictStatus",
    "canonical_json",
    "certificate_digest",
    "check_certificate",
    "decode_interval",
    "decode_interval_matrix",
    "decode_taylor_model",
    "encode_interval",
    "encode_interval_matrix",
    "encode_taylor_model",
    "formal_claim_forgery_errors",
    "generate_obligation",
    "honesty_gate",
    "interval_certificate",
    "kernel_earned_theorem_prover_verified",
    "kernel_root",
    "lean_check_available",
    "make_certificate",
    "positive_definite_certificate",
    "schema_errors_v1",
    "seal_certificate",
    "taylor_model_certificate",
    "verify_certificate_digest",
]
