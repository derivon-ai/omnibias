# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certificate router over named collapses: kind in, verdict plus a reason tree.

This is not a general theorem prover. Natural language is never a premise.
A caller names a finite kind; the engine adjudicates with a shipped collapse
or a finite catalog family, seals a v1 certificate, and optionally asks Lean
only when :func:`~omnibias.core.proof.lean_check.generate_obligation` already
applies. ``theorem_prover_verified`` is earned only by a genuine
``lake build``. A float residual is not a proof. ``BLOCKED`` is not falsity.
Continuum parents are not inferred.

Do not register ``gap`` as a named collapse. The ``gap`` *kind* is Enclosure
Collapse of a claimed ``OPT`` sandwich.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

from omnibias.core.collapse.identity import difference_coeffs, identity_collapse
from omnibias.core.collapse.pairing import pairing_collapse, pairing_value
from omnibias.core.collapse.rank import rank_collapse
from omnibias.core.collapse.schema import default_honesty
from omnibias.core.collapse.verdict import ObligationVerdict, adjudicate_residual
from omnibias.core.collapse.winding import integers_in, winding_collapse
from omnibias.core.proof.catalog import CatalogEntry, discover, register_catalog
from omnibias.core.proof.certificate import (
    NO_TRANSCENDENTAL_BACKEND,
    RESERVED_HONESTY_KEYS,
    TRANSCEND_BACKEND_KEY,
    decode_interval,
    encode_interval,
    make_certificate,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.proof.discovery import (
    DiscoveryResult,
    IntegerIntervalFamily,
    run_discovery,
)
from omnibias.core.proof.lift import residual_identically_zero
from omnibias.core.verified.interval import Interval

if TYPE_CHECKING:
    from omnibias.core.proof import (
        Certificate,
        Conjecture,
        ProofAttempt,
        ProofMachine,
        Verdict,
    )

EngineKind = Literal[
    "catalog_family",
    "enclosure_sign",
    "external",
    "gap",
    "identity",
    "pairing",
    "rank",
    "residual",
    "winding",
]

ENGINE_KINDS: frozenset[str] = frozenset(
    {
        "catalog_family",
        "enclosure_sign",
        "external",
        "gap",
        "identity",
        "pairing",
        "rank",
        "residual",
        "winding",
    }
)

_INTEGER_SQUARE = "integer_square"


def _honesty(*, spec_name: str, extra: Mapping[str, bool] | None = None) -> dict[str, bool]:
    payload = default_honesty(spec_name=spec_name)
    payload["unproven_claim"] = False
    payload["float_residual_is_proof"] = False
    payload["continuum_parent_inferred"] = False
    if extra is not None:
        payload.update(dict(extra))
    for key in RESERVED_HONESTY_KEYS:
        payload.pop(key, None)
    return payload


def _seal(
    claim: str,
    payload: Mapping[str, object],
    honesty: Mapping[str, bool],
    *,
    no_transcendentals: bool = False,
) -> dict[str, Any]:
    """Seal an engine result with explicit provenance where it is known.

    Direct interval comparisons use only the supplied IEEE-754 endpoints and
    outward-rounded arithmetic.  They must not inherit an unrelated
    process-global fallback mark from an earlier transcendental computation.
    """
    meta = (
        {TRANSCEND_BACKEND_KEY: NO_TRANSCENDENTAL_BACKEND}
        if no_transcendentals
        else None
    )
    return make_certificate(claim=claim, payload=payload, honesty=honesty, meta=meta)


def _require_mapping(data: Mapping[str, Any]) -> Mapping[str, Any]:
    return data


def _as_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | Fraction):
        raise TypeError(f"{name} must be a real number")
    return float(value)


def _as_int(value: object, *, name: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    return int(value)


def _as_str(value: object, *, name: str, default: str | None = None) -> str:
    if value is None and default is not None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _interval_bounds(lo: object, hi: object) -> Interval:
    return Interval(_as_float(lo, name="lo"), _as_float(hi, name="hi"))


def _domain_interval(value: object) -> Interval:
    if isinstance(value, Interval):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) == 2:
        return _interval_bounds(value[0], value[1])
    if isinstance(value, Mapping):
        return _interval_bounds(value["lo"], value["hi"])
    raise TypeError("domain must be [lo, hi] or {lo, hi}")


def _coeff_list(value: object, *, name: str) -> list[int | float | Fraction]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{name} must be a coefficient sequence")
    out: list[int | float | Fraction] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float | Fraction):
            raise TypeError(f"{name} entries must be int, float, or Fraction")
        out.append(item)
    return out


def _json_coeffs(coeffs: Sequence[int | float | Fraction]) -> list[str]:
    return [str(item) for item in coeffs]


def _coeffs_from_json(value: object) -> list[Fraction]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError("coefficient payload must be a sequence")
    return [Fraction(str(item)) for item in value]


def _residual_payload(box: Interval | None) -> dict[str, str] | None:
    if box is None:
        return None
    return encode_interval(box)


def _attempt(
    status: Literal["PROVED", "DISPROVED", "BLOCKED"],
    certificate: Mapping[str, Any] | None,
    detail: str,
) -> ProofAttempt:
    from omnibias.core.proof import ProofAttempt

    return ProofAttempt(
        status=status,
        certificate=dict(certificate) if certificate is not None else None,
        detail=detail,
        obligations=() if status != "BLOCKED" else (detail,),
    )


def _blocked(detail: str, certificate: Mapping[str, Any] | None = None) -> ProofAttempt:
    return _attempt("BLOCKED", certificate, detail)


def _from_obligation(
    kind: str,
    spec_name: str,
    verdict: ObligationVerdict,
    payload: Mapping[str, object],
    honesty: Mapping[str, bool],
) -> ProofAttempt:
    residual = verdict.outcome.residual
    body: dict[str, object] = {
        "type": "engine",
        "engine_kind": kind,
        "status": verdict.status,
        "spec_name": spec_name,
        "surviving": verdict.outcome.surviving,
        "detail": verdict.detail,
        "residual": _residual_payload(residual),
    }
    body.update(dict(payload))
    cert = _seal(kind, body, honesty)
    return _attempt(verdict.status, cert, verdict.detail)


@dataclass(frozen=True)
class ReasonStep:
    """One node of the reason tree. Prose is a rendering, not a premise."""

    kind: str
    status: str
    spec_name: str
    surviving: str | int | None
    residual: dict[str, float] | None
    detail: str
    honesty: Mapping[str, bool]

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "status": self.status,
            "spec_name": self.spec_name,
            "surviving": self.surviving,
            "residual": self.residual,
            "detail": self.detail,
            "honesty": dict(self.honesty),
        }


@dataclass(frozen=True)
class EngineResult:
    """ProofMachine verdict plus the collapse / catalog reason tree."""

    verdict: Verdict
    reason: tuple[ReasonStep, ...]

    @property
    def proved(self) -> bool:
        return self.verdict.proved

    @property
    def disproved(self) -> bool:
        return self.verdict.disproved

    @property
    def blocked(self) -> bool:
        return self.verdict.blocked

    def explain(self) -> str:
        """Render the reason tree. This is not a natural-language proof."""

        lines = [
            f"{self.verdict.status}: kind={self.verdict.kind} "
            f"prover={self.verdict.prover}"
        ]
        if self.verdict.detail:
            lines.append(self.verdict.detail)
        for step in self.reason:
            surviving = "none" if step.surviving is None else repr(step.surviving)
            lines.append(
                f"- {step.spec_name}: {step.status} surviving={surviving}; "
                f"{step.detail}"
            )
        lines.append("A float residual is not a proof.")
        if self.blocked:
            lines.append("BLOCKED is not falsity.")
        lines.append("Continuum parents are not inferred.")
        lines.append(
            "theorem_prover_verified is earned only by a genuine lake build."
        )
        return "\n".join(lines)


def _reason_from_certificate(
    kind: str,
    verdict: Verdict,
) -> tuple[ReasonStep, ...]:
    honesty: dict[str, bool] = {}
    spec_name = kind
    surviving: str | int | None = None
    residual: dict[str, float] | None = None
    detail = verdict.detail
    cert = verdict.certificate
    if isinstance(cert, Mapping):
        raw_honesty = cert.get("honesty")
        if isinstance(raw_honesty, Mapping):
            honesty = {str(key): bool(value) for key, value in raw_honesty.items()}
        payload = cert.get("payload")
        if isinstance(payload, Mapping):
            spec_raw = payload.get("spec_name")
            if isinstance(spec_raw, str) and spec_raw:
                spec_name = spec_raw
            raw_surviving = payload.get("surviving")
            if isinstance(raw_surviving, str | int) and not isinstance(raw_surviving, bool):
                surviving = raw_surviving
            raw_detail = payload.get("detail")
            if isinstance(raw_detail, str) and raw_detail:
                detail = raw_detail
            raw_residual = payload.get("residual")
            if isinstance(raw_residual, Mapping) and "lo" in raw_residual:
                try:
                    box = decode_interval(raw_residual)
                    residual = {"lo": box.lo, "hi": box.hi}
                except (KeyError, TypeError, ValueError):
                    residual = None
            elif payload.get("type") == "interval":
                interval = payload.get("interval")
                if isinstance(interval, Mapping):
                    try:
                        box = decode_interval(interval)
                        residual = {"lo": box.lo, "hi": box.hi}
                    except (KeyError, TypeError, ValueError):
                        residual = None
    return (
        ReasonStep(
            kind=kind,
            status=verdict.status,
            spec_name=spec_name,
            surviving=surviving,
            residual=residual,
            detail=detail,
            honesty=honesty,
        ),
    )


def _schema_errors(certificate: Certificate) -> list[str]:
    errors: list[str] = []
    if "digest" in certificate:
        errors.extend(schema_errors_v1(certificate))
        if not verify_certificate_digest(certificate):
            errors.append("digest mismatch")
    honesty = certificate.get("honesty")
    if not isinstance(honesty, Mapping):
        return ["honesty must be a mapping"]
    if honesty.get("theorem_prover_verified"):
        errors.append("theorem_prover_verified must not be asserted on the certificate")
    if honesty.get("float_residual_is_proof"):
        errors.append("float_residual_is_proof must be false")
    if honesty.get("continuum_parent_inferred"):
        errors.append("continuum_parent_inferred must be false")
    payload = certificate.get("payload")
    if payload is not None and not isinstance(payload, Mapping):
        errors.append("payload must be a mapping")
    return errors


def _payload_of(certificate: Certificate) -> Mapping[str, Any] | None:
    payload = certificate.get("payload")
    return payload if isinstance(payload, Mapping) else None


def _prove_external(conjecture: Conjecture) -> ProofAttempt:
    parent = _as_str(conjecture.data.get("parent"), name="parent", default="unspecified")
    detail = f"external obligation {parent!r} is not in scope; not a parent claim"
    honesty = _honesty(
        spec_name="enclosure",
        extra={"continuum_parent_inferred": False, "external_obligation": True},
    )
    cert = _seal(
        "external",
        {
            "type": "engine",
            "engine_kind": "external",
            "status": "BLOCKED",
            "spec_name": "external",
            "surviving": None,
            "detail": detail,
            "parent": parent,
            "residual": None,
        },
        honesty,
    )
    return _blocked(detail, cert)


def _replay_external(certificate: Certificate) -> bool | None:
    payload = _payload_of(certificate)
    if payload is None:
        return False
    return payload.get("engine_kind") == "external" and payload.get("status") == "BLOCKED"


def _prove_residual(conjecture: Conjecture) -> ProofAttempt:
    data = _require_mapping(conjecture.data)
    if "value" in data and ("lo" not in data or "hi" not in data):
        return _blocked("a float residual is not a certificate; pass lo and hi")
    try:
        box = _interval_bounds(data["lo"], data["hi"])
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked(str(exc))
    verdict = adjudicate_residual(box)
    honesty = _honesty(spec_name="verdict", extra={"verdict_collapse": True})
    payload: dict[str, object] = {
        "type": "interval",
        "interval": encode_interval(box),
        "engine_kind": "residual",
        "status": verdict.status,
        "spec_name": "verdict",
        "surviving": verdict.outcome.surviving,
        "detail": verdict.detail,
        "residual": encode_interval(box),
        "inputs": {"lo": box.lo, "hi": box.hi},
    }
    cert = _seal("residual", payload, honesty, no_transcendentals=True)
    return _attempt(verdict.status, cert, verdict.detail)


def _replay_residual(certificate: Certificate) -> bool | None:
    payload = _payload_of(certificate)
    if payload is None:
        return False
    raw = payload.get("residual")
    if not isinstance(raw, Mapping):
        raw = payload.get("interval")
    if not isinstance(raw, Mapping):
        return False
    try:
        box = decode_interval(raw)
        fresh = adjudicate_residual(box)
    except (KeyError, TypeError, ValueError):
        return False
    return fresh.status == payload.get("status")


def _prove_enclosure_sign(conjecture: Conjecture) -> ProofAttempt:
    data = _require_mapping(conjecture.data)
    try:
        box = _interval_bounds(data["lo"], data["hi"])
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked(str(exc))
    honesty = _honesty(spec_name="enclosure", extra={"enclosure_collapse": True})
    if box.lo > 0.0:
        status: Literal["PROVED", "DISPROVED", "BLOCKED"] = "PROVED"
        surviving: str | None = "positive"
        detail = "sound enclosure excludes 0 from below; quantity is positive"
    elif box.hi < 0.0:
        status = "PROVED"
        surviving = "negative"
        detail = "sound enclosure excludes 0 from above; quantity is negative"
    else:
        status = "BLOCKED"
        surviving = None
        detail = "enclosure contains 0; Inconclusive, not a sign proof"
    payload: dict[str, object] = {
        "type": "interval",
        "interval": encode_interval(box),
        "engine_kind": "enclosure_sign",
        "status": status,
        "spec_name": "enclosure",
        "surviving": surviving,
        "detail": detail,
        "residual": encode_interval(box),
        "inputs": {"lo": box.lo, "hi": box.hi},
    }
    cert = _seal("enclosure_sign", payload, honesty, no_transcendentals=True)
    return _attempt(status, cert, detail)


def _replay_enclosure_sign(certificate: Certificate) -> bool | None:
    payload = _payload_of(certificate)
    if payload is None:
        return False
    raw = payload.get("interval")
    if not isinstance(raw, Mapping):
        return False
    try:
        box = decode_interval(raw)
    except (KeyError, TypeError, ValueError):
        return False
    if box.lo > 0.0 or box.hi < 0.0:
        expected = "PROVED"
    else:
        expected = "BLOCKED"
    return payload.get("status") == expected


def _gap_box(data: Mapping[str, Any]) -> Interval:
    if "L" in data and "U" in data:
        return _interval_bounds(data["L"], data["U"])
    return _interval_bounds(data["lo"], data["hi"])


def _prove_gap(conjecture: Conjecture) -> ProofAttempt:
    data = _require_mapping(conjecture.data)
    try:
        box = _gap_box(data)
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked(str(exc))
    honesty = _honesty(
        spec_name="enclosure",
        extra={"enclosure_collapse": True, "gap_is_named_collapse": False},
    )
    expected_raw = data.get("expected")
    expected: float | None
    if expected_raw is None:
        expected = None
    else:
        try:
            expected = _as_float(expected_raw, name="expected")
        except TypeError as exc:
            return _blocked(str(exc))
    if expected is not None and not box.contains(expected):
        status: Literal["PROVED", "DISPROVED", "BLOCKED"] = "DISPROVED"
        surviving: str | None = "DISPROVED"
        detail = "expected OPT lies outside the sound sandwich"
    elif box.lo == box.hi:
        status = "PROVED"
        surviving = "point_plus_proof"
        detail = "sound OPT sandwich collapsed to a singleton"
    else:
        status = "BLOCKED"
        surviving = None
        detail = "OPT sandwich still has positive width; Inconclusive"
    payload: dict[str, object] = {
        "type": "interval",
        "interval": encode_interval(box),
        "engine_kind": "gap",
        "status": status,
        "spec_name": "enclosure",
        "surviving": surviving,
        "detail": detail,
        "residual": encode_interval(box),
        "inputs": {"L": box.lo, "U": box.hi, "expected": expected},
    }
    cert = _seal("gap", payload, honesty, no_transcendentals=True)
    return _attempt(status, cert, detail)


def _replay_gap(certificate: Certificate) -> bool | None:
    payload = _payload_of(certificate)
    if payload is None:
        return False
    raw = payload.get("interval")
    inputs = payload.get("inputs")
    if not isinstance(raw, Mapping) or not isinstance(inputs, Mapping):
        return False
    try:
        box = decode_interval(raw)
    except (KeyError, TypeError, ValueError):
        return False
    expected = inputs.get("expected")
    if isinstance(expected, int | float) and not isinstance(expected, bool):
        if not box.contains(float(expected)):
            return payload.get("status") == "DISPROVED"
    if box.lo == box.hi:
        return payload.get("status") == "PROVED"
    return payload.get("status") == "BLOCKED"


def _prove_identity(conjecture: Conjecture) -> ProofAttempt:
    data = _require_mapping(conjecture.data)
    try:
        left = _coeff_list(data["left"], name="left")
        right = _coeff_list(data["right"], name="right")
        domain = _domain_interval(data["domain"])
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked(str(exc))
    verdict = identity_collapse(left, right, domain)
    honesty = _honesty(spec_name="identity", extra={"identity_collapse": True})
    diff = difference_coeffs(left, right)
    residual = verdict.outcome.residual
    if verdict.proved and not diff:
        payload: dict[str, object] = {
            "type": "rational_identity",
            "lhs_terms": [[1, 1]],
            "rhs": 1,
            "engine_kind": "identity",
            "status": verdict.status,
            "spec_name": "identity",
            "surviving": verdict.outcome.surviving,
            "detail": verdict.detail,
            "residual": _residual_payload(residual),
            "inputs": {
                "left": _json_coeffs(left),
                "right": _json_coeffs(right),
                "domain": [domain.lo, domain.hi],
            },
        }
        cert = _seal("identity", payload, honesty)
        return _attempt(verdict.status, cert, verdict.detail)
    return _from_obligation(
        "identity",
        "identity",
        verdict,
        {
            "inputs": {
                "left": _json_coeffs(left),
                "right": _json_coeffs(right),
                "domain": [domain.lo, domain.hi],
            }
        },
        honesty,
    )


def _replay_identity(certificate: Certificate) -> bool | None:
    payload = _payload_of(certificate)
    if payload is None:
        return False
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    try:
        left = _coeffs_from_json(inputs["left"])
        right = _coeffs_from_json(inputs["right"])
        domain = _domain_interval(inputs["domain"])
        fresh = identity_collapse(left, right, domain)
    except (KeyError, TypeError, ValueError):
        return False
    return fresh.status == payload.get("status")


def _center(value: object) -> complex:
    if value is None:
        return 0j
    if isinstance(value, complex):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return complex(float(value), 0.0)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) == 2:
        return complex(_as_float(value[0], name="center.re"), _as_float(value[1], name="center.im"))
    raise TypeError("center must be a complex or [re, im]")


def _prove_winding(conjecture: Conjecture) -> ProofAttempt:
    data = _require_mapping(conjecture.data)
    try:
        coeffs = _coeff_list(data["coeffs"], name="coeffs")
        center = _center(data.get("center"))
        radius = _as_float(data.get("radius", 1.0), name="radius")
        contour = _as_str(data.get("contour"), name="contour", default="circle")
        half_width_raw = data.get("half_width")
        half_width = (
            None
            if half_width_raw is None
            else _as_float(half_width_raw, name="half_width")
        )
        half_height_raw = data.get("half_height")
        half_height = (
            None
            if half_height_raw is None
            else _as_float(half_height_raw, name="half_height")
        )
        expected_raw = data.get("expected")
        expected = None if expected_raw is None else _as_int(expected_raw, name="expected")
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked(str(exc))
    verdict = winding_collapse(
        [float(item) for item in coeffs],
        center,
        radius,
        expected=expected,
        contour=contour,
        half_width=half_width,
        half_height=half_height,
    )
    honesty = _honesty(spec_name="winding", extra={"winding_collapse": True})
    inputs: dict[str, object] = {
        "coeffs": _json_coeffs(coeffs),
        "center": [center.real, center.imag],
        "radius": radius,
        "expected": expected,
    }
    if contour != "circle" or half_width is not None or half_height is not None:
        inputs["contour"] = contour
        if half_width is not None:
            inputs["half_width"] = half_width
        if half_height is not None:
            inputs["half_height"] = half_height
    return _from_obligation(
        "winding",
        "winding",
        verdict,
        {"inputs": inputs},
        honesty,
    )


def _replay_winding(certificate: Certificate) -> bool | None:
    payload = _payload_of(certificate)
    if payload is None:
        return False
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    try:
        coeffs = _coeffs_from_json(inputs["coeffs"])
        center = _center(inputs.get("center"))
        radius = _as_float(inputs.get("radius", 1.0), name="radius")
        contour = _as_str(inputs.get("contour"), name="contour", default="circle")
        half_width_raw = inputs.get("half_width")
        half_width = (
            None
            if half_width_raw is None
            else _as_float(half_width_raw, name="half_width")
        )
        half_height_raw = inputs.get("half_height")
        half_height = (
            None
            if half_height_raw is None
            else _as_float(half_height_raw, name="half_height")
        )
        expected_raw = inputs.get("expected")
        expected = None if expected_raw is None else _as_int(expected_raw, name="expected")
        fresh = winding_collapse(
            [float(item) for item in coeffs],
            center,
            radius,
            expected=expected,
            contour=contour,
            half_width=half_width,
            half_height=half_height,
        )
    except (KeyError, TypeError, ValueError):
        return False
    if fresh.proved and isinstance(fresh.outcome.surviving, int):
        residual = fresh.outcome.residual
        if residual is not None and integers_in(residual) != (fresh.outcome.surviving,):
            return False
    return fresh.status == payload.get("status") and fresh.outcome.surviving == payload.get(
        "surviving"
    )


def _test_pack(value: object) -> list[list[int | float | Fraction]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError("tests must be a sequence of coefficient sequences")
    return [_coeff_list(item, name="test") for item in value]


def _prove_pairing(conjecture: Conjecture) -> ProofAttempt:
    data = _require_mapping(conjecture.data)
    try:
        residual = _coeff_list(data["residual"], name="residual")
        tests = _test_pack(data["tests"])
        lo = Fraction(str(data.get("lo", -1)))
        hi = Fraction(str(data.get("hi", 1)))
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked(str(exc))
    verdict = pairing_collapse(residual, tests, lo=lo, hi=hi)
    honesty = _honesty(
        spec_name="pairing",
        extra={"pairing_collapse": True, "not_a_strong_solution": True},
    )
    return _from_obligation(
        "pairing",
        "pairing",
        verdict,
        {
            "inputs": {
                "residual": _json_coeffs(residual),
                "tests": [_json_coeffs(test) for test in tests],
                "lo": str(lo),
                "hi": str(hi),
            }
        },
        honesty,
    )


def _replay_pairing(certificate: Certificate) -> bool | None:
    payload = _payload_of(certificate)
    if payload is None:
        return False
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    try:
        residual = _coeffs_from_json(inputs["residual"])
        raw_tests = inputs["tests"]
        if not isinstance(raw_tests, Sequence) or isinstance(raw_tests, str | bytes):
            return False
        tests = [_coeffs_from_json(test) for test in raw_tests]
        lo = Fraction(str(inputs.get("lo", -1)))
        hi = Fraction(str(inputs.get("hi", 1)))
        values = [pairing_value(residual, test, lo, hi) for test in tests]
        fresh = pairing_collapse(residual, tests, lo=lo, hi=hi)
    except (KeyError, TypeError, ValueError):
        return False
    if fresh.proved and any(value != 0 for value in values):
        return False
    return fresh.status == payload.get("status")


def _integer_matrix(value: object) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError("matrix must be a sequence of rows")
    rows: list[list[int]] = []
    width = 0
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, str | bytes):
            raise TypeError("matrix rows must be sequences")
        ints: list[int] = []
        for entry in row:
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise TypeError(
                    "rank collapse requires an exact integer matrix; "
                    "a float singular value is not a certificate"
                )
            ints.append(int(entry))
        if not ints:
            raise ValueError("matrix rows must be non-empty")
        if width == 0:
            width = len(ints)
        elif len(ints) != width:
            raise ValueError("matrix rows must share a width")
        rows.append(ints)
    if not rows:
        raise ValueError("matrix must be non-empty")
    return rows


def _prove_rank(conjecture: Conjecture) -> ProofAttempt:
    data = _require_mapping(conjecture.data)
    try:
        matrix = _integer_matrix(data["matrix"])
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked(str(exc))
    report = rank_collapse(matrix)
    honesty = _honesty(
        spec_name="rank",
        extra={"rank_collapse": True, "float_svd_is_proof": False},
    )
    return _from_obligation(
        "rank",
        "rank",
        report.verdict,
        {
            "kernel": [list(vec) for vec in report.kernel],
            "inputs": {"matrix": matrix},
        },
        honesty,
    )


def _replay_rank(certificate: Certificate) -> bool | None:
    payload = _payload_of(certificate)
    if payload is None:
        return False
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    try:
        matrix = _integer_matrix(inputs["matrix"])
        report = rank_collapse(matrix)
    except (KeyError, TypeError, ValueError):
        return False
    zeros = [0] * len(matrix)
    kernel = report.kernel
    for vec in kernel:
        if not residual_identically_zero(matrix, list(vec), zeros):
            return False
    return report.verdict.status == payload.get("status")


def _integer_square_family(data: Mapping[str, Any]) -> IntegerIntervalFamily:
    return IntegerIntervalFamily(
        lo=_as_int(data.get("lo"), name="lo", default=-3),
        hi=_as_int(data.get("hi"), name="hi", default=3),
        target_square=_as_int(
            data.get("target_square", data.get("target")),
            name="target_square",
            default=4,
        ),
    )


def _integer_square_factory(**kwargs: Any) -> DiscoveryResult:
    family = _integer_square_family(kwargs)
    proposer = _as_str(kwargs.get("proposer"), name="proposer", default="score_guided")
    budget = _as_int(kwargs.get("budget"), name="budget", default=8)
    return run_discovery(family.statement, family, proposer, budget=budget)


def _discovery_attempt(kind: str, result: DiscoveryResult) -> ProofAttempt:
    honesty = _honesty(
        spec_name="verdict",
        extra={"catalog_family": True, "float_residual_is_proof": False},
    )
    payload: dict[str, object] = {
        "type": "engine",
        "engine_kind": kind,
        "status": result.status,
        "spec_name": "catalog_family",
        "surviving": result.candidate if isinstance(result.candidate, str | int) else None,
        "detail": result.detail,
        "residual": None,
        "discovery": result.as_dict(),
        "inputs": {
            "family": result.family,
            "budget": result.budget,
            "proposer": result.proposer,
        },
    }
    cert = _seal(kind, payload, honesty)
    return _attempt(result.status, cert, result.detail)


def _prove_catalog_family(conjecture: Conjecture) -> ProofAttempt:
    data = _require_mapping(conjecture.data)
    try:
        family_name = _as_str(data.get("family"), name="family")
    except TypeError as exc:
        return _blocked(str(exc))
    if family_name == _INTEGER_SQUARE:
        try:
            result = _integer_square_factory(**dict(data))
        except (TypeError, ValueError) as exc:
            return _blocked(str(exc))
        return _discovery_attempt("catalog_family", result)
    try:
        found = discover(family_name, **{key: value for key, value in data.items() if key != "family"})
    except KeyError as exc:
        return _blocked(str(exc))
    if isinstance(found, DiscoveryResult):
        return _discovery_attempt("catalog_family", found)
    if isinstance(found, Mapping) and found.get("status") in {"PROVED", "DISPROVED", "BLOCKED"}:
        status = found["status"]
        detail = str(found.get("detail", "catalog factory returned a status mapping"))
        if status == "PROVED":
            proven: Literal["PROVED", "DISPROVED", "BLOCKED"] = "PROVED"
        elif status == "DISPROVED":
            proven = "DISPROVED"
        else:
            proven = "BLOCKED"
        honesty = _honesty(spec_name="verdict", extra={"catalog_family": True})
        cert = _seal(
            "catalog_family",
            {
                "type": "engine",
                "engine_kind": "catalog_family",
                "status": proven,
                "spec_name": "catalog_family",
                "surviving": None,
                "detail": detail,
                "residual": None,
                "discovery": dict(found),
                "inputs": {"family": family_name},
            },
            honesty,
        )
        return _attempt(proven, cert, detail)
    return _blocked("catalog factory did not return a DiscoveryResult")


def _replay_catalog_family(certificate: Certificate) -> bool | None:
    payload = _payload_of(certificate)
    if payload is None:
        return False
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    family_name = inputs.get("family")
    if family_name in {_INTEGER_SQUARE, "exists_square", None}:
        try:
            fresh = _integer_square_factory(
                lo=_as_int(inputs.get("lo"), name="lo", default=-3),
                hi=_as_int(inputs.get("hi"), name="hi", default=3),
                target_square=_as_int(
                    inputs.get("target_square"), name="target_square", default=4
                ),
                budget=_as_int(inputs.get("budget"), name="budget", default=8),
                proposer=_as_str(
                    inputs.get("proposer"), name="proposer", default="score_guided"
                ),
            )
        except (TypeError, ValueError):
            return False
        return fresh.status == payload.get("status")
    return payload.get("status") in {"PROVED", "DISPROVED", "BLOCKED"}


def _register_integer_square() -> None:
    register_catalog(
        CatalogEntry(
            kind=_INTEGER_SQUARE,
            obligation="some integer x in the interval with x^2 equal to the target",
            parent="toy",
            parent_status="already_true",
            package="omnibias.core.proof.engine",
            mode="exact_search",
            complete=True,
        ),
        _integer_square_factory,
    )


_register_integer_square()


def _store_catalog_inputs(conjecture: Conjecture, attempt: ProofAttempt) -> ProofAttempt:
    """Persist catalog box parameters on the sealed payload for replay."""

    cert = attempt.certificate
    if cert is None:
        return attempt
    payload = cert.get("payload")
    if not isinstance(payload, Mapping):
        return attempt
    inputs = dict(payload.get("inputs") or {}) if isinstance(payload.get("inputs"), Mapping) else {}
    data = conjecture.data
    for key in ("lo", "hi", "target_square", "target", "budget", "proposer", "family"):
        if key in data:
            inputs[key] = data[key]
    if "target_square" not in inputs and "target" in inputs:
        inputs["target_square"] = inputs["target"]
    new_payload = dict(payload)
    new_payload["inputs"] = inputs
    honesty_raw = cert.get("honesty")
    honesty = (
        {str(key): bool(value) for key, value in honesty_raw.items()}
        if isinstance(honesty_raw, Mapping)
        else _honesty(spec_name="verdict")
    )
    sealed = _seal("catalog_family", new_payload, honesty)
    return _attempt(attempt.status, sealed, attempt.detail)


def _prove_catalog_family_sealed(conjecture: Conjecture) -> ProofAttempt:
    return _store_catalog_inputs(conjecture, _prove_catalog_family(conjecture))


def build_engine_machine() -> ProofMachine:
    """ProofMachine with the core collapse / catalog kinds registered."""

    from omnibias.core.proof import FunctionProver, ProofMachine

    machine = ProofMachine()
    specs: tuple[tuple[str, Any, Any], ...] = (
        ("external", _prove_external, _replay_external),
        ("residual", _prove_residual, _replay_residual),
        ("enclosure_sign", _prove_enclosure_sign, _replay_enclosure_sign),
        ("gap", _prove_gap, _replay_gap),
        ("identity", _prove_identity, _replay_identity),
        ("winding", _prove_winding, _replay_winding),
        ("pairing", _prove_pairing, _replay_pairing),
        ("rank", _prove_rank, _replay_rank),
        ("catalog_family", _prove_catalog_family_sealed, _replay_catalog_family),
    )
    for kind, prove_fn, replay_fn in specs:
        machine.register(
            FunctionProver(
                name=kind,
                kinds=frozenset({kind}),
                prove_fn=prove_fn,
                schema_fn=_schema_errors,
                replay_fn=replay_fn,
            )
        )
    return machine


def prove(
    kind: str,
    data: Mapping[str, object] | None = None,
    *,
    name: str = "",
    lean_check: bool = False,
    strict: bool = False,
) -> EngineResult:
    """Adjudicate one named finite statement and return verdict plus reasons."""

    from omnibias.core.proof import Conjecture

    machine = build_engine_machine()
    conjecture = Conjecture(
        name=name or kind,
        kind=kind,
        data=dict(data or {}),
    )
    verdict = machine.evaluate(
        conjecture,
        replay=True,
        lean_check=lean_check,
        strict=strict,
    )
    return EngineResult(verdict=verdict, reason=_reason_from_certificate(kind, verdict))


__all__ = [
    "ENGINE_KINDS",
    "EngineKind",
    "EngineResult",
    "ReasonStep",
    "build_engine_machine",
    "prove",
]
