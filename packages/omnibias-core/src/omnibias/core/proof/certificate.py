# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certificate format v1: a versioned, canonical, tamper-evident schema.

omnibias produces many independently checkable certificates (interval / Taylor
-model enclosures, spectral gaps, blow-up witnesses).  This module gives them a
**single canonical on-disk form** so that they can be hashed, diffed, replayed,
and handed to a Lean kernel for a *kernel-checked* theorem.

Design
------
* **Bit-exact floats.**  Every ``float`` is stored as its IEEE-754 ``hex()``
  string, so a sealed certificate round-trips to the last bit on any platform --
  essential when the number *is* a rigorous bound.
* **Canonical JSON.**  :func:`canonical_json` sorts keys, uses compact
  separators, and tags any stray ``float`` with a ``{"__f64__": "0x..."}`` marker
  so the digest is independent of dict ordering and decimal formatting.
* **Tamper-evident.**  :func:`seal_certificate` appends a ``sha256`` digest over
  the canonical body; :func:`verify_certificate_digest` recomputes it, so any
  mutation (a widened bound, a flipped flag) invalidates the certificate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.taylor_model import TaylorModel
from omnibias.core.verified.transcend import (
    BACKEND_LIBM_FALLBACK,
    backend_name,
    libm_fallback_used,
    require_rigorous_backend,
    strict_backend,
)

#: The current certificate schema version.
CERTIFICATE_SCHEMA_VERSION = "1.0"
#: Schema versions this module can validate / decode.
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})

#: ``meta`` key recording which transcendental backend was available when the
#: enclosure was sealed (see :mod:`omnibias.core.verified.transcend`).  Stamped
#: automatically by :func:`make_certificate` so the provenance can never be
#: forgotten by an individual producer.  Values are ``"mpmath"`` or
#: ``"libm_fallback"``.
TRANSCEND_BACKEND_KEY = "transcend_backend"

#: Transcendental backends whose enclosures are *unconditionally* rigorous.  The
#: stdlib ``libm_fallback`` path is deliberately absent: it is rigorous only
#: *given* that the platform's ``math`` error stays inside the assumed ulp budget,
#: which the language does not guarantee.
UNCONDITIONAL_TRANSCEND_BACKENDS = frozenset({"arb", "mpfr", "mpmath"})

#: Backend stamps that mark *conditionally* rigorous transcendental enclosures.
#: Includes the legacy ``"libm"`` alias so older sealed certificates / hand-built
#: meta still trip the seal-time and schema gates.
CONDITIONAL_TRANSCEND_BACKENDS = frozenset({"libm", BACKEND_LIBM_FALLBACK})

#: ``honesty`` key by which a producer may assert that its enclosure rests on no
#: unverified libm assumption.  Asserting it while sealed on a conditional
#: backend is refused by :func:`make_certificate` (at seal time) and by
#: :func:`schema_errors_v1` (on already-sealed / hand-built certificates).
UNCONDITIONAL_CLAIM_KEY = "unconditional_transcendentals"

#: Honesty key earned **only** by a genuine Lean-kernel / ProofMachine pass.
#: Producers must never supply it to :func:`make_certificate` (or any other
#: honesty entry point); the sealed body must never carry it.
THEOREM_PROVER_VERIFIED_KEY = "theorem_prover_verified"

#: Honesty keys that producers may not supply.  These are earned by the Lean
#: kernel / ProofMachine, never asserted into a sealed certificate body.
#: Supplying any of them raises :class:`ValueError`.
RESERVED_HONESTY_KEYS = frozenset({THEOREM_PROVER_VERIFIED_KEY})

Cert = dict[str, Any]


# --------------------------------------------------------------------------- #
# Bit-exact leaf encoders.
# --------------------------------------------------------------------------- #
def _enc_float(x: float) -> str:
    return float(x).hex()


def _dec_float(s: str) -> float:
    return float.fromhex(s)


def encode_interval(iv: Interval) -> dict[str, str]:
    """Bit-exact JSON encoding of an :class:`Interval`."""
    return {"lo": _enc_float(iv.lo), "hi": _enc_float(iv.hi)}


def decode_interval(d: Mapping[str, Any]) -> Interval:
    """Inverse of :func:`encode_interval`."""
    return Interval(_dec_float(d["lo"]), _dec_float(d["hi"]))


def encode_interval_matrix(rows: Sequence[Sequence[Interval]]) -> list[list[dict[str, str]]]:
    """Bit-exact JSON encoding of a matrix (sequence of rows) of :class:`Interval`."""
    return [[encode_interval(iv) for iv in row] for row in rows]


def decode_interval_matrix(rows: Sequence[Sequence[Mapping[str, Any]]]) -> list[list[Interval]]:
    """Inverse of :func:`encode_interval_matrix`."""
    return [[decode_interval(iv) for iv in row] for row in rows]


def encode_taylor_model(tm: TaylorModel) -> dict[str, Any]:
    """Bit-exact JSON encoding of a :class:`TaylorModel`."""
    return {
        "center": _enc_float(tm.center),
        "radius": _enc_float(tm.radius),
        "order": tm.order,
        "coeffs": [encode_interval(c) for c in tm.coeffs],
        "remainder": encode_interval(tm.remainder),
    }


def decode_taylor_model(d: Mapping[str, Any]) -> TaylorModel:
    """Inverse of :func:`encode_taylor_model`."""
    return TaylorModel(
        _dec_float(d["center"]),
        _dec_float(d["radius"]),
        [decode_interval(c) for c in d["coeffs"]],
        decode_interval(d["remainder"]),
    )


# --------------------------------------------------------------------------- #
# Canonicalisation, digests, sealing.
# --------------------------------------------------------------------------- #
def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return {"__f64__": obj.hex()}
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, Sequence):
        return [_canonicalize(v) for v in obj]
    raise TypeError(f"non-serialisable certificate value of type {type(obj).__name__}")


def canonical_json(cert: Mapping[str, Any]) -> str:
    """Deterministic JSON string of ``cert`` (sorted keys, hex-tagged floats)."""
    return json.dumps(
        _canonicalize(dict(cert)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def certificate_digest(cert: Mapping[str, Any]) -> str:
    """``"sha256:<hex>"`` over the canonical body (excluding any ``digest`` key)."""
    body = {k: v for k, v in cert.items() if k != "digest"}
    return "sha256:" + hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def seal_certificate(cert: Mapping[str, Any]) -> Cert:
    """Return a copy with ``schema_version`` set and a fresh ``digest`` appended."""
    out: Cert = {k: v for k, v in cert.items() if k != "digest"}
    out.setdefault("schema_version", CERTIFICATE_SCHEMA_VERSION)
    out["digest"] = certificate_digest(out)
    return out


def verify_certificate_digest(cert: Mapping[str, Any]) -> bool:
    """``True`` iff ``cert`` carries a ``digest`` that matches its body."""
    digest = cert.get("digest")
    if not isinstance(digest, str):
        return False
    return digest == certificate_digest(cert)


# --------------------------------------------------------------------------- #
# High-level builders + validator.
# --------------------------------------------------------------------------- #
def _honesty_without_reserved(honesty: Mapping[str, bool] | None) -> dict[str, bool]:
    """Return a copy of ``honesty``, refusing any :data:`RESERVED_HONESTY_KEYS`.

    Reserved flags (notably :data:`THEOREM_PROVER_VERIFIED_KEY`) are earned only
    by the Lean kernel / ProofMachine.  A producer that tries to stamp them into
    the sealed body is refused with :class:`ValueError` so a valid digest can
    never carry a forged formal claim.
    """
    if honesty is None:
        return {"unproven_claim": False}
    reserved = sorted(k for k in honesty if k in RESERVED_HONESTY_KEYS)
    if reserved:
        raise ValueError(
            "honesty must not contain reserved key(s) "
            f"{reserved}; {THEOREM_PROVER_VERIFIED_KEY} is earned only by the "
            "Lean kernel / ProofMachine, never asserted by a certificate producer"
        )
    return dict(honesty)


def make_certificate(
    *,
    claim: str,
    payload: Mapping[str, Any],
    honesty: Mapping[str, bool] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    """Build and seal a v1 certificate around ``payload``.

    ``honesty`` defaults to ``{"unproven_claim": False}`` -- the omnibias convention
    that a certificate makes **no** open-problem claim unless evidence is attached.

    Reserved honesty keys (:data:`RESERVED_HONESTY_KEYS`, at minimum
    :data:`THEOREM_PROVER_VERIFIED_KEY`) may **not** be supplied: they are earned
    only by the Lean kernel / ProofMachine.  Passing one raises
    :class:`ValueError`.

    The transcendental backend is stamped into ``meta`` under
    :data:`TRANSCEND_BACKEND_KEY` unless the caller already supplied it (so a
    decoded certificate re-sealed elsewhere keeps its original provenance rather
    than acquiring the *re-sealing* machine's).  Recording it here rather than in
    each producer means a certificate can never quietly hide that its bounds rest
    on the conditionally-rigorous ``libm_fallback`` path; read it back with
    :func:`certificate_is_unconditional`.

    Seal-time enforcement (cannot be bypassed by forgetting to call
    :func:`~omnibias.core.verified.transcend.require_rigorous_backend`):

    * If ``honesty`` asserts :data:`UNCONDITIONAL_CLAIM_KEY`, or
      :func:`~omnibias.core.verified.transcend.strict_backend` /
      :func:`~omnibias.core.verified.transcend.certificate_mode` is active, the
      rigorous (mpmath) backend is *required* -- sealing raises
      :class:`RuntimeError` on the libm fallback.
    * If a transcendental enclosure was already produced via the libm fallback
      (:func:`~omnibias.core.verified.transcend.libm_fallback_used`) while the
      same unconditional / strict gate is active, sealing likewise raises: the
      payload cannot silently claim unconditional soundness.
    """
    honesty_out = _honesty_without_reserved(honesty)
    meta_out = dict(meta) if meta is not None else {}
    # Prefer an explicit caller stamp; otherwise record what this process would
    # use *and* upgrade to libm_fallback if that path already fed an enclosure
    # (so a late mpmath import cannot launder conditional bounds).
    if TRANSCEND_BACKEND_KEY not in meta_out:
        stamped = backend_name()
        if libm_fallback_used():
            stamped = BACKEND_LIBM_FALLBACK
        meta_out[TRANSCEND_BACKEND_KEY] = stamped

    needs_rigorous = bool(honesty_out.get(UNCONDITIONAL_CLAIM_KEY, False)) or strict_backend()
    if needs_rigorous:
        # mpmath must be available *now*, and the sealed stamp must not claim the
        # conditional path.  A sticky libm_fallback_used() under a still-missing
        # mpmath is already caught by require_rigorous_backend(); when a caller
        # tries to launder a hand-stamped libm_fallback meta while mpmath is
        # present, the stamp check below refuses.
        require_rigorous_backend()
        stamped_raw = meta_out.get(TRANSCEND_BACKEND_KEY)
        stamped = stamped_raw if isinstance(stamped_raw, str) else ""
        if stamped in CONDITIONAL_TRANSCEND_BACKENDS or (
            libm_fallback_used() and stamped not in UNCONDITIONAL_TRANSCEND_BACKENDS
        ):
            raise RuntimeError(
                "refusing to seal an unconditional / strict-mode certificate: a "
                "transcendental enclosure was produced under the conditionally-"
                "rigorous libm_fallback backend (or the stamp records it). Install "
                "mpmath, enter certificate_mode(), and recompute the enclosure "
                "before sealing."
            )

    cert: Cert = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "claim": str(claim),
        "payload": dict(payload),
        "honesty": honesty_out,
        "meta": meta_out,
    }
    return seal_certificate(cert)


def certificate_transcend_backend(cert: Mapping[str, Any]) -> str | None:
    """The backend recorded in ``cert``'s ``meta``, or ``None`` when absent.

    ``None`` means *unknown provenance* -- a certificate sealed before the stamp
    existed, or built by hand -- which callers must treat as conditional, not as
    a licence to assume the rigorous backend.
    """
    meta = cert.get("meta")
    if not isinstance(meta, Mapping):
        return None
    recorded = meta.get(TRANSCEND_BACKEND_KEY)
    return recorded if isinstance(recorded, str) else None


def certificate_is_unconditional(cert: Mapping[str, Any]) -> bool:
    """``True`` iff ``cert`` records a backend that needs no libm assumption.

    An absent or unrecognised stamp answers ``False``: soundness claims default to
    the weaker reading, so unknown provenance never passes for rigorous.
    """
    return certificate_transcend_backend(cert) in UNCONDITIONAL_TRANSCEND_BACKENDS


def interval_certificate(
    claim: str,
    interval: Interval,
    *,
    honesty: Mapping[str, bool] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    """A sealed certificate asserting a quantity lies in ``interval``."""
    payload = {"type": "interval", "interval": encode_interval(interval)}
    return make_certificate(claim=claim, payload=payload, honesty=honesty, meta=meta)


def taylor_model_certificate(
    claim: str,
    model: TaylorModel,
    *,
    honesty: Mapping[str, bool] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    """A sealed certificate carrying a :class:`TaylorModel` enclosure."""
    payload = {"type": "taylor_model", "taylor_model": encode_taylor_model(model)}
    return make_certificate(claim=claim, payload=payload, honesty=honesty, meta=meta)


def positive_definite_certificate(
    claim: str,
    pivots: Sequence[Interval],
    *,
    matrix: Sequence[Sequence[Interval]] | None = None,
    honesty: Mapping[str, bool] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    r"""A sealed certificate that a symmetric interval matrix box is positive definite.

    ``pivots`` are the interval ``LDL^T`` pivots ``D_jj`` of the matrix box (from
    :func:`omnibias.core.verified.eig_operator.interval_ldlt_pivots`).  The matrix box is
    certified positive definite exactly when every pivot has a strictly positive lower
    endpoint (zero negative inertia).  The optional ``matrix`` records the enclosed
    symmetric matrix the pivots factorise, for provenance and a future division-free
    ``L D L^T`` reconstruction check.

    Unlike the single-scalar ``eig_min`` interval an :func:`interval_certificate` would
    seal, the Lean bridge turns this payload into the kernel-verified *inertia-vector*
    obligation ``allPivotsPos`` (``matrix_positive_definite_certified``) -- the full
    positive-definiteness statement rather than its scalar shadow.
    """
    pivot_list = list(pivots)
    payload: dict[str, Any] = {
        "type": "positive_definite",
        "n": len(pivot_list),
        "pivots": [encode_interval(p) for p in pivot_list],
    }
    if matrix is not None:
        payload["matrix"] = encode_interval_matrix(matrix)
    return make_certificate(claim=claim, payload=payload, honesty=honesty, meta=meta)


def schema_errors_v1(cert: Mapping[str, Any]) -> list[str]:
    """Structural validation of a v1 certificate (empty list == valid)."""
    errors: list[str] = []
    version = cert.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(f"unsupported schema_version {version!r}")
    for key in ("claim", "payload", "honesty", "digest"):
        if key not in cert:
            errors.append(f"missing required field {key!r}")
    if "claim" in cert and not isinstance(cert["claim"], str):
        errors.append("'claim' must be a string")
    if "payload" in cert and not isinstance(cert["payload"], Mapping):
        errors.append("'payload' must be a mapping")
    if "honesty" in cert and not isinstance(cert["honesty"], Mapping):
        errors.append("'honesty' must be a mapping")
    honesty = cert.get("honesty")
    if isinstance(honesty, Mapping):
        reserved = sorted(k for k in honesty if k in RESERVED_HONESTY_KEYS)
        if reserved:
            errors.append(
                f"honesty must not contain reserved key(s) {reserved}; "
                f"{THEOREM_PROVER_VERIFIED_KEY} is earned only by the Lean kernel "
                "/ ProofMachine"
            )
        if bool(honesty.get(UNCONDITIONAL_CLAIM_KEY, False)) and not certificate_is_unconditional(
            cert
        ):
            recorded = certificate_transcend_backend(cert)
            errors.append(
                f"honesty.{UNCONDITIONAL_CLAIM_KEY} is asserted but the certificate was "
                f"sealed on transcendental backend {recorded!r}, whose rigour is only "
                "conditional on the platform libm error budget; install mpmath and "
                "re-seal, or drop the claim"
            )
    if "digest" in cert and not verify_certificate_digest(cert):
        errors.append("digest mismatch (tampered or stale certificate)")
    return errors


__all__ = [
    "CERTIFICATE_SCHEMA_VERSION",
    "CONDITIONAL_TRANSCEND_BACKENDS",
    "Cert",
    "RESERVED_HONESTY_KEYS",
    "SUPPORTED_SCHEMA_VERSIONS",
    "THEOREM_PROVER_VERIFIED_KEY",
    "TRANSCEND_BACKEND_KEY",
    "UNCONDITIONAL_CLAIM_KEY",
    "UNCONDITIONAL_TRANSCEND_BACKENDS",
    "canonical_json",
    "certificate_digest",
    "certificate_is_unconditional",
    "certificate_transcend_backend",
    "decode_interval",
    "decode_interval_matrix",
    "decode_taylor_model",
    "encode_interval",
    "encode_interval_matrix",
    "encode_taylor_model",
    "interval_certificate",
    "make_certificate",
    "positive_definite_certificate",
    "schema_errors_v1",
    "seal_certificate",
    "taylor_model_certificate",
    "verify_certificate_digest",
]
