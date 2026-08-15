# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The Mathlib-backed certificate checker (twin of ``omnibias.core.proof.lean_check``).

This bridge drives the Mathlib-backed Lean project
(``formal/omnibias-analytic``), the deliberate counterpart to the Mathlib-free
minimal kernel driven by :mod:`omnibias.core.proof.lean_check`.  It

1. extracts the **finite, rational** obligation carried by a certificate and
   re-derives it exactly (over :class:`~fractions.Fraction`), refusing any
   certificate whose ``digest`` does not match its body (tamper-evidence, reused
   from ``omnibias.core``).  Supported obligations (Phase 1):

   * **enclosed-quantity sign** -- an ``interval`` payload, a
     ``pinn_aposteriori_error`` finite margin, a ``taylor_model`` centre value,
     or any raw ``lo``/``hi`` mapping;
   * **positive-definite** -- every ``LDL^T`` pivot's rational lower endpoint is
     positive (the ``ℚ`` analogue of the kernel's integer inertia obligation,
     with no integer-scaling hack);
   * **Newton-Kantorovich / Krawczyk contraction** -- the radii polynomial is
     negative at the certified radius and the contraction constant is ``< 1``
     (and, for Krawczyk, the image box lies strictly inside ``[c - r, c + r]``):
     genuine rational *polynomial* inequalities the integer kernel cannot state;
   * **tower coefficients** -- a ``tower_coeffs`` payload whose integer list
     matches :mod:`omnibias.core.verified.coeffs` at a finite order; the Lean
     obligation is that those integers equal the ``OmnibiasAnalytic.Tower``
     recurrence (not an ``iteratedDeriv`` identity, and not a collapse);

2. emits a tiny Lean source file (``OmnibiasAnalytic/Generated.lean``) that
   discharges the obligation over ``ℚ`` -- against the project's *proven*
   ``enclosed_pos`` / ``enclosed_neg`` lemmas for the sign case, and by closing
   the concrete rational (in)equalities with ``norm_num`` for the rest.  Every
   rational is emitted **directly** (no common-denominator integer scaling);
3. invokes ``lake build`` so **Mathlib's kernel** re-checks it; and
4. reports whether the kernel accepted the proof.

The bridge only ever emits an obligation it has itself confirmed holds exactly
over ``ℚ``; a certificate whose stored bound does not reproduce the claimed
inequality yields ``None`` rather than a failing Lean file.

Trust tier (important).  A pass here sets the reserved :data:`MATHLIB_CLAIM_KEY`
(``"mathlib_verified"``) tier only.  This is a **distinct, larger** trust base
than the minimal Mathlib-free kernel: it never sets, and must never be conflated
with, ``theorem_prover_verified`` (earned solely by
:mod:`omnibias.core.proof.lean_check`), and it never implies ``unproven_claim``.

The module is dependency-light (standard library plus ``omnibias-core`` for the
digest check).  When no Lean toolchain (``lake``) or analytic checkout is present
it degrades gracefully -- :func:`mathlib_check_available` returns ``False`` and
:func:`check_certificate` returns an ``available=False`` result rather than
raising -- so a normal test / CI run without Lean is unaffected.  Only a genuine
``lake`` pass yields ``verified=True``; the flag can never be forged by the
certificate itself.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, TypeGuard

from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.formal.tower import (
    TOWER_COEFF_LIST_NAMES,
    tower_coeffs,
)

#: Reserved verdict / claim key for the Mathlib trust tier.  Deliberately
#: **distinct** from the minimal-kernel ``theorem_prover_verified`` so the two
#: trust bases are never conflated.
MATHLIB_CLAIM_KEY = "mathlib_verified"

#: Relative location of the Mathlib-backed analytic project within the repository.
_ANALYTIC_REL = Path("formal") / "omnibias-analytic"
#: The generated-obligation module the bridge overwrites (then restores).
_GENERATED_REL = Path("OmnibiasAnalytic") / "Generated.lean"


@dataclass(frozen=True)
class MathlibCheckResult:
    """The outcome of a Mathlib-kernel check of a single certificate."""

    verified: bool
    available: bool
    obligation: str
    detail: str


def analytic_root(start: Path | None = None) -> Path | None:
    """Locate ``formal/omnibias-analytic`` by walking up from ``start``."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / _ANALYTIC_REL
        if (candidate / "lakefile.lean").is_file():
            return candidate
    return None


def mathlib_check_available(start: Path | None = None) -> bool:
    """``True`` iff both a ``lake`` executable and the analytic checkout are present."""
    return shutil.which("lake") is not None and analytic_root(start) is not None


# --------------------------------------------------------------------------- #
# Rational leaf helpers.
# --------------------------------------------------------------------------- #
def _as_float(value: Any) -> float:
    """Decode a leaf that is either a bit-exact ``float.hex()`` string or a number."""
    return float.fromhex(value) if isinstance(value, str) else float(value)


def _frac(value: Any) -> Fraction:
    """Exact rational for a certificate leaf (hex-float string or raw number)."""
    return Fraction(_as_float(value))


def _rat_literal(value: Fraction | float) -> str:
    """Render ``value`` as an exact Lean ``ℚ`` literal (no floating point)."""
    frac = value if isinstance(value, Fraction) else Fraction(value)
    if frac.denominator == 1:
        return f"({frac.numerator} : \u211a)"
    return f"(({frac.numerator} : \u211a) / {frac.denominator})"


def _is_pair(value: Any) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) == 2


_HEADER = (
    "/- AUTO-GENERATED by omnibias.formal.mathlib_check. DO NOT EDIT. -/\n"
    "import OmnibiasAnalytic.Check.EnclosedSign\n\n"
    "namespace OmnibiasAnalytic.Generated\n\n"
)
_TOWER_HEADER = (
    "/- AUTO-GENERATED by omnibias.formal.mathlib_check. DO NOT EDIT. -/\n"
    "import OmnibiasAnalytic.Tower\n\n"
    "namespace OmnibiasAnalytic.Generated\n\n"
    "open OmnibiasAnalytic.Tower\n\n"
)
_FOOTER = "\nend OmnibiasAnalytic.Generated\n"


def _wrap(body: str) -> str:
    return _HEADER + body + _FOOTER


def _wrap_tower(body: str) -> str:
    return _TOWER_HEADER + body + _FOOTER


def _theorem_conj(props: Sequence[str], comment: str) -> str:
    """A closed ``theorem obligation`` proving a conjunction of numeric facts.

    A single fact is closed by ``norm_num``; a conjunction is split by the
    anonymous constructor (which flattens the right-nested ``∧``) so ``norm_num``
    only ever sees the atomic rational (in)equalities.
    """
    if len(props) == 1:
        return f"/-- {comment} -/\ntheorem obligation : {props[0]} := by norm_num\n"
    stmt = " \u2227 ".join(f"({p})" for p in props)
    holes = ", ".join(["?_"] * len(props))
    return (
        f"/-- {comment} -/\n"
        f"theorem obligation : {stmt} := by\n"
        f"  refine \u27e8{holes}\u27e9 <;> norm_num\n"
    )


# --------------------------------------------------------------------------- #
# Obligation extraction -> Lean source.
# --------------------------------------------------------------------------- #
def _gen_positive_definite(cert: Mapping[str, Any]) -> str | None:
    """ℚ positive-definite: every ``LDL^T`` pivot lower endpoint is strictly positive."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "positive_definite"):
        return None
    pivots = payload.get("pivots")
    if not (isinstance(pivots, Sequence) and not isinstance(pivots, str | bytes) and pivots):
        return None
    los: list[Fraction] = []
    for pivot in pivots:
        if not (isinstance(pivot, Mapping) and "lo" in pivot):
            return None
        los.append(_frac(pivot["lo"]))
    if not all(lo > 0 for lo in los):
        return None
    props = [f"0 < {_rat_literal(lo)}" for lo in los]
    comment = (
        f"Certified positive-definite: all {len(los)} LDL\u1d40 pivot lower endpoints are "
        "strictly positive, so the negative inertia is zero (factorisation a trusted input)."
    )
    return _wrap(_theorem_conj(props, comment))


def _gen_radii_polynomial(cert: Mapping[str, Any]) -> str | None:
    """Newton-Kantorovich contraction: ``p(r) < 0`` and ``kappa(r) < 1`` over ``ℚ``."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "radii_polynomial"):
        return None
    try:
        y0, z0, z1, z2, r = (
            _frac(payload["Y0"]),
            _frac(payload["Z0"]),
            _frac(payload["Z1"]),
            _frac(payload["Z2"]),
            _frac(payload["radius"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    p_value = y0 + (z0 + z1) * r + z2 * r * r - r
    kappa = z0 + z1 + 2 * z2 * r
    if not (p_value < 0 and kappa < 1):  # re-derived exactly; refuse to emit a false fact
        return None
    yl, z0l, z1l, z2l, rl = (_rat_literal(v) for v in (y0, z0, z1, z2, r))
    p_prop = f"{yl} + ({z0l} + {z1l}) * {rl} + {z2l} * {rl} ^ 2 - {rl} < 0"
    k_prop = f"{z0l} + {z1l} + 2 * {z2l} * {rl} < 1"
    comment = (
        "Newton-Kantorovich contraction: the radii polynomial is negative at the certified "
        "radius and the contraction constant is < 1 (existence/uniqueness a trusted input)."
    )
    return _wrap(_theorem_conj([p_prop, k_prop], comment))


def _gen_krawczyk(cert: Mapping[str, Any]) -> str | None:
    """Krawczyk test: ``kappa < 1`` and the image box is strictly inside ``[c-r, c+r]``."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "krawczyk"):
        return None
    center, enclosure = payload.get("center"), payload.get("enclosure")
    if not (
        isinstance(center, Sequence)
        and isinstance(enclosure, Sequence)
        and not isinstance(center, str | bytes)
        and not isinstance(enclosure, str | bytes)
        and center
        and len(center) == len(enclosure)
        and payload.get("kappa") is not None
        and payload.get("radius") is not None
    ):
        return None
    try:
        r, kappa = _frac(payload["radius"]), _frac(payload["kappa"])
        centers = [_frac(c) for c in center]
        boxes: list[tuple[Fraction, Fraction]] = []
        for edge in enclosure:
            if not _is_pair(edge):
                return None
            boxes.append((_frac(edge[0]), _frac(edge[1])))
    except (TypeError, ValueError):
        return None
    if not (kappa < 1):
        return None
    props = [f"{_rat_literal(kappa)} < 1"]
    rl = _rat_literal(r)
    for c, (lo, hi) in zip(centers, boxes, strict=True):
        if not (c - r < lo and hi < c + r):  # re-derived exactly
            return None
        cl = _rat_literal(c)
        props.append(f"{cl} - {rl} < {_rat_literal(lo)}")
        props.append(f"{_rat_literal(hi)} < {cl} + {rl}")
    comment = (
        "Krawczyk unique-zero test: contraction constant < 1 and the image box lies strictly "
        "inside [center - r, center + r] (existence/uniqueness a trusted input)."
    )
    return _wrap(_theorem_conj(props, comment))


def _taylor_centre_interval(model: Mapping[str, Any]) -> tuple[Fraction, Fraction] | None:
    """Enclosure of a Taylor model's value at its expansion centre: ``coeffs[0] + remainder``."""
    coeffs, remainder = model.get("coeffs"), model.get("remainder")
    if not (isinstance(coeffs, Sequence) and not isinstance(coeffs, str | bytes) and coeffs):
        return None
    c0 = coeffs[0]
    if not (
        isinstance(c0, Mapping)
        and "lo" in c0
        and "hi" in c0
        and isinstance(remainder, Mapping)
        and "lo" in remainder
        and "hi" in remainder
    ):
        return None
    return _frac(c0["lo"]) + _frac(remainder["lo"]), _frac(c0["hi"]) + _frac(remainder["hi"])


def _extract_sign_interval(cert: Mapping[str, Any]) -> tuple[Fraction, Fraction] | None:
    """Pull a rational ``(lo, hi)`` enclosure whose sign is the obligation."""
    payload = cert.get("payload")
    if isinstance(payload, Mapping):
        kind = payload.get("type")
        if kind == "interval":
            interval = payload.get("interval")
            if isinstance(interval, Mapping) and "lo" in interval and "hi" in interval:
                return _frac(interval["lo"]), _frac(interval["hi"])
        if kind == "pinn_aposteriori_error":
            finite = payload.get("finite_obligation")
            if isinstance(finite, Mapping) and finite.get("type") == "error_bound_le_threshold":
                margin = finite.get("margin")
                if _is_pair(margin):
                    return _frac(margin[0]), _frac(margin[1])
        if kind == "taylor_model":
            model = payload.get("taylor_model")
            if isinstance(model, Mapping):
                centre = _taylor_centre_interval(model)
                if centre is not None:
                    return centre
    interval = cert.get("interval")
    if isinstance(interval, Mapping) and "lo" in interval and "hi" in interval:
        return _frac(interval["lo"]), _frac(interval["hi"])
    return None


def _gen_tower_coeffs(cert: Mapping[str, Any]) -> str | None:
    """Integer tower coefficients equal the Lean recurrence at a finite order."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "tower_coeffs"):
        return None
    family, raw_n, raw_coeffs = payload.get("family"), payload.get("n"), payload.get("coeffs")
    if not (
        isinstance(family, str)
        and family in TOWER_COEFF_LIST_NAMES
        and isinstance(raw_n, int)
        and not isinstance(raw_n, bool)
        and raw_n >= 0
        and isinstance(raw_coeffs, Sequence)
        and not isinstance(raw_coeffs, str | bytes)
        and raw_coeffs
        and all(isinstance(c, int) and not isinstance(c, bool) for c in raw_coeffs)
    ):
        return None
    try:
        expected = tower_coeffs(family, raw_n)
    except ValueError:
        return None
    got = tuple(int(c) for c in raw_coeffs)
    if got != expected:
        return None
    lean_list = ", ".join(str(c) for c in expected)
    name = TOWER_COEFF_LIST_NAMES[family]
    comment = (
        f"Exact {family} tower coefficients at order {raw_n} equal the Lean "
        "recurrence (Python verified.coeffs re-derived; not an iteratedDeriv "
        "identity and not a collapse)."
    )
    body = (
        f"/-- {comment} -/\n"
        f"theorem obligation : {name} {raw_n} = [{lean_list}] := by native_decide\n"
    )
    return _wrap_tower(body)


def _gen_sign(cert: Mapping[str, Any]) -> str | None:
    """Enclosed-quantity sign, discharged against the proven ``enclosed_pos/neg`` lemmas."""
    interval = _extract_sign_interval(cert)
    if interval is None:
        return None
    lo, hi = interval
    if lo > 0:
        lit = _rat_literal(lo)
        body = (
            f"/-- Enclosed quantity is positive: {lit} \u2264 x \u21d2 0 < x. -/\n"
            f"theorem obligation (x : \u211a) (hx : {lit} \u2264 x) : 0 < x :=\n"
            f"  OmnibiasAnalytic.Check.enclosed_pos hx (by norm_num)\n"
        )
        return _wrap(body)
    if hi < 0:
        lit = _rat_literal(hi)
        body = (
            f"/-- Enclosed quantity is negative: x \u2264 {lit} \u21d2 x < 0. -/\n"
            f"theorem obligation (x : \u211a) (hx : x \u2264 {lit}) : x < 0 :=\n"
            f"  OmnibiasAnalytic.Check.enclosed_neg hx (by norm_num)\n"
        )
        return _wrap(body)
    return None


#: Ordered obligation generators.  The label is the obligation *class* reported
#: by :func:`classify_obligation`; :func:`generate_obligation` returns the first
#: non-``None`` source.  A single source of truth keeps the two in lockstep.
_GENERATORS: tuple[tuple[str, Callable[[Mapping[str, Any]], str | None]], ...] = (
    ("positive_definite", _gen_positive_definite),
    ("radii_polynomial", _gen_radii_polynomial),
    ("krawczyk", _gen_krawczyk),
    ("tower_coeffs", _gen_tower_coeffs),
    ("sign", _gen_sign),
)


def generate_obligation(cert: Mapping[str, Any]) -> str | None:
    r"""Return Lean source discharging ``cert``'s finite rational obligation, or ``None``.

    Tried in order: positive-definite pivots, the Newton-Kantorovich radii
    polynomial, the Krawczyk test, exact tower coefficients, then the
    enclosed-quantity sign (interval / PDE finite margin / Taylor-model centre
    value / raw ``lo``/``hi`` mapping).  Every emitted obligation is first
    re-derived exactly over ``ℚ`` (or exact ``int`` coefficients for the tower);
    a payload that does not reproduce the claimed fact yields ``None``.
    """
    for _label, generator in _GENERATORS:
        source = generator(cert)
        if source is not None:
            return source
    return None


def classify_obligation(cert: Mapping[str, Any]) -> str | None:
    """Return the obligation *class* the bridge would discharge for ``cert``.

    One of ``"positive_definite"``, ``"radii_polynomial"``, ``"krawczyk"``,
    ``"tower_coeffs"`` or ``"sign"`` -- whichever generator first yields Lean,
    re-derived exactly -- or ``None`` when the certificate carries no
    Mathlib-checkable finite obligation.  Runs no Lean: it is the cheap triage the loop driver uses before
    invoking ``lake``, and it agrees with :func:`generate_obligation` by
    construction (the same ordered registry).
    """
    for label, generator in _GENERATORS:
        if generator(cert) is not None:
            return label
    return None


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def check_certificate(
    cert: Mapping[str, Any],
    *,
    timeout: float = 1800.0,
    start: Path | None = None,
) -> MathlibCheckResult:
    """Generate the obligation, run ``lake build``, and report the Mathlib verdict.

    A certificate with a mismatched ``digest`` is rejected before any Lean is
    emitted (tamper-evidence).  If the toolchain is unavailable the result has
    ``available=False`` and ``verified=False``.  A ``verified=True`` result sets
    only the :data:`MATHLIB_CLAIM_KEY` tier -- never ``theorem_prover_verified``.
    """
    if "digest" in cert and not verify_certificate_digest(cert):
        return MathlibCheckResult(False, True, "", "certificate digest mismatch (tampered/stale)")

    obligation = generate_obligation(cert)
    if obligation is None:
        return MathlibCheckResult(
            False, True, "", "no Mathlib-checkable finite obligation in certificate"
        )

    root = analytic_root(start)
    if root is None or shutil.which("lake") is None:
        return MathlibCheckResult(
            False, False, obligation, "Lean toolchain or analytic checkout unavailable"
        )

    generated = root / _GENERATED_REL
    original = generated.read_text(encoding="utf-8") if generated.exists() else None
    try:
        generated.write_text(obligation, encoding="utf-8")
        proc = subprocess.run(
            ["lake", "build"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        ok = proc.returncode == 0
        detail = (
            "Mathlib kernel accepted the obligation" if ok else (proc.stderr or proc.stdout)[-2000:]
        )
        return MathlibCheckResult(ok, True, obligation, detail)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        return MathlibCheckResult(False, False, obligation, f"lake invocation failed: {exc}")
    finally:
        if original is not None:
            generated.write_text(original, encoding="utf-8")


__all__ = [
    "MATHLIB_CLAIM_KEY",
    "MathlibCheckResult",
    "analytic_root",
    "check_certificate",
    "classify_obligation",
    "generate_obligation",
    "mathlib_check_available",
]
