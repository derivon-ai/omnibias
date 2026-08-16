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
   * **NK / Krawczyk existence** -- a planted ``nk_existence`` payload whose
     rational plant matches the locked quadratic ``x² - 2``; the Lean
     obligation is the unique-root theorem on ``[5/4, 7/4]`` (not merely
     ``p(r) < 0``);
   * **enclosure trace** -- a planted ``enclosure_trace`` payload whose
     rational DAG Lean replays (tower Horner, NK bounds, Bernoulli /
     named ``zetaNeg1``, or 2x2 LDLT); not a transcendental enclosure;
   * **named unique zero** -- a planted ``named_zero`` payload whose
     rational box matches a locked named polynomial (circle ∩ line,
     Hopf radial ``r(1-r²)``, or Chebyshev ``T₃``); the Lean obligation
     is the unique-root theorem on that compact box;
   * **compact box** -- a planted ``compact_box`` payload: a named
     incompressible residual on ``[1/2, 1]²``, or the characteristic
     polynomial of a named 2x2 rational matrix (ratio ``5/8``); not a
     continuum regularity or continuum gauge claim;
   * **casimir** -- a planted ``casimir`` payload: the locked SU(2)
     fundamental gap ``3/4`` or the SU(3) fundamental ``4/3``; finite
     rational identities, not a continuum gauge claim;
   * **polymer** -- a planted ``polymer`` payload: the locked
     backtrack coordination ``15``, first-step ``20``, or crude ``24``
     at ``d=4``; finite arithmetic, not a continuum gauge claim;
   * **sixj** -- a planted ``sixj`` payload: the locked Racah value
     ``{1/2 1/2 0; 1/2 1/2 0} = -1/2`` or the vanishing all-``1/2``
     triad; finite rational identities, not a continuum gauge claim;
   * **haar_volume** -- a planted ``haar_volume`` payload: the locked
     Weyl prefactor ``6*4=24``; finite arithmetic, not a continuum Haar
     theorem and not 4-D SU(3) Yang-Mills;

2. emits a tiny Lean source file (``OmnibiasAnalytic/Generated.lean``) that
   discharges the obligation -- against the project's *proven*
   ``enclosed_pos`` / ``enclosed_neg`` lemmas for the sign case, by applying a
   Check unique-root theorem for planted NK existence, by applying a Check
   enclosure-trace plant theorem, by applying a Check named unique-zero
   theorem, by applying a Check compact-box theorem, by applying a Check
   Casimir theorem, and by closing the
   concrete rational (in)equalities with ``norm_num`` for the rest.
   Every rational is emitted **directly** (no common-denominator integer
   scaling);
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
from omnibias.formal.casimir import (
    LEGAL_CASIMIR_FAMILIES,
    lean_casimir_theorem,
    locked_casimir_matches,
)
from omnibias.formal.casimir import (
    family_facts_hold as casimir_facts_hold,
)
from omnibias.formal.compact import (
    LEGAL_COMPACT_FAMILIES,
    family_facts_hold,
    lean_compact_theorem,
    locked_compact_matches,
)
from omnibias.formal.haar import (
    LEGAL_HAAR_FAMILIES,
    lean_haar_theorem,
    locked_haar_matches,
)
from omnibias.formal.haar import (
    family_facts_hold as haar_facts_hold,
)
from omnibias.formal.named import (
    LEGAL_NAMED_FAMILIES,
    family_selfmap_holds,
    lean_named_theorem,
    locked_named_matches,
)
from omnibias.formal.nk import (
    LEGAL_NK_FAMILIES,
    LEGAL_NK_ROUTES,
    PLANT_KRAWCZYK_HI,
    PLANT_KRAWCZYK_KAPPA,
    PLANT_KRAWCZYK_LO,
    lean_nk_theorem,
    locked_plant_matches,
    plant_box,
    plant_radii_kappa,
    plant_radii_poly,
)
from omnibias.formal.polymer import (
    LEGAL_POLYMER_FAMILIES,
    lean_polymer_theorem,
    locked_polymer_matches,
)
from omnibias.formal.polymer import (
    family_facts_hold as polymer_facts_hold,
)
from omnibias.formal.sixj import (
    LEGAL_SIXJ_FAMILIES,
    lean_sixj_theorem,
    locked_sixj_matches,
)
from omnibias.formal.sixj import (
    family_facts_hold as sixj_facts_hold,
)
from omnibias.formal.tower import (
    TOWER_COEFF_LIST_NAMES,
    tower_coeffs,
)
from omnibias.formal.trace import (
    LEGAL_TRACE_FAMILIES,
    lean_trace_theorem,
    locked_trace_matches,
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
_NK_HEADER = (
    "/- AUTO-GENERATED by omnibias.formal.mathlib_check. DO NOT EDIT. -/\n"
    "import OmnibiasAnalytic.Check.Kantorovich.Plant\n\n"
    "namespace OmnibiasAnalytic.Generated\n\n"
    "open OmnibiasAnalytic.Check Set\n\n"
)
_TRACE_HEADER = (
    "/- AUTO-GENERATED by omnibias.formal.mathlib_check. DO NOT EDIT. -/\n"
    "import OmnibiasAnalytic.Check.Enclosure.Plant\n\n"
    "namespace OmnibiasAnalytic.Generated\n\n"
    "open OmnibiasAnalytic.Check QInterval Set\n\n"
)
_NAMED_HEADER = (
    "/- AUTO-GENERATED by omnibias.formal.mathlib_check. DO NOT EDIT. -/\n"
    "import OmnibiasAnalytic.Check.Kantorovich.Named\n\n"
    "namespace OmnibiasAnalytic.Generated\n\n"
    "open OmnibiasAnalytic.Check Set\n\n"
)
_COMPACT_HEADER = (
    "/- AUTO-GENERATED by omnibias.formal.mathlib_check. DO NOT EDIT. -/\n"
    "import OmnibiasAnalytic.Check.Compact\n\n"
    "namespace OmnibiasAnalytic.Generated\n\n"
    "open OmnibiasAnalytic.Check Set\n\n"
)
_CASIMIR_HEADER = (
    "/- AUTO-GENERATED by omnibias.formal.mathlib_check. DO NOT EDIT. -/\n"
    "import OmnibiasAnalytic.Check.Casimir\n\n"
    "namespace OmnibiasAnalytic.Generated\n\n"
    "open OmnibiasAnalytic.Check\n\n"
)
_POLYMER_HEADER = (
    "/- AUTO-GENERATED by omnibias.formal.mathlib_check. DO NOT EDIT. -/\n"
    "import OmnibiasAnalytic.Check.Polymer\n\n"
    "namespace OmnibiasAnalytic.Generated\n\n"
    "open OmnibiasAnalytic.Check\n\n"
)
_SIXJ_HEADER = (
    "/- AUTO-GENERATED by omnibias.formal.mathlib_check. DO NOT EDIT. -/\n"
    "import OmnibiasAnalytic.Check.SixJ\n\n"
    "namespace OmnibiasAnalytic.Generated\n\n"
    "open OmnibiasAnalytic.Check\n\n"
)
_HAAR_HEADER = (
    "/- AUTO-GENERATED by omnibias.formal.mathlib_check. DO NOT EDIT. -/\n"
    "import OmnibiasAnalytic.Check.HaarVolume\n\n"
    "namespace OmnibiasAnalytic.Generated\n\n"
    "open OmnibiasAnalytic.Check\n\n"
)
_FOOTER = "\nend OmnibiasAnalytic.Generated\n"


def _wrap(body: str) -> str:
    return _HEADER + body + _FOOTER


def _wrap_tower(body: str) -> str:
    return _TOWER_HEADER + body + _FOOTER


def _wrap_nk(body: str) -> str:
    return _NK_HEADER + body + _FOOTER


def _wrap_trace(body: str) -> str:
    return _TRACE_HEADER + body + _FOOTER


def _wrap_named(body: str) -> str:
    return _NAMED_HEADER + body + _FOOTER


def _wrap_compact(body: str) -> str:
    return _COMPACT_HEADER + body + _FOOTER


def _wrap_casimir(body: str) -> str:
    return _CASIMIR_HEADER + body + _FOOTER


def _wrap_polymer(body: str) -> str:
    return _POLYMER_HEADER + body + _FOOTER


def _wrap_sixj(body: str) -> str:
    return _SIXJ_HEADER + body + _FOOTER


def _wrap_haar(body: str) -> str:
    return _HAAR_HEADER + body + _FOOTER


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
        "radius and the contraction constant is < 1 (existence/uniqueness a trusted input; "
        "use nk_existence for a Lean unique root theorem)."
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
        "inside [center - r, center + r] (existence/uniqueness a trusted input; "
        "use nk_existence for a Lean unique root theorem)."
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


def _gen_nk_existence(cert: Mapping[str, Any]) -> str | None:
    """Planted unique root of ``x² - 2`` on ``[5/4, 7/4]`` via a Check lemma."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "nk_existence"):
        return None
    family, route = payload.get("family"), payload.get("route")
    if family not in LEGAL_NK_FAMILIES or route not in LEGAL_NK_ROUTES:
        return None
    if not locked_plant_matches(payload):
        return None
    if route == "radii":
        if not (plant_radii_poly() < 0 and plant_radii_kappa() < 1):
            return None
        comment = (
            "Unique root of x^2 - 2 in [5/4, 7/4] via the radii Lipschitz bound "
            "(planted quadratic; not a continuum PDE)."
        )
    else:
        lo, hi = plant_box()
        if not (
            PLANT_KRAWCZYK_KAPPA < 1
            and lo < PLANT_KRAWCZYK_LO
            and PLANT_KRAWCZYK_HI < hi
        ):
            return None
        comment = (
            "Unique root of x^2 - 2 in [5/4, 7/4] via the 1-D Krawczyk derivative "
            "bound (planted quadratic; not a continuum PDE)."
        )
    thm = lean_nk_theorem(route)
    body = (
        f"/-- {comment} -/\n"
        "theorem obligation :\n"
        "    ∃! x : ℝ, x ∈ Icc (5 / 4) (7 / 4) ∧ quadraticPlant x = 0 :=\n"
        f"  {thm}\n"
    )
    return _wrap_nk(body)


def _gen_enclosure_trace(cert: Mapping[str, Any]) -> str | None:
    """Replay a locked rational enclosure DAG via a Check plant theorem."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "enclosure_trace"):
        return None
    family = payload.get("family")
    if family not in LEGAL_TRACE_FAMILIES or not locked_trace_matches(payload):
        return None
    thm = lean_trace_theorem(family)
    if family == "tower":
        comment = (
            "Horner of sigmoidPoly 2 at 2/3 equals -2/27 "
            "(planted rational DAG; not an iteratedDeriv identity)."
        )
        stmt = (
            "(evalTrace towerHornerOps).getLast? = some (point (-2 / 27))"
        )
    elif family == "nk":
        comment = (
            "Replayed Y0, kappa, p(r) for x^2 - 2 and the unique root on "
            "[5/4, 7/4] (planted quadratic; not a continuum PDE)."
        )
        stmt = (
            "((evalTrace nkBoundOps)[8]? = some (point (1 / 12)) ∧ "
            "(evalTrace nkBoundOps)[10]? = some (point (1 / 3)) ∧ "
            "(evalTrace nkBoundOps)[14]? = some (point (-1 / 8))) ∧ "
            "∃! x : ℝ, x ∈ Icc (5 / 4) (7 / 4) ∧ quadraticPlant x = 0"
        )
    elif family == "bernoulli":
        comment = (
            "B2 = 1/6 and zetaNeg1 := -B2/2 = -1/12 "
            "(algebraic definition; not analytic continuation)."
        )
        stmt = (
            "(evalTrace bernoulliOps)[7]? = some (point (1 / 6)) ∧ "
            "(evalTrace bernoulliOps)[10]? = some (point (-1 / 12))"
        )
    else:
        comment = (
            "LDLT pivots of [[2, 1], [1, 2]] are 2 and 3/2, both positive "
            "(planted matrix; not a general SOS engine)."
        )
        stmt = (
            "(evalTrace ldltOps)[0]? = some (point 2) ∧ "
            "(evalTrace ldltOps)[6]? = some (point (3 / 2)) ∧ "
            "(0 : ℚ) < 2 ∧ (0 : ℚ) < 3 / 2"
        )
    body = (
        f"/-- {comment} -/\n"
        "theorem obligation :\n"
        f"    {stmt} :=\n"
        f"  {thm}\n"
    )
    return _wrap_trace(body)


def _gen_named_zero(cert: Mapping[str, Any]) -> str | None:
    """Apply a Check unique-zero theorem for a locked named polynomial."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "named_zero"):
        return None
    family = payload.get("family")
    if (
        family not in LEGAL_NAMED_FAMILIES
        or not locked_named_matches(payload)
        or not family_selfmap_holds(family)
    ):
        return None
    thm = lean_named_theorem(family)
    if family == "circle_line":
        comment = (
            "Unique intersection of the unit circle and y = x in [5/8, 7/8]^2 "
            "(named polynomial; not a continuum PDE)."
        )
        stmt = (
            "∃! p : ℝ × ℝ, "
            "p ∈ Icc (5 / 8) (7 / 8) ×ˢ Icc (5 / 8) (7 / 8) ∧ "
            "circleLine p = (0, 0)"
        )
    elif family == "hopf_radial":
        comment = (
            "Unique root of r(1 - r^2) on [3/4, 5/4] "
            "(named polynomial; not a Lohner time-2π return map)."
        )
        stmt = "∃! r : ℝ, r ∈ Icc (3 / 4) (5 / 4) ∧ hopfRadial r = 0"
    else:
        comment = (
            "Unique root of Chebyshev T3 on [3/4, 1] "
            "(algebraic CAP leaf; not a continuum CCF / Euler blow-up)."
        )
        stmt = "∃! z : ℝ, z ∈ Icc (3 / 4) 1 ∧ ccfChebyshev z = 0"
    body = (
        f"/-- {comment} -/\n"
        "theorem obligation :\n"
        f"    {stmt} :=\n"
        f"  {thm}\n"
    )
    return _wrap_named(body)


def _gen_compact_box(cert: Mapping[str, Any]) -> str | None:
    """Apply a Check compact-box residual or finite-matrix gap theorem."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "compact_box"):
        return None
    family = payload.get("family")
    if (
        family not in LEGAL_COMPACT_FAMILIES
        or not locked_compact_matches(payload)
        or not family_facts_hold(family)
    ):
        return None
    thm = lean_compact_theorem(family)
    if family == "ns_box":
        comment = (
            "Named incompressible residual lo on [1/2, 1]^2 "
            "(polynomial field; not a continuum regularity theorem)."
        )
        stmt = (
            "nsBoxDiv = 0 ∧ "
            "∀ p : ℝ × ℝ, p ∈ Icc (1 / 2 : ℝ) 1 ×ˢ Icc (1 / 2 : ℝ) 1 → "
            "(nsBoxResidual p).1 ≥ 1 / 2"
        )
    else:
        comment = (
            "Char-poly roots 8 and 5 of [[13/2, 3/2], [3/2, 13/2]] "
            "with ratio 5/8 (finite matrix; not a continuum gauge claim)."
        )
        stmt = (
            "(transferChar 8 = 0 ∧ transferChar 5 = 0) ∧ "
            "|(5 : ℝ)| / 8 < 1 ∧ 0 < (8 : ℝ) - 5"
        )
    body = (
        f"/-- {comment} -/\n"
        "theorem obligation :\n"
        f"    {stmt} :=\n"
        f"  {thm}\n"
    )
    return _wrap_compact(body)


def _gen_casimir(cert: Mapping[str, Any]) -> str | None:
    """Apply a Check SU(2) / SU(3) Casimir identity theorem."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "casimir"):
        return None
    family = payload.get("family")
    if (
        family not in LEGAL_CASIMIR_FAMILIES
        or not locked_casimir_matches(payload)
        or not casimir_facts_hold(family)
    ):
        return None
    thm = lean_casimir_theorem(family)
    if family == "su2_fund":
        comment = (
            "SU(2) Casimir gap C2(1)-C2(0)=3/4 "
            "(finite rational identity; not a continuum gauge claim)."
        )
        stmt = "casimirSU2 1 0 - casimirSU2 0 0 = 3 / 4"
    else:
        comment = (
            "SU(3) fundamental Casimir C2(1,0)=4/3 "
            "(finite rational identity; not a continuum gauge claim)."
        )
        stmt = "casimirSU3 1 0 0 = 4 / 3"
    body = (
        f"/-- {comment} -/\n"
        "theorem obligation :\n"
        f"    {stmt} :=\n"
        f"  {thm}\n"
    )
    return _wrap_casimir(body)


def _gen_polymer(cert: Mapping[str, Any]) -> str | None:
    """Apply a Check polymer-coordination identity theorem."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "polymer"):
        return None
    family = payload.get("family")
    if (
        family not in LEGAL_POLYMER_FAMILIES
        or not locked_polymer_matches(payload)
        or not polymer_facts_hold(family)
    ):
        return None
    thm = lean_polymer_theorem(family)
    if family == "backtrack_4":
        comment = (
            "backtrack polymer coordination at d=4 is 15 "
            "(finite arithmetic; not a continuum gauge claim)."
        )
        stmt = "polymerBacktrack 4 = 15"
    elif family == "first_step_4":
        comment = (
            "first-step polymer coordination at d=4 is 20 "
            "(finite arithmetic; not a continuum gauge claim)."
        )
        stmt = "polymerFirstStep 4 = 20"
    else:
        comment = (
            "crude polymer coordination at d=4 is 24 "
            "(finite arithmetic; not a continuum gauge claim)."
        )
        stmt = "polymerCrude 4 = 24"
    body = (
        f"/-- {comment} -/\n"
        "theorem obligation :\n"
        f"    {stmt} :=\n"
        f"  {thm}\n"
    )
    return _wrap_polymer(body)


def _gen_sixj(cert: Mapping[str, Any]) -> str | None:
    """Apply a Check Racah 6j identity theorem."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "sixj"):
        return None
    family = payload.get("family")
    if (
        family not in LEGAL_SIXJ_FAMILIES
        or not locked_sixj_matches(payload)
        or not sixj_facts_hold(family)
    ):
        return None
    thm = lean_sixj_theorem(family)
    if family == "half_half_zero":
        comment = (
            "Racah 6j {1/2 1/2 0; 1/2 1/2 0} = -1/2 "
            "(finite rational identity; not a continuum gauge claim)."
        )
        stmt = "sixjHalfHalfZero = -1 / 2"
    else:
        comment = (
            "all-1/2 6j vanishes "
            "(finite rational identity; not a continuum gauge claim)."
        )
        stmt = "sixjAllHalf = 0"
    body = (
        f"/-- {comment} -/\n"
        "theorem obligation :\n"
        f"    {stmt} :=\n"
        f"  {thm}\n"
    )
    return _wrap_sixj(body)


def _gen_haar(cert: Mapping[str, Any]) -> str | None:
    """Apply the Check Weyl-prefactor identity theorem."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "haar_volume"):
        return None
    family = payload.get("family")
    if (
        family not in LEGAL_HAAR_FAMILIES
        or not locked_haar_matches(payload)
        or not haar_facts_hold(family)
    ):
        return None
    thm = lean_haar_theorem(family)
    if family == "su3_dim_3_0":
        comment = (
            "SU(3) Weyl dimension (3,0)=10 "
            "(finite arithmetic; not a continuum Haar theorem)."
        )
        stmt = "su3Dim 3 0 = 10"
    else:
        comment = (
            "Weyl volume prefactor 6*4=24 "
            "(finite arithmetic; not a continuum Haar theorem)."
        )
        stmt = "haarWeylPrefactor = 24"
    body = (
        f"/-- {comment} -/\n"
        "theorem obligation :\n"
        f"    {stmt} :=\n"
        f"  {thm}\n"
    )
    return _wrap_haar(body)


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
    ("nk_existence", _gen_nk_existence),
    ("enclosure_trace", _gen_enclosure_trace),
    ("named_zero", _gen_named_zero),
    ("compact_box", _gen_compact_box),
    ("casimir", _gen_casimir),
    ("polymer", _gen_polymer),
    ("sixj", _gen_sixj),
    ("haar_volume", _gen_haar),
    ("sign", _gen_sign),
)


def generate_obligation(cert: Mapping[str, Any]) -> str | None:
    r"""Return Lean source discharging ``cert``'s finite rational obligation, or ``None``.

    Tried in order: positive-definite pivots, the Newton-Kantorovich radii
    polynomial, the Krawczyk test, exact tower coefficients, planted NK
    existence, a planted enclosure trace, a named unique-zero plant, a
    compact-box residual or finite-matrix gap, a named Casimir identity,
    a named polymer-coordination identity, a named Racah 6j identity,
    a named Weyl-volume prefactor,
    then the enclosed-quantity
    sign (interval / PDE finite margin / Taylor-model centre value / raw
    ``lo``/``hi`` mapping).  Every emitted
    obligation is first re-derived exactly over ``ℚ`` (or exact ``int``
    coefficients for the tower); a payload that does not reproduce the claimed
    fact yields ``None``.
    """
    for _label, generator in _GENERATORS:
        source = generator(cert)
        if source is not None:
            return source
    return None


def classify_obligation(cert: Mapping[str, Any]) -> str | None:
    """Return the obligation *class* the bridge would discharge for ``cert``.

    One of ``"positive_definite"``, ``"radii_polynomial"``, ``"krawczyk"``,
    ``"tower_coeffs"``, ``"nk_existence"``, ``"enclosure_trace"``,
    ``"named_zero"``, ``"compact_box"``, ``"casimir"``, ``"polymer"``,
    ``"sixj"``, ``"haar_volume"`` or ``"sign"`` -- whichever
    generator first yields Lean, re-derived exactly -- or ``None`` when the
    certificate carries no Mathlib-checkable finite obligation.  Runs no Lean:
    it is the cheap triage the loop driver uses before invoking ``lake``, and
    it agrees with :func:`generate_obligation` by construction (the same
    ordered registry).
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
