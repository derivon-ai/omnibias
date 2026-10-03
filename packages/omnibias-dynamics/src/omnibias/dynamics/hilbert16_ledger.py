# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""A machine-checked Hilbert-16 obligation ledger (item 4 of the plan).

[HILBERT16-PROGRAM.md](../../../../HILBERT16-PROGRAM.md) records an **exact
counterexample** blocking research gate G1 (for ``X'=-eps*s*X, Y'=r*Y`` no
fixed decay exponent bounds the kappa-sensitivity uniformly as ``s/r -> 0``),
names G1 as failed, and lists the dated Design-Roussarie-Rousseau (DRR) case
ledger with cases open in the published literature. This module does **not**
attempt to close any of that. It follows
:mod:`omnibias.core.proof.obligations.convergence_ledger`'s discipline
instead: **every** gate G1-G6, every dated DRR case, and the Part-A
(22-oval octic) obligation is one explicit :class:`H16Obligation` with a
declared status and ``external_premises``, and the parent flags
(``full_hilbert16_solved``, ``hilbert16_part_a_solved``,
``hilbert16_part_b_quadratic_solved``) are **derived** -- earned only when
every relevant entry is ``DISCHARGED`` with empty premises, never stamped by
hand.

``DISCHARGED_LOCAL_SCOPE`` is a status distinct from ``DISCHARGED``: this
repository's genuinely new narrow results (a sealed resonant-normal-form
conjugacy identity for one field, a computed Bautin basis for one family, a
proved collar-membership agreement on one graphic's ``[delta, delta0]``) earn
that local status, recorded here for visibility, but :func:`derived_parent_flags`
never counts it toward a parent claim -- a sealed local win in this repo can
never leak into "Hilbert's 16th problem is solved."

As of this ledger every entry is ``BLOCKED`` or ``CONDITIONAL``: **every
derived parent flag is false**, by construction, on every ledger this module
ships.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.realization_replay import source_digest

__all__ = [
    "DRR_CASES",
    "DRR_PUBLISHED_CORPUS",
    "GATE_NAMES",
    "QUADRATIC_GATE_NAMES",
    "H16Ledger",
    "H16LedgerCertificate",
    "H16Obligation",
    "H16Status",
    "LedgerCheckResult",
    "certify_h16_ledger",
    "check_ledger",
    "default_h16_ledger",
    "derived_parent_flags",
    "enrich_ledger_with_campaign_evidence",
    "payload_earns_parent_claim",
    "verify_h16_ledger",
]

#: A local, single-instance result (never leaks into a parent claim) versus a
#: genuinely general, empty-premise discharge of the named obligation.
H16Status = Literal["BLOCKED", "CONDITIONAL", "DISCHARGED_LOCAL_SCOPE", "DISCHARGED"]

PAYLOAD_TYPE = "hilbert16_ledger"

PARENT_CLAIM_KEYS: frozenset[str] = frozenset(
    {
        "full_hilbert16_solved",
        "hilbert16_part_a_solved",
        "hilbert16_part_b_quadratic_solved",
        "h16_uniform_finiteness_proved",
    }
)

GATE_NAMES: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6a", "G6b")
QUADRATIC_GATE_NAMES: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6a")
DRR_PUBLISHED_CORPUS: str = "drr_published_corpus"

#: Subscripts/superscripts retained exactly; ``I_2^1`` is not ``I_12^1``.
DRR_CASES: tuple[str, ...] = (
    "I_2^1",
    "I_4^1",
    "I_12^1",
    "I_13^1",
    "I_14^1",
    "I_6b^1",
    "H_13^3",
    "DI_2b",
    "H_14^3",
    "DF_1a",
    "DF_1b",
    "DF_2a",
    "DF_2b",
    "DH_1",
    "DH_2",
)

@dataclass(frozen=True)
class H16Obligation:
    """One named obligation: a gate, a DRR case, or a Part-A/B target.

    ``evidence_digest`` optionally names a certificate digest in this
    repository backing the entry (e.g. a
    :class:`~omnibias.dynamics.membership.CollarMembershipCertificate` seal)
    -- purely informational, never itself sufficient to change ``status``.
    """

    name: str
    statement: str
    status: H16Status
    external_premises: tuple[str, ...] = ()
    evidence_digest: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.statement:
            raise ValueError("an H16Obligation needs a name and a statement")
        if self.status not in ("BLOCKED", "CONDITIONAL", "DISCHARGED_LOCAL_SCOPE", "DISCHARGED"):
            raise ValueError(f"unknown H16 status {self.status!r}")
        object.__setattr__(self, "external_premises", tuple(self.external_premises))

    @property
    def fully_discharged(self) -> bool:
        """``True`` iff genuinely, generally proved with no external premise."""
        return self.status == "DISCHARGED" and not self.external_premises

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "statement": self.statement,
            "status": self.status,
            "external_premises": list(self.external_premises),
            "evidence_digest": self.evidence_digest,
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> H16Obligation:
        return cls(
            name=str(raw["name"]),
            statement=str(raw["statement"]),
            status=raw["status"],
            external_premises=tuple(str(p) for p in raw.get("external_premises", ())),
            evidence_digest=str(raw.get("evidence_digest", "")),
            source=str(raw.get("source", "")),
        )


@dataclass(frozen=True)
class H16Ledger:
    """The full obligation set: gates, DRR cases, and Part A/B targets."""

    entries: tuple[H16Obligation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        if not self.entries:
            raise ValueError("an H16Ledger needs at least one obligation")
        names = [entry.name for entry in self.entries]
        if len(names) != len(set(names)):
            raise ValueError("H16Ledger entry names must be unique")

    def by_name(self, name: str) -> H16Obligation:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def gates(self) -> tuple[H16Obligation, ...]:
        return tuple(e for e in self.entries if e.name in GATE_NAMES)

    def gates_quadratic(self) -> tuple[H16Obligation, ...]:
        return tuple(e for e in self.entries if e.name in QUADRATIC_GATE_NAMES)

    def drr_cases(self) -> tuple[H16Obligation, ...]:
        return tuple(e for e in self.entries if e.name in DRR_CASES)

    def drr_corpora(self) -> tuple[H16Obligation, ...]:
        return tuple(e for e in self.entries if e.name == DRR_PUBLISHED_CORPUS)

    def part_a(self) -> tuple[H16Obligation, ...]:
        return tuple(e for e in self.entries if e.name.startswith("part_a"))

    def phases(self) -> tuple[H16Obligation, ...]:
        from omnibias.dynamics.hilbert16_uniform_ledger import PHASE_NAMES

        return tuple(e for e in self.entries if e.name in PHASE_NAMES)

    def adversarial(self) -> tuple[H16Obligation, ...]:
        from omnibias.dynamics.hilbert16_uniform_ledger import ADVERSARIAL_NAMES

        return tuple(e for e in self.entries if e.name in ADVERSARIAL_NAMES)

    @property
    def digest(self) -> str:
        return source_digest(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {"entries": [entry.to_payload() for entry in self.entries]}

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> H16Ledger:
        return cls(tuple(H16Obligation.from_mapping(e) for e in raw["entries"]))


@dataclass(frozen=True)
class LedgerCheckResult:
    """Exact status of a ledger: the aggregate plus every blocking entry."""

    status: Literal["BLOCKED", "CONDITIONAL", "PROVED"]
    blocking: tuple[H16Obligation, ...]

    def to_payload(self) -> dict[str, Any]:
        return {"status": self.status, "blocking": [entry.to_payload() for entry in self.blocking]}


def check_ledger(ledger: H16Ledger) -> LedgerCheckResult:
    """Decide the ledger exactly: ``PROVED`` only if every entry fully discharges.

    ``BLOCKED`` if any entry is ``BLOCKED``; else ``CONDITIONAL`` if any entry
    is ``CONDITIONAL`` or ``DISCHARGED`` with nonempty premises (a
    ``DISCHARGED_LOCAL_SCOPE`` entry alone does **not** block -- it is a
    genuine local result, just not a general one -- but it also never turns
    the aggregate ``PROVED`` on its own).
    """
    blocked = [e for e in ledger.entries if e.status == "BLOCKED"]
    conditional = [
        e
        for e in ledger.entries
        if e.status == "CONDITIONAL" or (e.status == "DISCHARGED" and e.external_premises)
    ]
    local_only = [e for e in ledger.entries if e.status == "DISCHARGED_LOCAL_SCOPE"]
    if blocked:
        return LedgerCheckResult("BLOCKED", tuple(blocked))
    if conditional or local_only:
        return LedgerCheckResult("CONDITIONAL", tuple((*conditional, *local_only)))
    return LedgerCheckResult("PROVED", ())


def derived_parent_flags(ledger: H16Ledger) -> dict[str, bool]:
    """Parent flags earned only by an all-fully-discharged, matching entry set."""
    gates_quad_ok = bool(ledger.gates_quadratic()) and all(
        e.fully_discharged for e in ledger.gates_quadratic()
    )
    gates_full_ok = bool(ledger.gates()) and all(e.fully_discharged for e in ledger.gates())
    cases_ok = bool(ledger.drr_cases()) and all(e.fully_discharged for e in ledger.drr_cases())
    corpora_ok = bool(ledger.drr_corpora()) and all(e.fully_discharged for e in ledger.drr_corpora())
    part_a_entries = ledger.part_a()
    part_a_ok = bool(part_a_entries) and all(e.fully_discharged for e in part_a_entries)
    part_b_ok = gates_quad_ok and cases_ok and corpora_ok
    from omnibias.dynamics.hilbert16_uniform_ledger import (
        UNIFORM_PARENT_FLAG,
        uniform_finiteness_discharged,
    )

    uniform_ok = uniform_finiteness_discharged(ledger.entries)
    return {
        "hilbert16_part_a_solved": part_a_ok,
        "hilbert16_part_b_quadratic_solved": part_b_ok,
        "full_hilbert16_solved": part_a_ok and part_b_ok and gates_full_ok and uniform_ok,
        UNIFORM_PARENT_FLAG: uniform_ok,
    }


def payload_earns_parent_claim(payload: dict[str, Any], key: str) -> bool:
    """``True`` iff a serialized ledger payload genuinely earns parent flag ``key``.

    Guards against a stored certificate forging a parent claim: this
    re-derives the flag from the entries themselves rather than trusting a
    stored boolean.
    """
    if key not in PARENT_CLAIM_KEYS:
        return False
    if payload.get("type") != PAYLOAD_TYPE:
        return False
    ledger_raw = payload.get("ledger")
    if not isinstance(ledger_raw, dict):
        return False
    try:
        ledger = H16Ledger.from_mapping(ledger_raw)
    except (KeyError, ValueError):
        return False
    return derived_parent_flags(ledger).get(key, False)


@dataclass(frozen=True)
class H16LedgerCertificate:
    """A sealed snapshot of the ledger's decision and derived parent flags."""

    ledger: H16Ledger
    result: LedgerCheckResult
    parent_flags: dict[str, bool]
    source_digest: str
    seal: Cert


def certify_h16_ledger(ledger: H16Ledger) -> H16LedgerCertificate:
    """Seal the ledger's exact decision and derived (never asserted) parent flags."""
    result = check_ledger(ledger)
    flags = derived_parent_flags(ledger)
    seal = make_certificate(
        claim="Hilbert-16 obligation ledger: exact status replay, parent flags derived not asserted.",
        payload={
            "type": PAYLOAD_TYPE,
            "source_digest": ledger.digest,
            "ledger": ledger.to_payload(),
            "result": result.to_payload(),
            "parent_flags": flags,
        },
        honesty={
            **flags,
            "dulac_truncated_model_only": True,
            "physical_return_membership_proved": False,
            "graphic_finite_cyclicity_proved": False,
            "drr_case_closed": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return H16LedgerCertificate(ledger, result, flags, ledger.digest, seal)


def verify_h16_ledger(certificate: H16LedgerCertificate) -> bool:
    """Replay the ledger decision and parent-flag derivation from source."""
    if (
        certificate.source_digest != certificate.ledger.digest
        or not verify_certificate_digest(certificate.seal)
    ):
        return False
    try:
        expected = certify_h16_ledger(certificate.ledger)
    except (KeyError, TypeError, ValueError):
        return False
    return expected == certificate


def _gate_entries() -> tuple[H16Obligation, ...]:
    source = "packages/omnibias-dynamics/HILBERT16-PROGRAM.md"
    return (
        H16Obligation(
            "G1",
            "A finite weighted-chart description of the actual confluent singular "
            "passage: no-root, double-root, and first-root limits; complete "
            "physical first-hit domains; uniform variational remainder bounds; "
            "endpoint matching.",
            "BLOCKED",
            (
                "tracked sep^2 * outgoing-factor product absorbs the "
                "super-small W-ratio as a first-derivative obstruction; "
                "Gronwall sigma*kappa is not the sep=0 leading derivative "
                "(exact fold I-map dx/dkappa ~ r/kappa^2); remaining holes "
                "are a Cauchy bound on Z at (V,eps)~(0,0) from the actual "
                "embedding (inner relative Z remainder is O(eps)), physical C2 "
                "of log D' off the lifted map (fold I-map Z_x is "
                "local_fold_z_x; sep>0 remains; "
                "unfrozen identities are local_z_x_gap, holomorphic "
                "Z_v bound is local_z_v_bound, slow-line Z_V chain "
                "and fold holomorphic Z_v are local_z_slow_v), outgoing "
                "height-section first-hit on chart O (L=1/n, r1->0; the "
                "x-corridor is local_outgoing_corridor, the restored "
                "T_e=Theta(eps^2) hypotheses are local_post_corridor, and "
                "the C=0 T-h envelope is local_height_envelope, the "
                "C=2 leading |q| ratio is local_q_ratio_c2, and "
                "k=1+O(eps) on C=0 is local_k_zeta_remainder, and a "
                "rectangular Z majorant on L in [0,1] including L=0 is "
                "local_kill_zeta, and a cancelled-N holomorphic Z bound "
                "with 2 eps |V| |Z| < 1 on the slow line is "
                "local_cancelled_n, and C!=0 ell/V mixing with "
                "|g_h|=O(nu^2) is local_height_mix, and the C=0 "
                "actual-versus-comparison T_h gap is local_orbit_th, and "
                "the comparison-bootstrap T-h integral is local_th_integral, "
                "and a cubic (V,h) Lohner prefix plus V=-1/4 first-hit is "
                "local_vh_orbit, and matching-chart E_out first-hit of "
                "x=rho/nu under V=-eps x is local_e_out_section, and a "
                "finite shrinking-eps pack n in {16,20,25} is "
                "local_e_out_eps, and a kill-line O(1/eps^3) comparison "
                "speed bound is local_e_out_speed, and an incoming "
                "GRAZING O(1/eps^3) comparison speed bound is "
                "local_e_sigma_speed, and an incoming V=1/4 first-hit "
                "on the reverse cubic is local_e_sigma_in, and a "
                "declared-point E_sigma first-hit at (V,h)=(3/4,1/4) is "
                "local_e_sigma_hit, and a comparison GRAZING E_sigma "
                "zero from V=0 is local_e_sigma_from0, and a uniform "
                "cancelled-height comparison on eps in [0, 1/8] is "
                "local_e_sigma_unif, and an orbit-aligned E_sigma "
                "hit from (1/4,1/40) inside the V=1/4 wall box is "
                "local_e_sigma_wall, and a wall-box h-interval "
                "E_sigma cover of [1/50, 4/125] at V=1/4 is "
                "local_e_sigma_box, and an L=0 whole-wall h-span "
                "E_sigma cover of [19/1000, 1/25] at V=1/4 is "
                "local_e_sigma_span, and an L-pack wall-span "
                "E_sigma cover of [17/1000, 7/200] at V=1/4 on "
                "L in {9/25, 1/16} is "
                "local_e_sigma_pack, and a shrinking-eps aligned "
                "E_sigma pack n in {16,20,25} is local_e_sigma_eps, "
                "and a one-shot Lohner E_sigma from V=0 at eps=1/16 "
                "is local_e_sigma_oneshot, and a shrinking-eps "
                "one-shot pack n in {16,20,25} is "
                "local_e_sigma_oneshot_eps, and a compact aligned "
                "parametric-eps E_sigma cover of [1/25, 1/16] is "
                "local_e_sigma_eps_span, and a lower aligned "
                "parametric-eps E_sigma cover of [1/64, 1/16] is "
                "local_e_sigma_eps_lo, and kill-line Stage-B height "
                "inflation is local_stage_b, and a kill-line "
                "Stage-A shrinking-rectangle wall is "
                "local_stage_a, and a kill-line chi_b "
                "threshold is local_chi_b, and kill-line dx_e "
                "leading factors are local_dx_e_leading, and a "
                "uniform-in-chi dx_e majorant on the chi_b compact "
                "is local_dx_e_unif, and a kill-line Stage-C "
                "a_min floor is local_stage_c, and a kill-line "
                "Stage-C exit energy is local_stage_c_exit, and a "
                "kill-line Stage-C leading T_h floor is "
                "local_stage_c_th, and a kill-line Stage-C start "
                "gap at y_1=1 is local_stage_c_gap, and a kill-line "
                "Stage-C C=0 T envelope is local_stage_c_env, and a "
                "kill-line Stage-C C=2 integrating factor is "
                "local_stage_c_if, and a kill-line Stage-C C=2 T(h) "
                "majorant is local_stage_c_int, and a kill-line "
                "Stage-C C=2 lower T(h) envelope is "
                "local_stage_c_lo, and a kill-line Stage-C C=2 "
                "tight T(h) ratio is local_stage_c_k, and a "
                "kill-line Stage-C C=2 T-h bootstrap is "
                "local_stage_c_boot, and a kill-line Stage-C "
                "continuation rectangle is local_stage_c_rect, and a "
                "kill-line Stage-C comparison first-hit of h=1 is "
                "local_stage_c_hit, and a kill-line Stage-C comparison "
                "first-hit of E_out is local_stage_c_sec, and a "
                "kill-line Stage-C Lohner first-hit of matching-chart "
                "x=4 from Stage-C start is local_stage_c_oneshot, and a "
                "shrinking-eps Stage-C Lohner pack n in {16, 20, 25} "
                "is local_stage_c_oneshot_eps, and a parametric-eps "
                "Stage-C Lohner cover of [23/400, 1/16] is "
                "local_stage_c_eps_span, and a chart-O matching-chart "
                "Lohner pack at r1 in {1/4, 1/8, 0} is "
                "local_stage_c_origin, and a parametric-sep "
                "matching-chart Lohner cover of [3/2, 2] is "
                "local_stage_c_origin_span, and a nearer-interface "
                "matching-chart Lohner cover of [7/4, 2] from x=1/8 is "
                "local_stage_c_origin_iface, and a nearer-interface "
                "matching-chart Lohner cover of [15/8, 2] from x=1/16 is "
                "local_stage_c_origin_near, and a nearer-interface "
                "matching-chart Lohner cover of [31/16, 2] from x=1/32 is "
                "local_stage_c_origin_x32, and a uniform-in-r1 "
                "comparison first-hit on eps in [1/32, 1/16] is "
                "local_stage_c_compare, and a comparison "
                "first-hit for every eps in (0, 1/16] and every r1 in "
                "[0, 1] is local_stage_c_uniform, and a comparison "
                "first-hit from every start in (0, 1/2] is "
                "local_stage_c_interface, and a kill-line "
                "sep*S_pre bound on every sep in (0, 1] is "
                "local_sep_spre, and dx_e leading factors on "
                "lambda1 in [-4, -2] are local_dx_e_off, and "
                "dx_e leading factors for every lambda1 <= -2 are "
                "local_dx_e_ray, and dx_e leading factors for every "
                "lambda1 in [-3/2, -2) are local_dx_e_near, and "
                "dx_e leading factors for every lambda1 in (-3/2, 0) "
                "are local_dx_e_open, and the intrinsic eta-section "
                "overlap obstruction is local_weighted_section, and the "
                "one-scale frozen-section no-go is "
                "local_quasihomogeneous_dichotomy, but "
                "the outgoing "
                "height-section first-hit on chart O, the uniform Lohner "
                "first-hit for every eps, and "
                "complete first-hit",
                "X'=-eps*s*X, Y'=r*Y has no fixed decay exponent bounding "
                "kappa-sensitivity uniformly as s/r -> 0 (exact counterexample; "
                "chi-scale replaces frozen gamma, not a cyclicity obstruction)",
            ),
            source=source,
        ),
        H16Obligation(
            "G2",
            "Identity-aware displacement calculus: exact center/Bautin "
            "generators for actual (not merely declared) return maps; a "
            "proved class closed under composition, differentiation, and "
            "division, with finite termination.",
            "BLOCKED",
            (
                "this repository's Bautin engine (omnibias.dynamics.bautin) "
                "computes the ideal for a declared normal-form family, not "
                "for the actual singular return map of an arbitrary graphic",
                "local_bautin_stabilization_barrier derives V1 through V4 "
                "for the normalized quadratic family and proves finite "
                "membership through degree ten, but an exact formal "
                "continuation outside the ideal shows that no finite jet "
                "alone proves the all-orders tail",
                "the known function-class closure counterexample (Yeung, "
                "2025) is not yet survived by a proposed calculus",
            ),
            source=source,
        ),
        H16Obligation(
            "G3",
            "Bounded-format representation: jointly parameterized return and "
            "admission predicates in a uniformly controlled format (norms, "
            "analytic extensions, branches, Stokes data) sufficient for an "
            "effective o-minimal or quasianalytic zero theorem.",
            "BLOCKED",
            (
                "local_ln_format_barrier proves that the direct tau/log-W "
                "chain has growing domain and norm on the kill sequence, but "
                "no bounded-format normalized representation of the actual "
                "return family exists here",
                "local_complex_normal_flow now encloses a fixed-real-time "
                "complex flow of the cubic normal-form comparison field on "
                "one interior epsilon rectangle, and "
                "local_complex_normal_event_branch isolates one complex "
                "event-time branch with an independent real first-hit replay; "
                "local_complex_separation_cover continues that regular "
                "comparison event across sep=0 through sep=2, including the "
                "model D-C and r1=0 endpoints; "
                "local_complex_physical_e_out_cover isolates and matches the "
                "physical outgoing E_out branch on eight complex sep cells "
                "covering those endpoints; the incoming E_sigma branch, full "
                "physical quadratic return map, and exact "
                "differential-polynomial closure remain absent",
            ),
            source=source,
        ),
        H16Obligation(
            "G4",
            "Complete graphic capture: every degenerating sequence of nearby "
            "cycles has a subsequence in a chart with an open "
            "original-parameter neighborhood carrying a uniform count, "
            "including all incident sides and identity fibers.",
            "BLOCKED",
            ("no complete itinerary/endpoint capture exists for any graphic here",),
            source=source,
        ),
        H16Obligation(
            "G5",
            "Algebraic realization or obstruction: an explicit "
            "complex-nonsingular polynomial with complete topology realizing "
            "the proposed scheme, or a general obstruction covering every "
            "realization.",
            "BLOCKED",
            (
                "the shipped octic reaches 16 certified oval barriers "
                "against a Harnack upper bound of 22; the open 22-oval "
                "target is not realized",
                "local_part_a_polygon_sos validates a 22-annulus polygonal "
                "layout and a toy Positivstellensatz emptiness proof, but "
                "supplies neither a coefficient witness nor a complete "
                "semialgebraic reduction covering every realization",
            ),
            source=source,
        ),
        H16Obligation(
            "G6a",
            "Audited quadratic case inventory and assembly: every remaining "
            "Design-Roussarie-Rousseau graphic discharged with physical "
            "itinerary capture and uniform zero counts.",
            "BLOCKED",
            ("depends on G1-G5, none of which are discharged",),
            source=source,
        ),
        H16Obligation(
            "G6b",
            "Degree-controlled resolution/capture theorem and arbitrary-degree "
            "algebraic classifications beyond the audited quadratic inventory.",
            "BLOCKED",
            ("depends on G6a and the full gate set, none of which are discharged",),
            source=source,
        ),
    )


def _drr_published_corpus_entry() -> tuple[H16Obligation, ...]:
    return (
        H16Obligation(
            DRR_PUBLISHED_CORPUS,
            "Independently replay or discharge the roughly 105 quadratic "
            "graphics already published before the 15 remaining DRR ledger "
            "cases; no silent reliance on external theorems.",
            "CONDITIONAL",
            (
                "the published quadratic corpus is recorded in "
                "omnibias.dynamics.drr_audit but not independently "
                "formalized or replayed by this repository's Lean kernel",
            ),
            source="packages/omnibias-dynamics/drr_audit.py",
        ),
    )


def _drr_case_entries() -> tuple[H16Obligation, ...]:
    source = "packages/omnibias-dynamics/HILBERT16-PROGRAM.md"
    external_theorem = (
        "relies on an external published proof not independently formalized "
        "or replayed by this repository's Lean kernel"
    )
    return (
        H16Obligation(
            "I_2^1",
            "Finite cyclicity of the I_2^1 saddle-node-at-infinity graphic.",
            "BLOCKED",
            ("Huzak-Kristiansen (2026) names its cyclicity treatment as work in progress",),
            source=source,
        ),
        H16Obligation(
            "I_4^1",
            "Finite cyclicity of the I_4^1 saddle-node-at-infinity graphic.",
            "BLOCKED",
            ("Huzak-Kristiansen (2026) names its cyclicity treatment as work in progress",),
            source=source,
        ),
        H16Obligation(
            "I_12^1",
            "Finite cyclicity of the I_12^1 nilpotent-saddle graphic.",
            "CONDITIONAL",
            (external_theorem,),
            source="Rousseau-Shan-Zhu, 2016 (arXiv:1502.00689)",
        ),
        H16Obligation(
            "I_13^1",
            "Finite cyclicity of the I_13^1 nilpotent-saddle graphic.",
            "CONDITIONAL",
            (external_theorem,),
            source="Rousseau-Shan-Zhu, 2016 (arXiv:1502.00689)",
        ),
        H16Obligation(
            "I_14^1",
            "Finite cyclicity of the I_14^1 graphic.",
            "CONDITIONAL",
            (external_theorem,),
            source="Roussarie-Rousseau, 2015 (arXiv:1506.07104)",
        ),
        H16Obligation(
            "I_6b^1",
            "Finite cyclicity of the I_6b^1 graphic.",
            "BLOCKED",
            ("the 2015 theorem treats only the boundary blown-up limit periodic set",),
            source="Roussarie-Rousseau, 2015 (arXiv:1506.07104)",
        ),
        H16Obligation(
            "H_13^3",
            "Finite cyclicity of the H_13^3 graphic.",
            "BLOCKED",
            ("the 2015 theorem treats only the boundary blown-up limit periodic set",),
            source="Roussarie-Rousseau, 2015 (arXiv:1506.07104)",
        ),
        H16Obligation(
            "DI_2b",
            "Finite cyclicity of the DI_2b graphic.",
            "BLOCKED",
            ("the 2015 theorem treats only the boundary blown-up limit periodic set",),
            source="Roussarie-Rousseau, 2015 (arXiv:1506.07104)",
        ),
        H16Obligation(
            "H_14^3",
            "Finite cyclicity of the H_14^3 graphic.",
            "BLOCKED",
            (
                "Lu's arXiv:2607.13785v3 claim is a preprint, not an "
                "independently established theorem in this dossier",
            ),
            source="Lu, arXiv:2607.13785v3 (preprint claim)",
        ),
        H16Obligation(
            "DF_1a",
            "Finite cyclicity of the DF_1a graphic.",
            "CONDITIONAL",
            (external_theorem,),
            source="Huzak, 2018 (doi:10.3934/cpaa.2018063)",
        ),
        H16Obligation(
            "DF_2a",
            "Finite cyclicity of the DF_2a graphic.",
            "CONDITIONAL",
            (external_theorem,),
            source="Huzak, 2018 (doi:10.3934/cpaa.2018063)",
        ),
        H16Obligation(
            "DF_1b",
            "Finite cyclicity of the DF_1b graphic.",
            "BLOCKED",
            ("explicitly open in Huzak, 2018",),
            source="Huzak, 2018 (doi:10.3934/cpaa.2018063)",
        ),
        H16Obligation(
            "DF_2b",
            "Finite cyclicity of the DF_2b graphic.",
            "BLOCKED",
            ("explicitly open in Huzak, 2018",),
            source="Huzak, 2018 (doi:10.3934/cpaa.2018063)",
        ),
        H16Obligation(
            "DH_1",
            "Finite cyclicity of the DH_1 graphic.",
            "BLOCKED",
            ("explicitly open in Huzak, 2018",),
            source="Huzak, 2018 (doi:10.3934/cpaa.2018063)",
        ),
        H16Obligation(
            "DH_2",
            "Finite cyclicity of the DH_2 graphic.",
            "BLOCKED",
            ("explicitly open in Huzak, 2018",),
            source="Huzak, 2018 (doi:10.3934/cpaa.2018063)",
        ),
    )


def _part_a_entries() -> tuple[H16Obligation, ...]:
    source = "packages/omnibias-dynamics/HILBERT16-ALGEBRAIC-TARGET.md"
    return (
        H16Obligation(
            "part_a_22_oval_wide_deep",
            "Construct or obstruct a smooth real octic realizing "
            "``4 + 1<2 + 1<14>>`` with complete Harnack count 22.",
            "BLOCKED",
            (
                "no explicit rational polynomial with complete_real_scheme "
                "and the wide/deep (19,3) rooted tree has been found",
            ),
            source=source,
        ),
        H16Obligation(
            "part_a_22_oval_sibling",
            "Construct or obstruct a smooth real octic realizing "
            "``14 + 1<2 + 1<4>>`` with complete Harnack count 22.",
            "BLOCKED",
            (
                "no explicit rational polynomial with complete_real_scheme "
                "and the sibling (19,3) rooted tree has been found",
            ),
            source=source,
        ),
        H16Obligation(
            "part_a_octic_four_tcurve_excluded",
            "Construct or obstruct the four algebraically open maximal octic "
            "schemes excluded from T-curves but not yet realized.",
            "BLOCKED",
            ("the four T-curve-excluded open schemes remain unrealized",),
            source="packages/omnibias-dynamics/HILBERT16-ALGEBRAIC-PROGRAM.md",
        ),
        H16Obligation(
            "part_a_degree_8_archive",
            "Classify every nonempty degree-8 real scheme in the Geiselmann "
            "archive (2,367 schemes): realize or obstruct each.",
            "BLOCKED",
            ("only bounded patchwork searches have been run; archive not exhausted",),
            source="packages/omnibias-dynamics/HILBERT16-ALGEBRAIC-PROGRAM.md",
        ),
        H16Obligation(
            "part_a_arbitrary_degree",
            "Hilbert's 16th problem Part A in full generality: classify real "
            "algebraic curve topologies for every degree.",
            "BLOCKED",
            ("arbitrary-degree classification is open beyond the octic frontier",),
            source="packages/omnibias-dynamics/HILBERT16-ALGEBRAIC-PROGRAM.md",
        ),
    )


def _local_scope_entries() -> tuple[H16Obligation, ...]:
    """Genuinely new, strictly local results from this plan. Never a parent input."""
    return (
        H16Obligation(
            "local_resonant_normal_form",
            "A sealed truncated Poincare-Dulac conjugacy identity at a "
            "single declared hyperbolic saddle with rational eigenvalues.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.saddle_normal_form.certify_resonant_normal_form",
        ),
        H16Obligation(
            "local_bautin_basis",
            "A computed reduced Groebner basis of the focal-value ideal for "
            "one declared quadratic family, matching Bautin's classical "
            "basis length of three.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.bautin.bautin_basis",
        ),
        H16Obligation(
            "local_collar_membership",
            "A sound interval agreement between a declared Dulac model and "
            "a field-derived corner expansion on a collar bounded away from "
            "the corner, for one declared graphic.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.membership.certify_collar_membership",
        ),
        H16Obligation(
            "local_df2a_declared_replay",
            "Declared interior DF_2a displacement model reproduces cyclicity "
            "<= 3 with certified entry-exit surrogate and compactification.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.df2a.reproduce_df2a_cyclicity",
        ),
        H16Obligation(
            "local_drr_graphic_inventory",
            "Dated 121-graphic DRR inventory with published/conditional/open "
            "status recorded for every graphic.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.drr_audit.audit_report",
        ),
        H16Obligation(
            "local_weighted_section",
            "Exact negative assessment of the intrinsic eta-section: "
            "ordinary q=sep^2 hit-time derivatives grow like q^-1 and "
            "q^-2 at D intersect C, while the chart-O matching speed "
            "vanishes with r1. This falsifies H1 as a G1 discharge, not "
            "every possible closing map.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.weighted_section.report",
        ),
        H16Obligation(
            "local_quasihomogeneous_dichotomy",
            "No single positive scale, including every rational monomial "
            "sigma=eps^a sep^b, separately bounds both the event factor "
            "and the W-ratio to a frozen eps^N section. A moving sep^2 "
            "section is a counterexample to excluding all weighted atlases.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.quasihomogeneous_dichotomy.report",
        ),
        H16Obligation(
            "local_ln_format_barrier",
            "Every finite direct tau/log-W truncation has an exact two-function "
            "LN chain with fixed degree and coefficients, but its analytic "
            "outer radius and chain sup norm grow linearly on the corrected "
            "kill sequence. Normalized zero-equivalent chains remain open.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.ln_format_barrier.report",
        ),
        H16Obligation(
            "local_complex_normal_flow",
            "An outward-rounded complex Taylor integrator encloses the "
            "fixed-real-time flow of the cubic Hilbert-XVI normal-form "
            "comparison field for every epsilon in one complex rectangle. "
            "A complex implicit first-hit, the full physical quadratic "
            "return map, and Log-Noetherian closure remain open.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.complex_normal_flow.report",
        ),
        H16Obligation(
            "local_complex_normal_event_branch",
            "Parametric complex interval Newton isolates one event-time "
            "branch of the cubic normal-form comparison field uniformly on "
            "a complex epsilon rectangle; a source-derived real stopped-event "
            "certificate independently proves the first transverse crossing. "
            "The full physical quadratic return family and LN closure remain "
            "open.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.complex_event_branch.report",
        ),
        H16Obligation(
            "local_complex_separation_cover",
            "One regular V-event branch of the cubic comparison field is "
            "isolated on a single complex sep rectangle containing sep=0 and "
            "sep=2, while an exact-source real stopped-event replay proves "
            "the first crossing for every sep in [0,2]. The singular physical "
            "entry/exit continuations and overlap matching remain open.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.complex_separation_cover.report",
        ),
        H16Obligation(
            "local_complex_physical_e_out_cover",
            "Eight rational complex sep cells isolate the physical outgoing "
            "E_out event branch of the cubic comparison field from the D-C "
            "endpoint sep=0 through the chart-O endpoint sep=2. Uniqueness "
            "matches adjacent cells at common boundaries, and exact-source "
            "real stopped events independently prove first crossing. The "
            "incoming E_sigma branch and full quadratic return remain open.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.complex_physical_e_out.report",
        ),
        H16Obligation(
            "local_abelian_return_transfer",
            "Exact rational threshold arithmetic transfers a supplied Abelian "
            "zero cover to a return displacement with a supplied uniform "
            "second-order remainder. The open DRR graphics supply neither the "
            "Hamiltonian reduction nor that physical remainder.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.abelian_return_transfer.report",
        ),
        H16Obligation(
            "local_bautin_stabilization_barrier",
            "The normalized quadratic field has exact-Q focal generators "
            "V1,V2,V3 and a certified V4 ideal-membership witness, while a "
            "formal continuation sharing that finite prefix can leave the "
            "ideal at V5. Finite jets therefore require an independent "
            "all-orders recurrence and do not discharge G2.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.bautin_stabilization_barrier.report",
        ),
        H16Obligation(
            "local_songling_lower_bound_audit",
            "The exact rational Songling field and four published section "
            "boxes are transcribed, and binary64 interval cancellation is "
            "proved unable to resolve the 8*epsilon perturbation. The 2022 "
            "multiple-precision four-cycle theorem predates this audit and "
            "is not replayed here.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.songling_lower_bound.report",
        ),
        H16Obligation(
            "local_part_a_polygon_sos",
            "Exact rational polygonal barriers validate the wide/deep "
            "22-annulus target layout, and the SOS engine certifies a toy "
            "empty basic closed set. The open octic scheme has no complete "
            "basic-closed encoding or all-realizations symmetry reduction, "
            "so no realization or obstruction is earned.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.geometry.part_a_obstruction.audit_part_a_obstruction_route",
        ),
        H16Obligation(
            "local_entry_exit_product",
            "Exact slow-line partial-fraction leading map and the tracked "
            "sep^2 * (h_1 / (eps^3 mu sep^2))^(C eps) product identity, "
            "negative on the super-small kill sequence.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.entry_exit_leading.report",
        ),
        H16Obligation(
            "local_fold_imap",
            "Exact sep=0 slow-line I-map first derivative and leading C2 of "
            "log(dx/dkappa); Gronwall sigma*kappa is not the leading map.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.fold_leading.report",
        ),
        H16Obligation(
            "local_shrinking_root_imap",
            "Exact two-root slow-line I-map and the r1->0 remainder; "
            "outgoing first-hit of the large first-root section remains open.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.shrinking_root_leading.report",
        ),
        H16Obligation(
            "local_canonical_zeta",
            "Algebraic r=-1 slow-line zeta on the lambda0=lambda1=0 slice "
            "and the implicit-k sample at nonzero lambda1, with a Cauchy "
            "majorant for Z at (0,0) on lambda=0. The fold (L, lambda1) "
            "compact is local_fold_zeta.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.canonical_zeta.report",
        ),
        H16Obligation(
            "local_fold_zeta",
            "Cauchy majorant for Z on a declared real fold compact "
            "rstar in [1.4, 1.6] with Picard-included k; disc identities "
            "lambda1^2 = 4 L = 4 r^2. Not physical C2 or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.fold_zeta.report",
        ),
        H16Obligation(
            "local_physical_c2",
            "Frozen-Z first-log-derivative and C2 remainder identities "
            "versus the lifted fold map. Uniform-in-eps, Z_x bound, and sep>0 "
            "remain open.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.physical_c2.report",
        ),
        H16Obligation(
            "local_z_x_gap",
            "Unfrozen-Z first-log-derivative gap identities including "
            "Z_x versus the lifted fold map. A bound on Z_x, sep>0, "
            "and uniform-in-eps remain open.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.z_x_gap.report",
        ),
        H16Obligation(
            "local_z_v_bound",
            "Holomorphic Z_v identities and an Interval enclosure "
            "|Z_v|<1/4 on the cancelled-N kill compact, excluding 0. "
            "Fold I-map Z_x is local_fold_z_x. Not sep>0, first-hit, "
            "or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.z_v_bound.report",
        ),
        H16Obligation(
            "local_z_slow_v",
            "Slow-line Z_V = -Z_v/ell identities, |Z_V|<1/4 on the "
            "cancelled-N kill compact, and holomorphic |Z_v|<1/4 on "
            "the fold wall r in [1.4, 1.6]. Fold I-map Z_x is "
            "local_fold_z_x. Not sep>0, first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.z_slow_v.report",
        ),
        H16Obligation(
            "local_fold_z_x",
            "Matching-chart Z_x = Z_v eps/ell identities and an "
            "Interval enclosure |Z_x|<1/100 on the fold I-map compact "
            "r in [1.4, 1.6], eps in [0, 0.02]. Not sep>0, first-hit, "
            "or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.fold_z_x.report",
        ),
        H16Obligation(
            "local_stage_b",
            "Kill-line Stage-B height-inflation Picard inclusion with "
            "|Delta x|<1/3 on lambda1=-2, sep in (0, 1], eps in "
            "[0, 1/16], and a positive tracked exponent. Not Stage "
            "A/C, C2, first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_b.report",
        ),
        H16Obligation(
            "local_stage_a",
            "Kill-line Stage-A wall identities B_-(a) = "
            "theta(1+theta) sep^2 and B_-'(bnd) = -sep(1-2 theta) "
            "at theta=1/8, plus Interval a>1/4 and leading Psi_pre "
            "factor <1/4 on sep in [0, 1]. Not dx_e/dkappa, chi, "
            "Stage C, first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_a.report",
        ),
        H16Obligation(
            "local_chi_b",
            "Kill-line chi_b identities (1+theta)/theta = 9, "
            "decay c = 1/16, and worst-case (K+1)/r1 = 8, plus "
            "Interval sep*S_pre < 3 and chi_b < 9 on sep in "
            "[1/2^16, 1]. Not dx_e/dkappa, Stage C, first-hit, "
            "or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.chi_b.report",
        ),
        H16Obligation(
            "local_dx_e_leading",
            "Kill-line dx_e leading identities: algebraic "
            "prefactor 3/8, slope half 3/8, and threshold net "
            "floor 3/16, plus Interval prefactor <1/2 and net "
            "exponent >1/8 after the y0 log remainder on the "
            "chi_b compact. Not the uniform-in-chi bound, Stage "
            "C, first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.dx_e_leading.report",
        ),
        H16Obligation(
            "local_dx_e_unif",
            "Kill-line uniform-in-chi dx_e identities: extra "
            "coefficient 3/32 at r1 = 1/2, written c = 1/16 "
            "weaker by 1/32, and threshold lift 27/32, plus "
            "Interval C < 2 and extra > 1/16 on the chi_b "
            "compact. Not Stage C, first-hit, dx_e off the kill "
            "line, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.dx_e_unif.report",
        ),
        H16Obligation(
            "local_stage_c",
            "Kill-line Stage-C a_min identities: written "
            "a_min = 1/2, integrating factor 1/(1/4) = 4, and "
            "declared floor 1/4, plus Interval end_lo > 1/4 "
            "and 1/x < 8 on the Stage-B end box. Not an "
            "outgoing orbit, first-hit, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c.report",
        ),
        H16Obligation(
            "local_stage_c_exit",
            "Kill-line Stage-C exit identities: T_e/eps^2 = 1/8 "
            "at x = 1/2, eps y0 = 1/256, and gap room 31/256, "
            "plus Interval T_e/eps^2 in (1/16, 1) and T_e > h_e "
            "on the Stage-B end box. Not an outgoing orbit, "
            "first-hit, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_exit.report",
        ),
        H16Obligation(
            "local_stage_c_th",
            "Kill-line Stage-C T_h identities: midpoint product "
            "-sep^2/4, worst leading T_h = 3/4, and room 1/4 "
            "above 1/2, plus Interval T_h > 1/2 on the Stage-B "
            "end box at y_1 = 1. Not an outgoing orbit, "
            "first-hit, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_th.report",
        ),
        H16Obligation(
            "local_stage_c_gap",
            "Kill-line Stage-C start-gap identities: wall "
            "1/8 - 1/16 = 1/16, h_1 cube 1/4096, and exact wall "
            "T-h = 1/4096 at eps = 1/16, plus Interval "
            "(T_e - h_1)/eps^2 > 1/32 at y_1 = 1. Not an "
            "outgoing orbit, first-hit, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_gap.report",
        ),
        H16Obligation(
            "local_stage_c_env",
            "Kill-line Stage-C C=0 envelope identities: wall "
            "T_e/eps^2 = 1/8 below 1, declared c = 1/32, and "
            "c + room = 1, plus Interval "
            "(1/32)(eps^2+h) <= T(h) <= eps^2+h. Not a C!=0 "
            "orbit, first-hit, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_env.report",
        ),
        H16Obligation(
            "local_stage_c_if",
            "Kill-line Stage-C C=2 integrating-factor identities: "
            "2*3=6, 2*6=12, and edge 12(1/4-1/16)=9/4, plus "
            "Interval exponent <= 3 and (h/h_1)^{C eps} < 32 on "
            "eps in (0, 1/16]. Not T(h)<=C(eps^2+h) after the "
            "remaining integral, first-hit, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_if.report",
        ),
        H16Obligation(
            "local_stage_c_int",
            "Kill-line Stage-C C=2 T(h)-integral identities: "
            "C eps = 1/8, 1-alpha = 7/8, and slope 16/7, plus "
            "Interval T(h) <= 64 (eps^2+h). Not first-hit, C2, "
            "or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_int.report",
        ),
        H16Obligation(
            "local_stage_c_lo",
            "Kill-line Stage-C C=2 lower-envelope identities: "
            "1/2 - 1/32 = 15/32, 1 + 1/16 = 17/16, and wrapping "
            "start remainder 15/512, plus Interval "
            "T(h) >= (1/32)(eps^2+h). Not first-hit, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_lo.report",
        ),
        H16Obligation(
            "local_stage_c_k",
            "Kill-line Stage-C C=2 tight-ratio identities: "
            "6*(1/16)=3/8, 3*2=6, and 6-16/7=26/7, plus Interval "
            "T(h) <= 6 (eps^2+h) from the edge factor. Not "
            "T-h=O(eps), first-hit, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_k.report",
        ),
        H16Obligation(
            "local_stage_c_boot",
            "Kill-line Stage-C C=2 T-h bootstrap identities: "
            "1+6=7, 2*6+3=15, and 2*7*3=42, plus Interval "
            "T-h < 1 at the compact edge. Not O(eps), first-hit, "
            "C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_boot.report",
        ),
        H16Obligation(
            "local_stage_c_rect",
            "Kill-line Stage-C continuation-rectangle identities: "
            "1+1=2, 2*2=4, and (1/16)*(1/4)=1/64, plus Interval "
            "left wall < 3 and right wall > 0. Not first-hit, C2, "
            "or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_rect.report",
        ),
        H16Obligation(
            "local_stage_c_hit",
            "Kill-line Stage-C comparison first-hit identities: "
            "1/(1/64)=64, 64*3=192, and 1-1/4096=4095/4096, plus "
            "Interval time < 1024 to h=1. Not Lohner, signed-label "
            "section, chart O, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_hit.report",
        ),
        H16Obligation(
            "local_stage_c_sec",
            "Kill-line Stage-C E_out comparison first-hit identities: "
            "1/4-1/12=1/6, (1/2)/2=1/4, and 1/64+1/256=5/256, plus "
            "Interval E_out > 0 at start, dE_out/dh < 0, and h_hit < 1/8. "
            "Not Lohner, chart O, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_sec.report",
        ),
        H16Obligation(
            "local_stage_c_oneshot",
            "Kill-line Stage-C Lohner first-hit identities: "
            "(1/4)/(1/16)=4, 120*(1/20)=6, and 80*(1/20)=4, plus "
            "unique transverse first-hit of matching-chart x=4 from "
            "(x,y)=(1/4,1) on sep in {0, 3/5, 1} at eps=1/16. A short "
            "horizon of 80 steps does not certify. Not every eps, "
            "chart O, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_oneshot.report",
        ),
        H16Obligation(
            "local_stage_c_oneshot_eps",
            "Kill-line Stage-C shrinking-eps Lohner identities: "
            "(1/4)/(1/20)=5, (1/4)/(1/25)=25/4, and 160*(1/20)=8, plus "
            "unique transverse first-hit of matching-chart x=n/4 from "
            "(x,y)=(1/4,1) at eps=1/n for n in {16, 20, 25}. A short "
            "horizon of 120 steps at n=20 does not certify. Not every "
            "eps, chart O, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_oneshot_eps.report",
        ),
        H16Obligation(
            "local_stage_c_eps_span",
            "Kill-line Stage-C parametric-eps Lohner identities: "
            "23/400+1/200=1/16, 4*(1/800)=1/200, and 800*(1/16)=50, "
            "plus unique transverse first-hit of 4 eps x=1 from "
            "(x,y)=(1/4,1) on four slabs of width 1/800 covering "
            "[23/400, 1/16]. A single slab over that compact is "
            "unresolved. Not every eps, chart O, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_eps_span.report",
        ),
        H16Obligation(
            "local_stage_c_origin",
            "Kill-line matching-chart Lohner identities: "
            "1-2/2=0, 1-(3/2)/2=1/4, and 1-(7/4)/2=1/8, plus "
            "unique transverse first-hit of matching-chart x=4 from "
            "(x,y)=(1/4,1) on sep in {3/2, 7/4, 2} (r1 in {1/4, 1/8, 0}) "
            "at eps=1/16. A short horizon of 120 steps at sep=2 does "
            "not certify. Not every r1, complete first-hit on chart O, "
            "C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_origin.report",
        ),
        H16Obligation(
            "local_stage_c_origin_span",
            "Kill-line parametric-sep matching-chart Lohner identities: "
            "3/2+1/2=2, 8*(1/16)=1/2, and 16*(1/2)=8, plus unique "
            "transverse first-hit of matching-chart x=4 from "
            "(x,y)=(1/4,1) on eight slabs of width 1/16 covering "
            "[3/2, 2]. A single slab over that compact is unresolved. "
            "Not every r1, complete first-hit on chart O, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_origin_span.report",
        ),
        H16Obligation(
            "local_stage_c_origin_iface",
            "Kill-line nearer-interface matching-chart Lohner identities: "
            "7/4+1/4=2, 8*(1/32)=1/4, and 32*(1/4)=8, plus unique "
            "transverse first-hit of matching-chart x=4 from "
            "(x,y)=(1/8,1) on eight slabs of width 1/32 covering "
            "[7/4, 2]. A single slab over that compact is unresolved. "
            "Not every r1, complete first-hit on chart O, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_origin_iface.report",
        ),
        H16Obligation(
            "local_stage_c_origin_near",
            "Kill-line nearer-interface matching-chart Lohner identities: "
            "15/8+1/8=2, 8*(1/64)=1/8, and 64*(1/8)=8, plus unique "
            "transverse first-hit of matching-chart x=4 from "
            "(x,y)=(1/16,1) on eight slabs of width 1/64 covering "
            "[15/8, 2]. A single slab over that compact is unresolved. "
            "A short horizon of 160 steps at sep=2 does not certify. "
            "Not every r1, complete first-hit on chart O, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_origin_near.report",
        ),
        H16Obligation(
            "local_stage_c_origin_x32",
            "Kill-line nearer-interface matching-chart Lohner identities: "
            "31/16+1/16=2, 8*(1/128)=1/16, and 128*(1/16)=8, plus unique "
            "transverse first-hit of matching-chart x=4 from "
            "(x,y)=(1/32,1) on eight slabs of width 1/128 covering "
            "[31/16, 2]. A single slab over that compact is unresolved. "
            "A short horizon of 160 steps at sep=2 does not certify. "
            "Not every r1, complete first-hit on chart O, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_origin_x32.report",
        ),
        H16Obligation(
            "local_stage_c_compare",
            "Kill-line comparison identities: 2*1-1^2=1, (1/4)/(1/32)=8, "
            "and 310*(1/40)=31/4, plus a phase-wise Interval speed bound "
            "from (x,y)=(1/4,1) that reaches x=8 for every r1 in [0,1] "
            "and every eps in [1/32, 1/16]. Freezing y at 1 stalls. "
            "Not every eps, not a Lohner tube, not the shrinking "
            "interface, not complete first-hit on chart O, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_compare.report",
        ),
        H16Obligation(
            "local_stage_c_uniform",
            "Kill-line neck identities: 1/4+7/4=2, 70*(1/40)=7/4, and "
            "2*(4-1/4)/(1/16)=120, plus a phase-wise dy/dx bound on "
            "[1/4, 2] that keeps dx/dsigma >= eps/2 from (x,y)=(1/4,1) "
            "for every r1 in [0,1] and every eps in (0, 1/16]. The "
            "matching section x=(1/4)/eps is therefore hit, with time "
            "at most (1-eps)/(2 eps^2). Holding y at 1 stalls. Not a "
            "Lohner tube, not eps>1/16, not the shrinking interface, "
            "not complete first-hit on chart O, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_uniform.report",
        ),
        H16Obligation(
            "local_stage_c_interface",
            "Kill-line entrance identities: 1+(1/2)*(1/2-2)=1/4, "
            "60*(1/40)=3/2, and 5*(1/4)/(1/16)^2=320, plus a dy/dx "
            "bound from the worst entrance x=1/2 that keeps "
            "dx/dsigma >= eps/5 for every start in (0, 1/2], every "
            "r1 in [0,1], and every eps in (0, 1/16]. An interface "
            "r1(1+theta) that lands in (0, 1/2] is included, so "
            "chart-O sequences r1->0 are included. Holding y at 1 "
            "stalls. Not a Lohner tube, not the height-section flag "
            "on L=1/n, not eps>1/16, not complete first-hit on chart "
            "O, C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.stage_c_interface.report",
        ),
        H16Obligation(
            "local_sep_spre",
            "Kill-line identities: 2^48 * 2^{-48} = 1, 4/(1/2) = 8, "
            "and (3/16)(4 - 11/5) = 27/80, plus a dyadic-plus-tail "
            "enclosure sep*S_pre < 11/5 and chi_b <= 8 for every sep "
            "in (0, 1]. The dx_e log remainder on eps in [0, 1/16] "
            "keeps the net exponent above 1/8. ln(1/16) < -2. "
            "Feeding S = 4 into the net floor stalls. Not dx_e off "
            "the kill line, not uniform-in-chi, not Stage C, "
            "first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.sep_spre.report",
        ),
        H16Obligation(
            "local_dx_e_off",
            "Identities 1-5/8=3/8, 1/(1/2)=2, and 11/5+2=21/5, plus "
            "an Interval enclosure h(u)<11/5 on u=sep/r1 in (0, 2]. "
            "For lambda1 in [-4, -2] and sep in (0, 1], "
            "sep*S_pre < r1*(11/5), the net exponent stays above "
            "1/8, chi_b <= 21/5, and C < 2. Comparing h with 1 "
            "stalls. Not lambda1 < -4, not lambda1 in (-2, 0), "
            "not Stage C, first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.dx_e_off.report",
        ),
        H16Obligation(
            "local_dx_e_ray",
            "Identities X=2*rstar equals 2 at rstar=1, "
            "(3/8)/(2 rstar)*rstar=3/16, and 11/5+2=21/5. "
            "For every rstar>=1 and every sep in (0, 1], the net "
            "exponent stays above 1/8, chi_b<=21/5, and C<2. "
            "Dropping the rstar surplus stalls. Not lambda1 in "
            "(-2, 0), not Stage C, first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.dx_e_ray.report",
        ),
        H16Obligation(
            "local_dx_e_near",
            "Identities 3/4-5/8=1/8, 1/(1/4)=4, and 11/5+3=26/5. "
            "For every lambda1 in [-3/2, -2) and every sep in (0, 1], "
            "the net exponent stays above 1/8, chi_b<=26/5, and C<2. "
            "Dropping the rstar surplus stalls. Not lambda1 in "
            "(-3/2, 0), not Stage C, first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.dx_e_near.report",
        ),
        H16Obligation(
            "local_dx_e_open",
            "Identities 1-(5/8)*(8/5)=0, 11/5+5=36/5, and "
            "3/16-1/20=11/80. For every lambda1 in (-3/2, 0) and "
            "every sep in (0, min(1, (8/5) rstar)), the eps cap "
            "keeps the net exponent above 1/8 and C < (1/5)/a. "
            "Holding eps=1/16 at sep=1/4096 stalls. Not Stage C, "
            "first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.dx_e_open.report",
        ),
        H16Obligation(
            "local_outgoing_corridor",
            "Cleared two-root I-map from the matching interface r1(1+theta) "
            "to a compact physical x_*; r1 log r1 majorized by "
            "2 sqrt(r1)-2 r1. Not height-section first-hit, uniform a_min, "
            "or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.outgoing_corridor.report",
        ),
        H16Obligation(
            "local_post_corridor",
            "After the x-corridor, T_*=Theta(eps^2) and (h/h_e)^(C eps)->1 "
            "independently of r1; leading T_h > 1/2 at a fixed y0. Not "
            "height-section first-hit, sealed T-h=O(eps), or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.post_corridor.report",
        ),
        H16Obligation(
            "local_height_envelope",
            "C=0 comparison conserves T-h = T_e-h_e; |q|/(eps(eps^2+T)) "
            "at matching is independent of eps and bounded as r1->0. Not "
            "height-section first-hit, actual-field T-h=O(eps), or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.height_envelope.report",
        ),
        H16Obligation(
            "local_q_ratio_c2",
            "On lambda1=-2 the leading |q| ratio is <2 for every x, "
            "uniformly in r1->0. Not k=1+O(eps), zeta remainder, "
            "height-section first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.q_ratio_c2.report",
        ),
        H16Obligation(
            "local_k_zeta_remainder",
            "On C=0, A=1 the normal k is 1+O(nu) with exact O(nu^2) "
            "remainder versus the V-jet; the cubic correction past the "
            "two-root leading term is eps^4 x^3/3. Not a Z bound, "
            "T-h along the orbit, height-section first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.k_zeta_remainder.report",
        ),
        H16Obligation(
            "local_kill_zeta",
            "Cauchy majorant for Z on lambda1=-2, L in [0, 1], including "
            "the kill limit L=0. Rectangular and not small enough for "
            "C=2+delta. Not T-h along the orbit, height-section first-hit, "
            "or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.kill_zeta.report",
        ),
        H16Obligation(
            "local_cancelled_n",
            "Cancelled-N holomorphic Z on lambda1=-2, L in [0, 1], with "
            "2 eps |V| |Z| < 1 on a declared real slow-line compact. Not "
            "T-h along the orbit, C!=0 |g_h|, height-section first-hit, "
            "or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.cancelled_n.report",
        ),
        H16Obligation(
            "local_height_mix",
            "C!=0 ell/V mixing: V_v+ell=0, V_h+C nu^2 v=0, and the "
            "first-order g jet is independent of h; |ell_h|=|C| nu^2 "
            "on a declared compact with ell>0. Not T-h along the orbit, "
            "height-section first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.height_mix.report",
        ),
        H16Obligation(
            "local_orbit_th",
            "Actual-versus-comparison T_h gap splits as "
            "(k-1)+(q-C eps(T+eps^2))/h and equals k-1 at flux touching; "
            "|k-1|<=3 nu on the C=0 box. Not an integrated T-h orbit, "
            "height-section first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.orbit_th.report",
        ),
        H16Obligation(
            "local_th_integral",
            "Comparison-bootstrap integral of (T-h)_h after T<=K(eps^2+h) "
            "splits into O(eps) plus an eps^3 log majorant; < 9 eps on a "
            "declared compact. Not a Lohner-validated (V,h) orbit, "
            "height-section first-hit, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.th_integral.report",
        ),
        H16Obligation(
            "local_vh_orbit",
            "QR-Lohner prefix of the cubic (V,h) field keeps T-h < 9 eps "
            "and certify_stopped_event hits V=-1/4 on a declared matching "
            "compact. Matching-chart E_out is local_e_out_section. Not "
            "GRAZING E_sigma, uniform r1->0, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.vh_orbit.report",
        ),
        H16Obligation(
            "local_e_out_section",
            "Matching-chart E_out = V+rho+nu rho h+C nu^2 rho h^2 is the "
            "image of x=rho/nu under V=-eps x; certify_stopped_event hits "
            "it transversely on L in {9/25, 1/16, 0} including the kill "
            "limit L=0; GRAZING E_sigma=V-1+... is excluded at L=0. Not "
            "uniform eps->0 or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_out_section.report",
        ),
        H16Obligation(
            "local_e_out_eps",
            "Matching-chart E_out first-hit on the finite shrinking pack "
            "eps=1/n for n in {16, 20, 25} at L=0 inside T=n^2/8. A short "
            "horizon at n=16 does not certify. Not a uniform-in-eps "
            "theorem, GRAZING E_sigma, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_out_eps.report",
        ),
        H16Obligation(
            "local_e_out_speed",
            "Kill-line comparison F=f+4 eps^3 g has F_V=eps phi with "
            "phi(-eps)=eps^2(1+4 eps)>0 and g<0 on V in [-rho,-eps], so "
            "-Vdot >= 3 eps^3 and the hitting time of V=-rho is at most "
            "(rho-eps)/(3 eps^3) for every eps in (0, 1/16]. O(1/eps^3) "
            "comparison majorant, not Lohner for every eps, GRAZING "
            "E_sigma, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_out_speed.report",
        ),
        H16Obligation(
            "local_e_sigma_speed",
            "Incoming reverse cubic on V in [0, 1] has F decreasing "
            "because phi(0)=-2 eps(1-2 eps^2)<0 for eps in (0, 1/16], "
            "so Vdot_rev >= 4 eps^3(1+eps) and the time to cross "
            "Delta V=1 is at most 1/(4 eps^3(1+eps)). Lohner wrapping "
            "refuses a certified E_sigma first-hit. O(1/eps^3) "
            "comparison majorant, not GRAZING first-hit or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_speed.report",
        ),
        H16Obligation(
            "local_e_sigma_in",
            "Incoming reverse cubic from the GRAZING start V=0, "
            "h=4 eps^3 has unique transverse first-hit of V=1/4 on "
            "L in {9/25, 1/16, 0}. A short horizon does not certify. "
            "Lohner wrapping still refuses E_sigma near V=1. Not "
            "GRAZING first-hit or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_in.report",
        ),
        H16Obligation(
            "local_e_sigma_hit",
            "Declared incoming point (V,h)=(3/4,1/4) on the reverse "
            "cubic has unique transverse first-hit of GRAZING E_sigma "
            "on L in {9/25, 1/16, 0}. A short horizon does not "
            "certify. The GRAZING start V=0 still excludes E_sigma on "
            "this compact horizon. Not the GRAZING band from V=0 or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_hit.report",
        ),
        H16Obligation(
            "local_e_sigma_from0",
            "Comparison tube from GRAZING V=0, h=4 eps^3 isolates a "
            "unique increasing E_sigma zero on L in {9/25, 1/16, 0} "
            "at eps=1/16, V*=6/5. V*=1 does not yet change sign. "
            "Lohner wrapping still refuses certify_stopped_event "
            "from V=0. Not uniform eps or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_from0.report",
        ),
        H16Obligation(
            "local_e_sigma_unif",
            "Cancelled height majorant on V*=6/5 and eps in [0, 1/8] "
            "has E_sigma>0 and dE/dV>0 on eight Interval slabs. A "
            "single slab wraps. V*=1 does not change sign. Not a "
            "Lohner event for every eps or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_unif.report",
        ),
        H16Obligation(
            "local_e_sigma_wall",
            "Orbit-aligned point (V,h)=(1/4,1/40) lies in the "
            "certified GRAZING-from-V=0 V=1/4 return box on L in "
            "{9/25, 1/16, 0} and has unique transverse first-hit of "
            "GRAZING E_sigma. A short horizon does not certify. The "
            "GRAZING start V=0 still excludes E_sigma on this compact "
            "horizon. Not enclosure continuation of the whole h-box, "
            "not a single Lohner run from V=0, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_wall.report",
        ),
        H16Obligation(
            "local_e_sigma_box",
            "Twelve h-slabs covering [1/50, 4/125] at V=1/4 certify "
            "unique transverse GRAZING E_sigma at L=0; the interval "
            "is a declared rational sub-box of every L-pack V=1/4 "
            "wall box; the aligned slab containing h=1/40 certifies "
            "on L in {9/25, 1/16}. A single slab is unresolved. The "
            "GRAZING start V=0 still excludes E_sigma on this compact "
            "horizon. Not the whole wall h-interval, not a single "
            "Lohner run from V=0, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_box.report",
        ),
        H16Obligation(
            "local_e_sigma_span",
            "Twenty-one h-slabs covering [19/1000, 1/25] at V=1/4 "
            "certify unique transverse GRAZING E_sigma at L=0; the "
            "interval contains the whole L=0 V=1/4 wall box. A "
            "single slab is unresolved. The GRAZING start V=0 still "
            "excludes E_sigma on this compact horizon. Not the L in "
            "{9/25, 1/16} walls, not a single Lohner run from V=0, "
            "or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_span.report",
        ),
        H16Obligation(
            "local_e_sigma_pack",
            "Eighteen h-slabs covering [17/1000, 7/200] at V=1/4 "
            "certify unique transverse GRAZING E_sigma on L in "
            "{9/25, 1/16}; the interval contains both remaining "
            "L-pack V=1/4 wall boxes. A single slab is unresolved. "
            "The GRAZING start V=0 still excludes E_sigma on this "
            "compact horizon. Not a single Lohner run from V=0, "
            "or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_pack.report",
        ),
        H16Obligation(
            "local_e_sigma_eps",
            "Aligned (V,h)=(1/4,1/40) certifies unique transverse "
            "GRAZING E_sigma at eps=1/n for n in {16, 20, 25} on "
            "L=0 inside T=10; each GRAZING-from-V=0 V=1/4 return "
            "box contains that point. A short horizon at n=16 is "
            "unresolved. The GRAZING start V=0 still excludes "
            "E_sigma on this compact horizon. Not a wall-span "
            "cover at every n, not a single Lohner run from V=0, "
            "or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_eps.report",
        ),
        H16Obligation(
            "local_e_sigma_oneshot",
            "A single certify_stopped_event from GRAZING V=0, "
            "h=4 eps^3 hits unique transverse GRAZING E_sigma on "
            "L in {9/25, 1/16, 0} at eps=1/16 with step=1/4 and "
            "max_steps=280. A short horizon of 200 steps does not "
            "certify. Not uniform in eps, not Z_x C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_oneshot.report",
        ),
        H16Obligation(
            "local_e_sigma_oneshot_eps",
            "A single certify_stopped_event from GRAZING V=0, "
            "h=4 eps^3 hits unique transverse GRAZING E_sigma at "
            "eps=1/n for n in {16, 20, 25} on L=0 inside T=70, 100, "
            "250. A short horizon at n=16 does not certify. Not "
            "uniform in eps, not Z_x C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_oneshot_eps.report",
        ),
        H16Obligation(
            "local_e_sigma_eps_span",
            "Three equal eps-slabs of width 3/400 covering [1/25, 1/16] "
            "from aligned (V,h)=(1/4,1/40) certify unique transverse "
            "GRAZING E_sigma at L=0; the compact contains {1/16, 1/20, "
            "1/25}. The last slab containing eps=1/16 certifies on L in "
            "{9/25, 1/16}. A single slab over the whole compact is "
            "unresolved. The GRAZING start V=0 still excludes E_sigma "
            "on the last slab. Not a uniform-in-eps theorem for every "
            "eps, not Lohner from V=0 on this compact, not Z_x C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_eps_span.report",
        ),
        H16Obligation(
            "local_e_sigma_eps_lo",
            "Six equal eps-slabs of width 1/128 covering [1/64, 1/16] "
            "from aligned (V,h)=(1/4,1/40) certify unique transverse "
            "GRAZING E_sigma at L=0; the compact contains the previous "
            "[1/25, 1/16] span and {1/16, 1/20, 1/25, 1/32, 1/64}. The "
            "last slab containing eps=1/16 certifies on L in {9/25, "
            "1/16}. A single slab over the whole compact is unresolved. "
            "The GRAZING start V=0 still excludes E_sigma on the last "
            "slab. Not a uniform-in-eps theorem for every eps, not "
            "Lohner from V=0 on this compact, not Z_x C2, or G1.",
            "DISCHARGED_LOCAL_SCOPE",
            (),
            source="omnibias.dynamics.e_sigma_eps_lo.report",
        ),
    )


def enrich_ledger_with_campaign_evidence(ledger: H16Ledger) -> H16Ledger:
    """Attach replay digests to local-scope entries without changing parent flags."""
    from omnibias.dynamics.df2a import reproduce_df2a_cyclicity
    from omnibias.dynamics.drr_audit import audit_report

    digest_map = {
        "local_df2a_declared_replay": reproduce_df2a_cyclicity().source_digest,
        "local_drr_graphic_inventory": source_digest(audit_report()),
    }
    entries = []
    for entry in ledger.entries:
        if entry.name in digest_map:
            entries.append(
                H16Obligation(
                    entry.name,
                    entry.statement,
                    entry.status,
                    entry.external_premises,
                    evidence_digest=digest_map[entry.name],
                    source=entry.source,
                )
            )
        else:
            entries.append(entry)
    return H16Ledger(tuple(entries))


def default_h16_ledger() -> H16Ledger:
    """The shipped ledger: gates, DRR cases, Part A, and this plan's local wins.

    Every derived parent flag is false on this ledger, by construction --
    verified by a dedicated regression test that would fail loudly the day
    that stops being true by accident.
    """
    from omnibias.dynamics.hilbert16_uniform_ledger import (
        adversarial_entries,
        uniform_phase_entries,
        uniformity_field_entries,
    )

    return H16Ledger(
        (
            *_gate_entries(),
            *_drr_case_entries(),
            *_drr_published_corpus_entry(),
            *_part_a_entries(),
            *_local_scope_entries(),
            *uniform_phase_entries(),
            *uniformity_field_entries(),
            *adversarial_entries(),
        )
    )
