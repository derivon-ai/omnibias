# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Seal a :class:`CountCertificate` into the tamper-evident v1 certificate + Lean bridge.

A :class:`~omnibias.logic.model_count.certificate.CountCertificate` is a plain, in-memory
enclosure ``lower <= #models <= upper``.  :func:`seal_count_certificate` turns it into a
canonical, hash-sealed :data:`~omnibias.core.proof.certificate.Cert` (schema v1) that is

* **portable + tamper-evident** -- it round-trips through
  :func:`~omnibias.core.proof.certificate.verify_certificate_digest`; any mutated bound or
  flipped flag invalidates the digest; and
* **kernel-checkable** -- it carries exactly the *finite, rational* obligation the
  Mathlib-free Lean kernel (:mod:`omnibias.core.proof.lean_check`) already knows how to
  discharge, so ``check_certificate`` can earn a genuine ``theorem_prover_verified`` pass
  (and degrades gracefully to ``available=False`` with no Lean toolchain).

The **obligation classifier** picks the strongest finite claim the instance actually admits
-- it never widens what is proven:

* **exact-count identity** (payload ``type = "rational_identity"``) -- when the enclosure is
  *tight* (exact), *unweighted* (integer count), and small enough (``num_clauses <=
  identity_max_clauses``): the finite integer identity ``Z0 - S_1 + S_2 - ... = #models``
  (full inclusion-exclusion over the integer subset measures).  The kernel re-checks the
  **integer arithmetic** of the assembly; the inclusion-exclusion theorem and the subset
  measures are trusted Python inputs (recorded honestly in ``meta``).  This covers a
  kernel-checked ``#models = 0`` (certified UNSAT) too.
* **enclosed-quantity sign** (payload ``type = "interval"``) -- otherwise, when the certified
  lower bound is positive: the kernel proves ``0 < #models`` from the enclosure, i.e.
  **certified satisfiability**.
* **no finite obligation** -- a pure upper bound (zero lower bound) is still sealed and
  tamper-evident, but carries no sign/identity for the kernel; ``check_certificate`` reports
  ``"no finite Lean-checkable obligation"``.

Honesty: the sealed certificate hard-wires ``unproven_claim = False`` and makes **no** poly-time
exact-count claim (``#SAT`` is ``#P``-hard); it certifies a finite arithmetic fact about a
worst-case-sound enclosure, nothing more.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from omnibias.core.proof import (
    FORMAL_CLAIM_KEY,
    Certificate,
    Conjecture,
    FunctionProver,
    ProofAttempt,
    ProofMachine,
    Prover,
    Verdict,
    VerdictStatus,
)
from omnibias.core.proof.certificate import (
    Cert,
    encode_interval,
    make_certificate,
    schema_errors_v1,
)
from omnibias.core.verified.interval import Interval
from omnibias.logic.model_count.certificate import CountCertificate
from omnibias.logic.model_count.enclosure import _inclusion_exclusion_terms, count_enclosure
from omnibias.logic.model_count.problem import ModelCountProblem, exact_model_count

#: Default cap on ``#clauses`` for the exact-count identity (full inclusion-exclusion is
#: ``O(2^m)``; the certified regime is small / structured instances).
_IDENTITY_MAX_CLAUSES = 20


def _exact_identity_terms(
    problem: ModelCountProblem, cap: int
) -> tuple[list[list[int]], int] | None:
    r"""Integer ``(lhs_terms, count)`` for ``Z0 - S_1 + S_2 - ... = #models``, or ``None``.

    Returns the full (``order = #clauses``) inclusion-exclusion identity as integer terms
    ``[[c_k, m_k], ...]`` with ``c_0 = 1`` (the ``Z0`` term) and ``c_k = (-1)^k`` (the ``S_k``
    Bonferroni term), whose signed sum is the exact integer model count.  ``None`` when the
    instance is too large, or any term is not integral (a weighted instance) -- the caller
    then falls back to the sign obligation.
    """
    clauses = problem.cnf.clauses
    m = len(clauses)
    if m > cap:
        return None
    z0, s_terms = _inclusion_exclusion_terms(problem, order=m)
    if z0.denominator != 1 or any(sk.denominator != 1 for sk in s_terms):
        return None
    lhs_terms: list[list[int]] = [[1, int(z0)]]
    net = z0
    for k, sk in enumerate(s_terms, start=1):
        coeff = 1 if k % 2 == 0 else -1
        lhs_terms.append([coeff, int(sk)])
        net += coeff * sk
    if net.denominator != 1:
        return None
    return lhs_terms, int(net)


def seal_count_certificate(
    certificate: CountCertificate,
    *,
    problem: ModelCountProblem | None = None,
    identity_max_clauses: int = _IDENTITY_MAX_CLAUSES,
    claim: str | None = None,
    meta_extra: Mapping[str, Any] | None = None,
) -> Cert:
    r"""Seal a :class:`CountCertificate` into a tamper-evident, Lean-checkable v1 certificate.

    Parameters
    ----------
    certificate:
        The enclosure to seal.
    problem:
        The :class:`~omnibias.logic.model_count.problem.ModelCountProblem` the certificate
        counts.  Required to attach the **exact-count identity** obligation (the integer
        inclusion-exclusion terms are recomputed from it); when omitted only the
        enclosed-quantity sign obligation is available.
    identity_max_clauses:
        Skip the (``O(2^m)``) exact identity when the formula has more than this many clauses.
    claim:
        Override the auto-generated human-readable claim string.
    meta_extra:
        Extra key/value pairs merged into the certificate ``meta`` (e.g. an embedded problem
        specification the :func:`count_prover` replay twin recomputes from). Never affects the
        obligation the kernel checks.

    Returns
    -------
    :data:`~omnibias.core.proof.certificate.Cert`
        A sealed v1 certificate (``schema_version`` + ``digest``) whose ``payload`` carries
        the finite obligation the Lean kernel can discharge.  Feed it to
        :func:`omnibias.core.proof.check_certificate`.
    """
    lower, upper = float(certificate.lower), float(certificate.upper)

    identity: tuple[list[list[int]], int] | None = None
    if problem is not None and certificate.tight and not certificate.weighted:
        identity = _exact_identity_terms(problem, identity_max_clauses)
        # Defensive: only attach an identity that reproduces the sealed enclosure value.
        if identity is not None:
            _lhs, count = identity
            if abs(float(count) - lower) > 0.5 or abs(float(count) - upper) > 0.5:
                identity = None

    obligation: str
    if identity is not None:
        lhs_terms, count = identity
        payload: dict[str, object] = {
            "type": "rational_identity",
            "lhs_terms": lhs_terms,
            "rhs": count,
            "count": count,
            "interval": encode_interval(Interval(float(count), float(count))),
        }
        obligation = "rational_identity"
        default_claim = (
            f"#models = {count} (exact, unweighted) via the finite inclusion-exclusion "
            f"identity Z0 - S_1 + S_2 - ... = #models over the integer subset measures"
        )
    else:
        payload = {"type": "interval", "interval": encode_interval(Interval(lower, upper))}
        obligation = "interval_sign" if lower > 0.0 else "none"
        span = f"{lower} <= #models <= {upper}"
        default_claim = (
            f"{span} (certified enclosure; lower bound > 0 witnesses satisfiability)"
            if lower > 0.0
            else f"{span} (certified enclosure; no positive-sign obligation)"
        )

    honesty = {"unproven_claim": False, "worst_case_sound": True}
    meta: dict[str, object] = {
        "kind": "model_count",
        "method": certificate.method,
        "order": certificate.order,
        "weighted": certificate.weighted,
        "tight": certificate.tight,
        "total": certificate.total,
        "obligation": obligation,
        "poly_time_exact": False,
        "trusts": (
            "the inclusion-exclusion theorem + the integer subset measures; the kernel "
            "re-checks only the finite arithmetic. Search / relaxation heuristics are untrusted."
        ),
    }
    if problem is not None:
        meta["n"] = problem.n
        meta["num_clauses"] = len(problem.cnf.clauses)
    if meta_extra is not None:
        meta.update(meta_extra)

    return make_certificate(
        claim=claim if claim is not None else default_claim,
        payload=payload,
        honesty=honesty,
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# ProofMachine integration: a registerable prover + convenience driver.
# --------------------------------------------------------------------------- #
#: Conjecture ``kind`` the :func:`count_prover` handles.
COUNT_KIND = "model_count"
#: Default ``#clauses`` cap for the prover's exact enclosure order (kept modest so the default
#: full inclusion-exclusion stays affordable; pass ``order`` explicitly to override).
_PROVER_MAX_CLAUSES = 16
#: Default ``n`` cap for the independent ``O(2^n)`` oracle replay twin.
_REPLAY_ORACLE_MAX_N = 20


def _problem_spec(problem: ModelCountProblem) -> dict[str, Any]:
    """A compact, self-contained problem spec embedded in ``meta`` for the replay twin."""
    spec: dict[str, Any] = {
        "n": problem.n,
        "clauses": [list(clause.literals) for clause in problem.cnf.clauses],
    }
    if problem.weights is not None:
        w = problem.weights
        spec["weights"] = [[float(w[i, 0]), float(w[i, 1])] for i in range(problem.n)]
    return spec


def _adjudicate(
    claim_kind: str, certificate: CountCertificate, data: Mapping[str, Any]
) -> tuple[VerdictStatus, str, list[str]]:
    """Map a count enclosure + a claim to an intrinsic ``PROVED | DISPROVED | BLOCKED``."""
    lower, upper, tight, order = (
        certificate.lower,
        certificate.upper,
        certificate.tight,
        certificate.order,
    )
    if claim_kind == "enclosure":
        return (
            "PROVED",
            f"certified enclosure {lower:g} <= #models <= {upper:g}",
            [f"sound Bonferroni enclosure (order {order}, tight={tight})"],
        )
    if claim_kind == "sat":
        if lower > 0.0:
            return "PROVED", f"satisfiable: certified lower bound {lower:g} > 0", ["#models >= lower > 0"]
        if tight and upper == 0.0:
            return "DISPROVED", "unsatisfiable: exact count is 0", ["#models == 0 refutes satisfiability"]
        return (
            "BLOCKED",
            f"satisfiability undetermined at order {order} (lower={lower:g})",
            ["raise the enclosure order or supply model witnesses"],
        )
    if claim_kind == "unsat":
        if tight and upper == 0.0:
            return "PROVED", "unsatisfiable: exact count is 0", ["#models == 0"]
        if lower > 0.0:
            return "DISPROVED", f"a model exists: certified lower bound {lower:g} > 0", ["#models > 0 refutes unsat"]
        return "BLOCKED", f"unsat undetermined at order {order}", ["raise the order for a tight [0, 0]"]
    if claim_kind == "count":
        raw = data.get("value", data.get("target"))
        if raw is None:
            return "BLOCKED", "claim 'count' requires data['value'] (the asserted integer count)", ["supply data['value']"]
        target = float(raw)
        if not (lower - 1e-9 <= target <= upper + 1e-9):
            return (
                "DISPROVED",
                f"asserted count {target:g} lies outside the certified [{lower:g}, {upper:g}]",
                ["target outside the enclosure"],
            )
        if not tight:
            return (
                "BLOCKED",
                f"count not pinned exactly at order {order} (enclosure [{lower:g}, {upper:g}])",
                ["raise the order to collapse the enclosure to a point"],
            )
        if abs(lower - target) <= 1e-9:  # tight -> lower == upper is the exact count
            return "PROVED", f"exact count == {target:g}", [f"tight enclosure pins #models = {lower:g}"]
        return "DISPROVED", f"exact count is {lower:g}, not {target:g}", ["tight enclosure refutes the asserted count"]
    return "BLOCKED", f"unknown claim kind {claim_kind!r}", ["use one of: enclosure, sat, unsat, count"]


def count_prover(
    *,
    name: str = "omnibias-logic:model_count",
    identity_max_clauses: int = _PROVER_MAX_CLAUSES,
    replay_oracle_max_n: int = _REPLAY_ORACLE_MAX_N,
) -> FunctionProver:
    r"""A :class:`~omnibias.core.proof.FunctionProver` that certifies a (weighted) model count.

    Register it on a :class:`~omnibias.core.proof.ProofMachine` to adjudicate a
    :class:`~omnibias.core.proof.Conjecture` of ``kind = "model_count"`` (build one with
    :func:`model_count_conjecture`). The prover:

    * builds a :func:`~omnibias.logic.count_enclosure` sandwich (default ``order`` counts
      exactly when ``#clauses <= identity_max_clauses``), seals it with
      :func:`seal_count_certificate`, and returns the sealed certificate;
    * decides ``PROVED | DISPROVED | BLOCKED`` for the conjecture's ``claim``
      (``"enclosure"`` | ``"sat"`` | ``"unsat"`` | ``"count"``) -- never overclaiming (an
      exact ``"count"`` / ``"unsat"`` needs a *tight* enclosure, else ``BLOCKED``);
    * validates its own certificate (``schema_fn``) and offers an **independent** replay twin
      (``replay_fn``): the sealed count is re-derived by the ``O(2^n)`` enumeration oracle
      (:func:`~omnibias.logic.exact_model_count`, a different algorithm than inclusion-
      exclusion) up to ``replay_oracle_max_n``, plus a standalone check of the integer
      identity ``sum_i c_i m_i = rhs``.

    The machine then layers on the tamper-evident digest gate, the honesty gate, and -- with
    ``lean_check=True`` -- the Lean-kernel gate that sets ``theorem_prover_verified`` only on a
    genuine ``lake build`` pass.
    """

    def prove(conjecture: Conjecture) -> ProofAttempt:
        data = conjecture.data or {}
        problem = data.get("problem")
        if not isinstance(problem, ModelCountProblem):
            return ProofAttempt(
                status="BLOCKED",
                certificate=None,
                obligations=("conjecture.data['problem'] must be a ModelCountProblem",),
                detail="no ModelCountProblem supplied",
            )
        claim_kind = str(data.get("claim", "enclosure"))
        num_clauses = len(problem.cnf.clauses)
        order = data.get("order")
        order = int(order) if order is not None else (num_clauses if num_clauses <= identity_max_clauses else 2)
        certificate = count_enclosure(problem, order=order, witnesses=data.get("witnesses"))
        sealed = seal_count_certificate(
            certificate,
            problem=problem,
            identity_max_clauses=identity_max_clauses,
            meta_extra={"problem": _problem_spec(problem), "claim_kind": claim_kind},
        )
        status, detail, obligations = _adjudicate(claim_kind, certificate, data)
        return ProofAttempt(status=status, certificate=sealed, obligations=tuple(obligations), detail=detail)

    def schema(certificate: Certificate) -> list[str]:
        errors = list(schema_errors_v1(certificate))
        payload = certificate.get("payload")
        if not isinstance(payload, Mapping):
            return [*errors, "payload must be a mapping"]
        ptype = payload.get("type")
        if ptype not in ("rational_identity", "interval"):
            errors.append(f"unexpected model-count payload type {ptype!r}")
        elif ptype == "rational_identity":
            if not isinstance(payload.get("lhs_terms"), list) or not isinstance(payload.get("rhs"), int):
                errors.append("rational_identity payload needs integer lhs_terms + rhs")
        else:
            iv = payload.get("interval")
            if not (isinstance(iv, Mapping) and "lo" in iv and "hi" in iv):
                errors.append("interval payload needs lo/hi endpoints")
        return errors

    def replay(certificate: Certificate) -> bool | None:
        payload = certificate.get("payload", {})
        ptype = payload.get("type") if isinstance(payload, Mapping) else None
        # 1) Standalone arithmetic check of the finite identity (oracle-independent).
        if ptype == "rational_identity":
            terms, rhs = payload.get("lhs_terms"), payload.get("rhs")
            if not (isinstance(terms, list) and isinstance(rhs, int)):
                return False
            if sum(c * m for c, m in terms) != rhs:
                return False
        # 2) Independent O(2^n) enumeration-oracle recount from the embedded problem spec.
        meta = certificate.get("meta")
        spec = meta.get("problem") if isinstance(meta, Mapping) else None
        if not isinstance(spec, Mapping):
            return None  # no embedded twin -> no independent replay available
        from omnibias.logic.model_count.frontends import model_count

        try:
            problem = model_count(list(spec["clauses"]), spec.get("weights"), n_vars=int(spec["n"]))
        except (ValueError, KeyError, TypeError):
            return None
        if problem.n > replay_oracle_max_n:
            return None
        exact = exact_model_count(problem)
        if ptype == "rational_identity":
            return abs(exact - float(payload["rhs"])) <= 1e-9
        if ptype == "interval" and isinstance(payload, Mapping):
            iv = payload.get("interval", {})
            lo, hi = float.fromhex(iv["lo"]), float.fromhex(iv["hi"])
            return lo - 1e-9 <= exact <= hi + 1e-9
        return None

    return FunctionProver(
        name=name,
        kinds=frozenset({COUNT_KIND}),
        prove_fn=prove,
        schema_fn=schema,
        replay_fn=replay,
    )


def model_count_conjecture(
    name: str,
    problem: ModelCountProblem,
    *,
    claim: str = "enclosure",
    order: int | None = None,
    value: float | None = None,
    witnesses: object | None = None,
    claims: Mapping[str, bool] | None = None,
) -> Conjecture:
    r"""Build a ``kind = "model_count"`` :class:`~omnibias.core.proof.Conjecture` for :func:`count_prover`.

    ``claim`` selects the statement to adjudicate (``"enclosure"`` | ``"sat"`` | ``"unsat"`` |
    ``"count"``); ``value`` is the asserted integer count for ``claim="count"``. ``claims``
    are honesty claims the machine's honesty gate enforces (e.g.
    ``{"theorem_prover_verified": True}`` to *require* a Lean-kernel pass).
    """
    data: dict[str, Any] = {"problem": problem, "claim": claim}
    if order is not None:
        data["order"] = order
    if value is not None:
        data["value"] = value
    if witnesses is not None:
        data["witnesses"] = witnesses
    return Conjecture(name=name, kind=COUNT_KIND, data=data, claims=dict(claims or {}))


def prove_model_count(
    problem: ModelCountProblem,
    *,
    claim: str = "enclosure",
    order: int | None = None,
    value: float | None = None,
    witnesses: object | None = None,
    assert_theorem_prover: bool = False,
    lean_check: bool = False,
    strict: bool = False,
    name: str | None = None,
) -> Verdict:
    r"""One-call driver: adjudicate a model-count :class:`~omnibias.core.proof.Conjecture` end to end.

    Registers :func:`count_prover` on a fresh :class:`~omnibias.core.proof.ProofMachine` and
    evaluates the conjecture, returning the full :class:`~omnibias.core.proof.Verdict` (status,
    sealed certificate, schema / replay / honesty gates, and ``theorem_prover_verified``).

    Set ``assert_theorem_prover=True`` to *require* a genuine Lean-kernel pass: the formal
    honesty gate then downgrades a ``PROVED`` verdict to ``BLOCKED`` unless the kernel actually
    verified the certificate (so with no toolchain the claim honestly blocks). ``lean_check``
    runs the kernel opportunistically without requiring it; ``strict`` additionally demands a
    present, valid digest and an agreeing independent replay twin.
    """
    formal = {FORMAL_CLAIM_KEY: True} if assert_theorem_prover else None
    conjecture = model_count_conjecture(
        name if name is not None else f"model_count[{claim}]",
        problem,
        claim=claim,
        order=order,
        value=value,
        witnesses=witnesses,
        claims=formal,
    )
    # cast: FunctionProver satisfies the Prover protocol at runtime; the frozen dataclass's
    # read-only ``name`` is the only (variance) reason mypy will not infer it structurally.
    machine = ProofMachine().register(cast(Prover, count_prover()))
    return machine.evaluate(conjecture, lean_check=lean_check or assert_theorem_prover, strict=strict)


__all__ = [
    "COUNT_KIND",
    "count_prover",
    "model_count_conjecture",
    "prove_model_count",
    "seal_count_certificate",
]
