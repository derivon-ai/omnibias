# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Relaxation collapse: a sound contraction enclosure decides a steady state.

A uniquely relaxing GKSL generator has a unique trace-1 fixed point
``rho_ss``. The moving parameter is ``relaxation_rate`` (``gamma t``);
its limit is ``inf``. Given a caller-declared distance budget ``eps >
0``, the adjudicated claim is finite and falsifiable: "every initial
state is certified to lie within inf-norm distance ``eps`` of
``rho_ss`` after time ``t``." On a *sound* enclosure of the contraction
factor ``C(t) = ||exp(t L) - Pi||_inf``:

* ``C.hi < eps`` -- ``PROVED``; surviving object is the **steady state**.
* ``C.lo > eps`` -- ``DISPROVED``; some deviation is still certified
  larger than ``eps``.
* otherwise, or if uniqueness cannot be certified -- ``BLOCKED``
  (``Inconclusive``, not falsity). Pure dephasing is the disagreement
  gate versus einselection: every diagonal state is fixed, so there is
  a steady-state *manifold*, not a unique fixed point, and this collapse
  must refuse.

Do not conflate this with founding bias collapse (``delta -> 0``),
temperature collapse (``beta -> inf``, the feasibility sense -- named
here only as the T=0 thermal occupancy step already minted by
:mod:`omnibias.core.occupancy`, never re-requested), or Enclosure
Collapse (``width -> 0`` of a sound enclosure, yielding a point plus a
proof). The surviving object here is a density matrix, not a
derivative, not a 0/1 step, and not a scalar point-plus-proof.

``distance_budget`` is always an explicit caller argument; it is never
defaulted.
"""

from __future__ import annotations

import math
from typing import Literal

from omnibias.core.collapse.schema import (
    CollapseOutcome,
    CollapseSpec,
    add_registry_hook,
    default_honesty,
    get_collapse,
    register_collapse,
)
from omnibias.core.collapse.verdict import ObligationVerdict
from omnibias.core.lindblad import LindbladModel
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.lindblad import contraction_enclosure, steady_state_enclosure

RELAXATION_SPEC = CollapseSpec(
    name="relaxation",
    parameter="relaxation_rate",
    limit="inf",
    surviving_object="steady_state",
    failure="contraction_not_certified",
    home="omnibias.core.collapse.relaxation",
    register="measure",
)


def _honesty() -> dict[str, bool]:
    payload = default_honesty(spec_name="relaxation")
    payload["relaxation_collapse"] = True
    payload["wave_function_collapse_claim"] = False
    payload["measurement_problem_resolved"] = False
    payload["single_outcome_claim"] = False
    payload["born_rule_derived"] = False
    payload["markovian_model_declared_not_derived"] = True
    payload["non_markovian_claim"] = False
    payload["general_closed_form_claim"] = False
    payload["thermodynamic_limit_taken"] = False
    payload["continuum_limit_taken"] = False
    payload["quantum_advantage_claim"] = False
    payload["float_residual_is_proof"] = False
    payload["continuum_parent_inferred"] = False
    return payload


_OutcomeStatus = Literal["collapsed", "excluded", "inconclusive"]


def _outcome(
    status: _OutcomeStatus, residual: Interval | None, eps: float, *, detail: str
) -> CollapseOutcome:
    surviving: str | None
    if status == "collapsed":
        surviving = "steady_state"
    elif status == "excluded":
        surviving = "DISPROVED"
    else:
        surviving = None
    return CollapseOutcome(
        status=status,
        spec_name="relaxation",
        surviving=surviving,
        residual=residual,
        detail=detail,
        honesty=_honesty(),
    )


def relaxation_collapse(
    model: LindbladModel,
    *,
    time: IntervalLike,
    distance_budget: float,
) -> ObligationVerdict:
    """Decide whether the unique ``rho_ss`` has been reached at sensitivity ``eps``.

    ``distance_budget`` is always an explicit caller argument. Comparisons
    are strict: an enclosure that merely touches ``eps`` lands in
    ``BLOCKED``. A non-unique kernel (pure dephasing) is a refusal, not a
    proof.
    """
    if isinstance(distance_budget, bool) or not isinstance(distance_budget, int | float):
        raise TypeError("distance_budget must be a real number")
    eps = float(distance_budget)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"distance_budget must be a positive finite number, got {eps!r}")

    time_iv = time if isinstance(time, Interval) else Interval.from_value(time)
    if time_iv.lo < 0.0:
        raise ValueError(f"time must be nonnegative, got lo={time_iv.lo!r}")

    if steady_state_enclosure(model) is None:
        outcome = _outcome(
            "inconclusive",
            None,
            eps,
            detail=(
                "steady state is not a unique certified fixed point "
                "(contraction_not_certified); Inconclusive, not falsity"
            ),
        )
        return ObligationVerdict(
            status="BLOCKED",
            outcome=outcome,
            existential=True,
            evaluated=1,
            complete=False,
            detail=outcome.detail,
        )

    contraction = contraction_enclosure(model, time_iv)
    if contraction is None:
        outcome = _outcome(
            "inconclusive",
            None,
            eps,
            detail="propagator enclosure refused; contraction_not_certified",
        )
        return ObligationVerdict(
            status="BLOCKED",
            outcome=outcome,
            existential=True,
            evaluated=1,
            complete=False,
            detail=outcome.detail,
        )

    if contraction.hi < eps:
        outcome = _outcome(
            "collapsed",
            contraction,
            eps,
            detail=(
                f"contraction enclosure [{contraction.lo}, {contraction.hi}] "
                f"stays below eps={eps}; unique steady state survives"
            ),
        )
        return ObligationVerdict(
            status="PROVED",
            outcome=outcome,
            existential=True,
            evaluated=1,
            complete=True,
            detail=outcome.detail,
        )
    if contraction.lo > eps:
        outcome = _outcome(
            "excluded",
            contraction,
            eps,
            detail=(
                f"contraction enclosure [{contraction.lo}, {contraction.hi}] "
                f"exceeds eps={eps}; deviation stays detectable"
            ),
        )
        return ObligationVerdict(
            status="DISPROVED",
            outcome=outcome,
            existential=True,
            evaluated=1,
            complete=True,
            detail=outcome.detail,
        )
    outcome = _outcome(
        "inconclusive",
        contraction,
        eps,
        detail=(
            f"contraction enclosure [{contraction.lo}, {contraction.hi}] "
            f"straddles eps={eps}; Inconclusive, not falsity"
        ),
    )
    return ObligationVerdict(
        status="BLOCKED",
        outcome=outcome,
        existential=True,
        evaluated=1,
        complete=False,
        detail=outcome.detail,
    )


def _reseed() -> None:
    try:
        get_collapse("relaxation")
    except KeyError:
        register_collapse(RELAXATION_SPEC)


add_registry_hook(_reseed)


__all__ = [
    "RELAXATION_SPEC",
    "relaxation_collapse",
]
