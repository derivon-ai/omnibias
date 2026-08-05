# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified argmax: the measure-mode ``beta -> inf`` collapse of a Gibbs law to its mode.

Over ``N`` logits ``s`` the Gibbs law ``p_beta(i) ∝ exp(beta s_i)`` collapses onto a Dirac at
the mode ``argmax(s)`` as ``beta -> inf``. Softmax is not the contribution here (everyone has
it); the contribution is a *sound, closed-form certificate* of how far that collapse has
gone, with three honest sub-claims (all over sorted logits ``s_(1) >= s_(2) >= ...``, margin
``m = s_(1) - s_(2)``):

* **Value gap** (measure -> point value): ``max(s) <= lse_beta(s) <= max(s) + log(N)/beta`` --
  reuses the DP :func:`certify_soft_dp` / :func:`logsumexp_gap_bound` with ``num_paths = N``.
* **Mass concentration** (measure -> Dirac): ``p_max >= 1 / (1 + (N-1) e^{-beta m})``, a
  closed-form lower bound on the Gibbs mass at the mode; hence the mass is ``>= 1 - eps`` once
  ``beta >= log((N-1)/eps) / m`` (:func:`beta_for_confidence`).
* **Argmax stability** (optional, decode robustness): the decoded ``argmax`` is the unique
  maximiser for *every* perturbation ``||delta||_inf <= eps`` iff ``m > 2 eps`` -- the flat
  analogue of :class:`omnibias.struct.DecodingCertificate` (a margin-over-``L^inf``-ball).

**Two axes, never conflated.** This is the ``beta -> inf`` *feasibility / temperature* sense
of "collapse" (the same axis as ``omnibias-discrete`` / ``omnibias-qubo``) -- it is
**not** the founding ``delta -> 0`` bias collapse. The founding derivative tower is only the
*engine* that differentiates ``lse_beta`` exactly (the closed-form ``softplus^(n) = sigma^(n-1)``
jets in :mod:`omnibias.struct.torch._logsumexp`); do not conflate the two. The certificate is
honest: it never claims ``lse_beta == max`` or ``p_max == 1``, only the closed-form bounds that
tighten as ``beta -> inf``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from omnibias.struct._core.certificate import DPGapCertificate, certify_soft_dp

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

FloatArray = NDArray[np.float64]


def mass_concentration_bound(margin: float, num_choices: int, beta: float) -> float:
    r"""Closed-form lower bound on the Gibbs mass at the mode: ``1 / (1 + (N-1) e^{-beta m})``.

    For a Gibbs law ``p_i ∝ e^{beta s_i}`` with top-two margin ``m = s_(1) - s_(2) >= 0``, every
    non-mode term obeys ``e^{-beta (s_(1) - s_i)} <= e^{-beta m}`` (since ``s_i <= s_(2)``), so
    ``p_max = 1 / (1 + sum_{i != mode} e^{-beta (s_(1) - s_i)}) >= 1 / (1 + (N-1) e^{-beta m})``.
    A larger ``num_choices`` or smaller ``beta`` / ``margin`` only *lowers* the bound -- it is
    never optimistic. ``num_choices == 1`` returns ``1.0`` (a singleton is already a Dirac).
    """
    if num_choices < 1:
        raise ValueError(f"num_choices must be >= 1, got {num_choices}")
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if num_choices == 1:
        return 1.0
    if margin < 0.0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    if math.isinf(margin):
        return 1.0
    return 1.0 / (1.0 + (num_choices - 1) * math.exp(-beta * margin))


def beta_for_confidence(margin: float, num_choices: int, target_mass: float) -> float:
    r"""Smallest ``beta`` for which :func:`mass_concentration_bound` certifies ``p_max >= target``.

    Inverting the bound: ``1/(1+(N-1)e^{-beta m}) >= t`` iff
    ``beta >= log((N-1) t / (1 - t)) / m``. Requires ``0 < target_mass < 1`` and ``margin > 0``
    (with a unique mode, any confidence ``< 1`` is reachable at finite ``beta``).
    """
    if not 0.0 < target_mass < 1.0:
        raise ValueError(f"target_mass must be in (0, 1), got {target_mass}")
    if margin <= 0.0:
        raise ValueError(f"margin must be > 0 to reach a target confidence, got {margin}")
    if num_choices <= 1:
        return 0.0
    ratio = (num_choices - 1) * target_mass / (1.0 - target_mass)
    return max(math.log(ratio) / margin, 0.0)


def argmax_stability_margin(logits: NDArray[Any] | list[float]) -> float:
    r"""The top-two logit margin ``m = s_(1) - s_(2)`` (``+inf`` for a single choice).

    The decoded ``argmax`` is the unique maximiser over every input perturbation
    ``||delta||_inf <= eps`` exactly when ``m > 2 eps``; equivalently the certified robustness
    radius of the decode is ``m / 2``.
    """
    s = np.asarray(logits, dtype=float).reshape(-1)
    if s.size == 0:
        raise ValueError("logits must be non-empty")
    if s.size == 1:
        return math.inf
    top2 = np.partition(s, -2)[-2:]
    return float(top2[1] - top2[0])


@dataclass(frozen=True)
class SelectionCertificate:
    r"""A closed-form certificate of the ``beta -> inf`` Gibbs-to-Dirac (mode) collapse.

    Composes the value-gap sandwich (:class:`DPGapCertificate`, ``num_paths = N``) with the
    mass-concentration lower bound and an optional ``L^inf`` argmax-stability radius. Honest by
    construction: it states only bounds that hold, never ``lse_beta == max`` or ``p_max == 1``.

    Attributes
    ----------
    value_gap:
        The ``max(s) <= lse_beta(s) <= max(s) + log(N)/beta`` sandwich (``sense == "max"``).
    argmax:
        The decoded mode index ``argmax(s)``.
    margin:
        The top-two logit margin ``m = s_(1) - s_(2)`` (``+inf`` for ``N == 1``).
    p_max:
        The exact Gibbs mass at the mode, ``softmax(beta s)_argmax``.
    p_max_lower:
        The closed-form lower bound ``1 / (1 + (N-1) e^{-beta m})`` on ``p_max``.
    eps:
        The queried ``L^inf`` perturbation radius (``None`` if stability was not queried).
    tol:
        Numerical tolerance for the soundness self-checks.
    """

    value_gap: DPGapCertificate
    argmax: int
    margin: float
    p_max: float
    p_max_lower: float
    eps: float | None = None
    tol: float = 1e-9

    @property
    def beta(self) -> float:
        """The inverse temperature ``beta`` of the relaxation."""
        return self.value_gap.beta

    @property
    def num_choices(self) -> int:
        """The number ``N`` of choices (logits)."""
        return self.value_gap.num_paths

    @property
    def gap_bound(self) -> float:
        """The closed-form value gap ``log(N)/beta``."""
        return self.value_gap.gap_bound

    @property
    def hard_value(self) -> float:
        """The hard optimum ``max(s)``."""
        return self.value_gap.hard_value

    @property
    def soft_value(self) -> float:
        """The soft value ``lse_beta(s)``."""
        return self.value_gap.soft_value

    @property
    def mass_concentration_holds(self) -> bool:
        """Whether the measured ``p_max`` respects its closed-form lower bound."""
        return self.p_max >= self.p_max_lower - self.tol

    @property
    def argmax_stable(self) -> bool | None:
        """Whether the decode is the unique argmax over the ``eps``-ball (``None`` if no ``eps``)."""
        if self.eps is None:
            return None
        return self.margin > 2.0 * self.eps + self.tol

    @property
    def robust_radius(self) -> float:
        """The certified ``L^inf`` radius ``m / 2`` within which the argmax is unique."""
        return self.margin / 2.0

    @property
    def is_sound(self) -> bool:
        """Whether every asserted bound holds for the measured values (the soundness check)."""
        return self.value_gap.is_sound and self.mass_concentration_holds

    @property
    def certified(self) -> bool:
        """Whether a rigorous closed-form collapse certificate was produced *and* is sound."""
        return self.value_gap.certified and self.mass_concentration_holds


def certify_argmax(
    logits: NDArray[Any] | list[float],
    beta: float,
    *,
    eps: float | None = None,
    tol: float = 1e-9,
) -> SelectionCertificate:
    r"""Certify the ``beta -> inf`` Gibbs-to-Dirac collapse of ``softmax(beta * logits)``.

    Computes, in closed form (no optimization): the value-gap sandwich via
    :func:`certify_soft_dp` (``num_paths = len(logits)``), the exact mode mass ``p_max`` and its
    lower bound :func:`mass_concentration_bound`, and -- when ``eps`` is given -- the
    ``L^inf`` argmax-stability verdict ``m > 2 eps``. Returns a :class:`SelectionCertificate`.

    Parameters
    ----------
    logits:
        A 1-D array of ``N >= 1`` real scores ``s``.
    beta:
        The inverse temperature ``beta > 0``.
    eps:
        Optional ``L^inf`` perturbation radius for the decode-stability sub-claim.
    tol:
        Numerical tolerance for the soundness self-checks.
    """
    s = np.asarray(logits, dtype=float).reshape(-1)
    n = int(s.size)
    if n < 1:
        raise ValueError("logits must be non-empty")
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if eps is not None and eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}")

    scaled = beta * s
    shift = float(np.max(scaled))
    exp_shifted = np.exp(scaled - shift)
    soft_value = float(shift + np.log(np.sum(exp_shifted))) / beta
    hard_value = float(np.max(s))
    p_max = float(np.max(exp_shifted) / np.sum(exp_shifted))
    argmax = int(np.argmax(s))
    margin = argmax_stability_margin(s)
    p_max_lower = mass_concentration_bound(margin, n, beta)

    value_gap = certify_soft_dp(
        hard_value=hard_value,
        soft_value=soft_value,
        num_paths=n,
        beta=beta,
        sense="max",
        tol=tol,
    )
    return SelectionCertificate(
        value_gap=value_gap,
        argmax=argmax,
        margin=margin,
        p_max=p_max,
        p_max_lower=p_max_lower,
        eps=None if eps is None else float(eps),
        tol=float(tol),
    )


def seal_selection_certificate(
    cert: SelectionCertificate, *, meta: dict[str, Any] | None = None
) -> Cert:
    r"""Seal a :class:`SelectionCertificate` as a tamper-evident v1 certificate.

    The payload records the closed-form collapse claims (value gap, mass concentration, and the
    optional ``L^inf`` argmax-stability radius). Honesty fixes ``unproven_claim=False``; this is a
    textbook analytic inequality, so ``theorem_prover_verified`` is not sought (the digest is
    the guarantee). Verify with :func:`omnibias.core.proof.certificate.verify_certificate_digest`.
    """
    # Imported here (not at module load) so that `import omnibias.struct` stays free of the
    # order-sensitive omnibias.core.proof <-> omnibias.core.verified import cycle; warming
    # ``verified`` first fully resolves that cycle before ``proof.certificate`` is read.
    import omnibias.core.verified  # noqa: F401  (side-effect: breaks the core import cycle)
    from omnibias.core.proof.certificate import make_certificate

    payload: dict[str, Any] = {
        "type": "certified_argmax",
        "beta": cert.beta,
        "num_choices": cert.num_choices,
        "argmax": cert.argmax,
        "hard_value": cert.hard_value,
        "soft_value": cert.soft_value,
        "value_gap_bound": cert.gap_bound,
        "margin": cert.margin,
        "p_max": cert.p_max,
        "p_max_lower": cert.p_max_lower,
    }
    if cert.eps is not None:
        payload["eps"] = cert.eps
        payload["argmax_stable"] = bool(cert.argmax_stable)
    claim = (
        f"softmax(beta={cert.beta!r}) over {cert.num_choices} choices collapses onto mode "
        f"{cert.argmax}: max <= lse_beta <= max + {cert.gap_bound!r}, and Gibbs mass "
        f">= {cert.p_max_lower!r}"
    )
    payload_meta: dict[str, Any] = {
        "scope": "closed_form_inequality",
        "label": "closed-form (log(N)/beta value gap + mode-mass concentration bound)",
    }
    if meta:
        payload_meta.update(meta)
    return make_certificate(
        claim=claim,
        payload=payload,
        honesty={"unproven_claim": False},
        meta=payload_meta,
    )


__all__ = [
    "SelectionCertificate",
    "argmax_stability_margin",
    "beta_for_confidence",
    "certify_argmax",
    "mass_concentration_bound",
    "seal_selection_certificate",
]
