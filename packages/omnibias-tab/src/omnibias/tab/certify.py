# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Sound certificates for a trained soft-tree ensemble.

:func:`certify_tab` bundles, for a :class:`~omnibias.tab._core.params.TabParams` and an
input hyper-box, the guarantees a tabular model can *earn* (not just assert):

* **output bounds** -- a rigorous enclosure of every output over the box;
* **Lipschitz** -- an upper bound on the model's sensitivity (any-``n`` continuity);
* **per-feature monotonicity** -- the certified sign of ``dF / dx_f`` over the box, usable
  as a *sound* monotone constraint (GBMs offer this as a soft prior; here it is proved);
* a certified **train-soft / deploy-hard rounding gap** as ``beta -> inf``
  (:func:`certify_tab_gap`) -- how much hardening the soft splits can move the score.

Two engines, both sound (a looser bound only widens the certified gap):

* the **interval** engine (:mod:`omnibias.tab._core.verified`) works for any depth using
  only ``omnibias-core``'s outward-rounded substrate;
* for the **additive** (``depth == 1``) tier -- a genuine ``Linear -> Sigmoid -> Linear``
  network -- the tighter / sealed **verify** engine (``omnibias-verify`` Taylor +
  branch-and-bound) is used when available, and falls back to the interval engine
  otherwise.

Honesty (per the discrete consumers' yes-if framing): the certificates are genuine sound
enclosures, never exact-optimality claims, and the rounding gap is never asserted zero.

Terminology: the ``beta -> inf`` gate hardening is the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias ``delta -> 0``
limit to ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from omnibias.tab._core.params import TabParams
from omnibias.tab._core.verified import (
    interval_jacobian,
    interval_output_bounds,
    lipschitz_from_jacobian,
    normalize_box,
    rounding_gap,
)


@dataclass(frozen=True)
class RoundingGapCertificate:
    r"""A certified bound on ``|F_soft - F_hard|`` as ``beta -> inf`` over a sample set."""

    beta: float
    max_gap: float
    mean_gap: float
    measured_max: float
    per_output_max: tuple[float, ...]

    @property
    def is_sound(self) -> bool:
        """The certified bound must dominate the actually-measured gap (self-check)."""
        return self.max_gap >= self.measured_max - 1e-9


@dataclass(frozen=True)
class TabCertificate:
    r"""The certificate bundle returned by :func:`certify_tab`."""

    beta: float
    method: str
    output_bounds: tuple[tuple[float, float], ...]
    lipschitz: float
    output_index: int
    monotonicity: dict[int, str] = field(default_factory=dict)
    monotone_ok: bool | None = None
    rounding: RoundingGapCertificate | None = None

    @property
    def certified(self) -> bool:
        """All *requested* monotonicity constraints held (``True`` if none were requested)."""
        return True if self.monotone_ok is None else self.monotone_ok

    @property
    def rounding_gap(self) -> float | None:
        return None if self.rounding is None else self.rounding.max_gap


def _try_build_verify_network(params: TabParams, beta: float) -> object | None:
    if not params.config.is_additive:
        return None
    try:
        import torch  # noqa: F401
        from omnibias.tab.torch.model import SoftTreeEnsemble
        from omnibias.verify.torch import network_from_sequential
    except ImportError:
        return None
    model = SoftTreeEnsemble(params.config, params)
    seq = model.to_additive_sequential(beta)
    net: object = network_from_sequential(seq)
    return net


def _want_verdict(sign: int) -> str:
    return "increasing" if sign > 0 else "decreasing"


def certify_tab(
    params: TabParams,
    feature_box: object,
    *,
    monotone_features: dict[int, int] | None = None,
    X: np.ndarray | None = None,
    beta: float | None = None,
    output_index: int = 0,
    use_verify: bool = True,
    norm: str = "l2",
) -> TabCertificate:
    r"""Certify a soft-tree model over ``feature_box`` (a ``(2, d)`` lo/hi array or box).

    Parameters
    ----------
    params:
        The trained ensemble (``model.to_params()``).
    feature_box:
        The input hyper-box to certify over (e.g. ``np.stack([X.min(0), X.max(0)])``).
    monotone_features:
        Optional ``{feature_index: +1 | -1}`` constraints (``+1`` = require increasing);
        ``monotone_ok`` reports whether all held. If ``None``, every feature's verdict is
        still recorded but no pass/fail is asserted.
    X:
        Optional samples for the certified soft->hard rounding gap (:func:`certify_tab_gap`).
    beta:
        Gate sharpness to certify at (defaults to the config's ``beta_final`` -- the
        deploy-time value).
    output_index:
        Which output the monotonicity verdicts refer to (default ``0``).
    use_verify:
        Use the tighter / sealed ``omnibias-verify`` engine for the additive tier when
        available (falls back to the interval engine otherwise).
    norm:
        Lipschitz norm (``"l2"`` default or ``"inf"``).
    """
    b = float(params.config.beta_final if beta is None else beta)
    d = params.n_features
    box_iv = normalize_box(feature_box, d)
    feats = list(monotone_features.keys()) if monotone_features is not None else list(range(d))

    net = _try_build_verify_network(params, b) if use_verify else None
    monotonicity: dict[int, str] = {}

    if net is not None:
        from omnibias.verify import lipschitz_bound, reachable_box
        from omnibias.verify import monotonicity as vmono

        method = "verify"
        obounds = reachable_box(net, box_iv)
        output_bounds = tuple((float(iv.lo), float(iv.hi)) for iv in obounds)
        lipschitz = float(lipschitz_bound(net, box_iv, norm=norm))
        for f in feats:
            monotonicity[f] = vmono(net, box_iv, output_index, f).verdict
    else:
        method = "interval"
        obounds = interval_output_bounds(params, feature_box, b)
        output_bounds = tuple((float(iv.lo), float(iv.hi)) for iv in obounds)
        jac = interval_jacobian(params, feature_box, b)
        lipschitz = float(max(lipschitz_from_jacobian(jac, norm=norm)))
        for f in feats:
            dv = jac[output_index][f]
            monotonicity[f] = (
                "increasing" if dv.lo >= 0.0 else "decreasing" if dv.hi <= 0.0 else "unknown"
            )

    monotone_ok: bool | None = None
    if monotone_features is not None:
        monotone_ok = all(
            monotonicity.get(f) == _want_verdict(sign) for f, sign in monotone_features.items()
        )

    rounding = certify_tab_gap(params, X, beta=b) if X is not None else None

    return TabCertificate(
        beta=b,
        method=method,
        output_bounds=output_bounds,
        lipschitz=lipschitz,
        output_index=output_index,
        monotonicity=monotonicity,
        monotone_ok=monotone_ok,
        rounding=rounding,
    )


def certify_tab_gap(
    params: TabParams,
    X: np.ndarray,
    *,
    beta: float | None = None,
) -> RoundingGapCertificate:
    r"""Certified train-soft / deploy-hard rounding gap on the samples ``X``.

    Returns a sound per-sample, per-output bound on ``|F_soft - F_hard|`` (the score change
    from hardening the soft splits at ``beta -> inf``), aggregated to its max / mean over
    ``X``, alongside the actually-measured gap for the ``is_sound`` self-check.
    """
    b = float(params.config.beta_final if beta is None else beta)
    bound, measured = rounding_gap(params, X, b)
    return RoundingGapCertificate(
        beta=b,
        max_gap=float(bound.max()),
        mean_gap=float(bound.mean()),
        measured_max=float(measured.max()),
        per_output_max=tuple(float(v) for v in bound.max(axis=0)),
    )


__all__ = [
    "RoundingGapCertificate",
    "TabCertificate",
    "certify_tab",
    "certify_tab_gap",
]
