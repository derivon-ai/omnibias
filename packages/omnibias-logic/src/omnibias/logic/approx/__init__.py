# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Statistical model-count estimators -- **NOT worst-case sound** (read this first).

This subpackage is deliberately quarantined from the sound surface of ``omnibias.logic``. Its
estimators carry only a **probabilistic / coverage** guarantee, never a rigorous enclosure:

* :func:`approx_model_count` -- ApproxMC-style ``(epsilon, delta)`` XOR-hashing: the interval
  brackets the true count within ``(1 + epsilon)`` with probability ``>= 1 - delta`` (it can
  fail with probability ``delta``);
* :class:`ConformalCounter` -- split-conformal coverage: marginal coverage ``>= 1 - alpha``
  over the calibration distribution (distribution-dependent, not adversarial).

Both return an :class:`ApproxCount`, a type that is **structurally distinct** from
:class:`~omnibias.logic.model_count.certificate.CountCertificate` and hard-wires
``worst_case_sound = False`` -- so a statistical estimate can never be passed off as a rigorous
bound. It is **not** hoisted into the top-level ``omnibias.logic`` namespace; import it
explicitly (``from omnibias.logic.approx import approx_model_count``) as a reminder that you
are leaving the sound surface. For guaranteed counts use the exact router
(:func:`omnibias.logic.count`) or the certified enclosure
(:func:`omnibias.logic.count_enclosure`).
"""

from __future__ import annotations

from omnibias.logic.approx.conformal import ConformalCounter, monte_carlo_estimate
from omnibias.logic.approx.hashing import approx_model_count
from omnibias.logic.approx.result import NOT_SOUND_DISCLAIMER, ApproxCount

__all__ = [
    "ApproxCount",
    "ConformalCounter",
    "NOT_SOUND_DISCLAIMER",
    "approx_model_count",
    "monte_carlo_estimate",
]
