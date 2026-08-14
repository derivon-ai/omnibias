# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""omnibias-tab: differentiable, second-order-trained, certified soft decision trees.

A decision-tree split is a hard threshold ``1[w.x > t]``. omnibias makes it a **soft
oblique gate** ``g(x) = sigmoid(beta * (w.x - t))`` and anneals ``beta -> inf`` to recover
a genuine hard split. Learning an *optimal* hard tree is NP-hard, so -- exactly like the
discrete consumers -- ``tab`` delivers a well-posed **yes-if** object rather than an
impossible exactness claim:

1. a **differentiable soft-tree ensemble** -- an ensemble of oblivious soft trees whose
   ``depth == 1`` tier is a pure sum-of-sigmoids (additive, directly certifiable) and
   whose ``depth >= 2`` tier multiplies gates into ``2**depth`` leaf memberships (native
   interactions), with bit-identical :mod:`omnibias.tab.torch` / :mod:`omnibias.tab.jax`
   forwards;
2. **exact second-order training** of the whole model via :mod:`omnibias.torch.optim`
   (``CubicNewton`` / ``TrustRegionNewtonCG`` / ``KFAC`` / ``NaturalGradient``), plus a
   stagewise **Newton-boosting** driver (the GBM-mirror) using the closed-form loss
   curvature;
3. **sound certificates** (:mod:`omnibias.tab.certify`): output bounds, Lipschitz,
   per-feature monotonicity, an optional sealed scalar global-min, and a certified
   **train-soft / deploy-hard** rounding gap as ``beta -> inf``.

Terminology: the gate's ``sigmoid(beta * (w.x - t))``, ``beta -> inf`` is the
feasibility / temperature sense of "collapse" (a soft indicator hardening to a 0/1 step),
distinct from the **founding bias collapse** (the multi-bias ``delta -> 0`` limit of an
``OMBU`` to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``). The
derivative tower is still used -- exact gate curvature feeds the second-order trainer and
the ``beta -> inf`` limit earns the certified rounding gap.

The trainable module + optimizers need the ``torch`` extra; the functional twin needs
``jax``; the sealed additive certificate needs ``verify`` (each degrades gracefully).

.. important::

    **Bit-parity with the PyTorch twin requires 64-bit JAX** --
    ``jax.config.update("jax_enable_x64", True)`` before the first JAX array is
    created (or ``JAX_ENABLE_X64=1``). JAX otherwise truncates to ``float32``
    while PyTorch uses ``float64``, so the twins stay internally consistent but
    agree only to ``float32`` tolerance. Where a value feeds a threshold, a
    rounding step or an ``argmax``, that is enough to change the decision rather
    than just the last digits. See :mod:`omnibias.jax.precision`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.tab._core.config import SoftTreeConfig
from omnibias.tab._core.forward import (
    forward_np,
    hard_forward_np,
    leaf_memberships,
    predict_np,
    scores_to_prob,
)
from omnibias.tab._core.loss import loss_value, metric, score_grad_hess
from omnibias.tab._core.params import TabParams, init_params
from omnibias.tab.arrangement import (
    arrangement_params,
    arrangement_weights,
    certify_arrangement_gap,
    hard_predict_np,
    make_axis_rule,
    make_oblique_xor,
    obliqueness_diagnostic,
    predict_proba_np,
    tree_params,
)
from omnibias.tab.certify import (
    ComposedCertificate,
    RoundingGapCertificate,
    TabCertificate,
    certify_composed,
    certify_tab,
    certify_tab_gap,
)

try:
    __version__ = _pkg_version("omnibias-tab")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "both"

__all__ = [
    "ComposedCertificate",
    "RoundingGapCertificate",
    "SoftTreeConfig",
    "TabCertificate",
    "TabParams",
    "__lineage__",
    "__version__",
    "arrangement_params",
    "arrangement_weights",
    "certify_arrangement_gap",
    "certify_composed",
    "certify_tab",
    "certify_tab_gap",
    "forward_np",
    "hard_forward_np",
    "hard_predict_np",
    "init_params",
    "leaf_memberships",
    "loss_value",
    "make_axis_rule",
    "make_oblique_xor",
    "metric",
    "obliqueness_diagnostic",
    "predict_np",
    "predict_proba_np",
    "score_grad_hess",
    "scores_to_prob",
    "tree_params",
]
