# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Sound enclosure of the truncation-plus-bootstrap policy-gradient bias
(theory 10-03).

A short-horizon adjoint trainer (SHAC/PEARL-style, see
:mod:`omnibias.control.jax.policy` / :mod:`omnibias.control.torch.policy`)
replaces the true continuation adjoint ``lambda_h`` at the truncation point
``h`` with a learned estimate ``lambda_pred_h``. Every per-step Jacobian
``M_0, ..., M_{h-1}`` used for the *first* ``h`` steps is exact (assembled
by autodiff / the jet tower, the same as the untruncated adjoint); the only
error source is the terminal substitution. Linearising the costate
recursion in that one substitution gives an exact first-order bias
expression, and this module bounds its norm:

.. math::
    \big\| \nabla_\theta J - \nabla_\theta J_h \big\| \;\le\;
    \Big(\sum_{k=0}^{h-1} \big\| \partial\pi/\partial\theta_k \big\|
    \, \big\| B_k \big\| \, \big\| \Phi_{k+1} \big\| \Big)
    \, \big\| \lambda_{\mathrm{pred},h} - \lambda_{\mathrm{true},h} \big\|

where :math:`\Phi_{k+1} = M_{h-1} M_{h-2} \cdots M_{k+1}` (empty product at
``k=h-1`` is the identity). This is the discrete adjoint-sensitivity chain
rule applied once to the one substitution actually made, not a bound
derived from first principles about neural-network approximation.

The **operator norms of the per-step matrices are ordinary floating-point
computations** (``numpy`` induced-inf-norm), not outward-rounded interval
arithmetic -- the ``M_k / B_k / dpi/dtheta_k`` are point values computed
once by autodiff and are not themselves being enclosed. The certified part
of the chain is :func:`terminal_adjoint_error_bound`: a genuinely sound,
outward-rounded enclosure of ``||lambda_pred_h - lambda_true_h||`` over a
declared input box, built by composing :func:`omnibias.verify.lipschitz_bound`
(reused verbatim, not reimplemented) with a measured point residual at the
box centre -- the standard Lipschitz-extension argument
``sup_box ||e(y)|| <= |e(y0)| + L * radius(box)``.

This is **founding bias collapse** (``delta -> 0``), not temperature
collapse (``beta -> inf``, the feasibility sense): nothing here sharpens a
``beta -> inf`` gate. do not conflate the two. The overall
bound is therefore *conditional*: sound given (1) the supplied
``error_net`` genuinely represents ``lambda_pred - lambda_true`` over the
box (a modelling choice made by the caller, not verified here) and (2) the
point residual at the box centre is itself accurate. Neither condition is
checked by this module; callers must record which case applies (see
``theory/10-control/03-certified-horizon-gradient-bias.md``,
"Honesty boundaries"). ``theorem_prover_verified`` is never asserted here.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.verify import Network, lipschitz_bound

FloatArray = NDArray[np.float64]

DISCLAIMER = (
    "sound enclosure of the terminal-substitution policy-gradient bias, "
    "conditional on a caller-supplied adjoint-net error model; founding bias "
    "collapse, not temperature collapse; not a from-first-principles "
    "generalization bound"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "unconditional_bound_claimed": False,
        "generalization_bound_claimed": False,
        "stretch_claim": False,
        "continuum_claimed": False,
        "theorem_prover_verified": False,
        "skips_chain_rule": False,
    }


def _op_norm_sym(a: FloatArray) -> float:
    r"""``max(||A||_inf, ||A||_1)``: a sound bound on *both* ``||A||_inf`` and
    ``||A^T||_inf`` (since ``||A^T||_inf = ||A||_1``), so the chain below never
    has to track a transpose direction to stay sound. Ordinary floating-point
    arithmetic, not outward-rounded interval arithmetic (see module docstring).
    """
    m = np.asarray(a, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {m.shape}")
    if m.size == 0:
        return 0.0
    row_sum = float(np.max(np.sum(np.abs(m), axis=1)))
    col_sum = float(np.max(np.sum(np.abs(m), axis=0)))
    return max(row_sum, col_sum)


@dataclass(frozen=True)
class GradientBiasReport:
    """Sound (conditional) bound on ``||grad_theta J - grad_theta J_h||``."""

    bound: float
    terminal_error_bound: float
    horizon: int
    per_step_terms: list[float]
    certified: bool


def truncation_bias_bound(
    dpi_dtheta_seq: Sequence[FloatArray],
    b_seq: Sequence[FloatArray],
    m_seq: Sequence[FloatArray],
    terminal_error_bound: float,
) -> GradientBiasReport:
    r"""Sum the per-step sensitivity terms into one bound (gate G6).

    ``m_seq[k]`` is the exact closed-loop Jacobian
    :func:`omnibias.core.adjoint.closed_loop_jacobian` at step ``k``
    (``k = 0 .. h-1``);     ``dpi_dtheta_seq`` / ``b_seq`` match in length.
    ``terminal_error_bound`` is a non-negative sound bound on
    ``||lambda_pred_h - lambda_true_h||_inf`` (see
    :func:`terminal_adjoint_error_bound`); every norm in this function's
    chain is the (symmetrised) infinity-induced operator norm, so the
    result bounds the resulting gradient-term error in the vector
    infinity norm.
    """
    horizon = len(m_seq)
    if not (len(dpi_dtheta_seq) == len(b_seq) == horizon):
        raise ValueError("dpi_dtheta_seq, b_seq, m_seq must have equal length")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    err = float(terminal_error_bound)
    if err < 0.0:
        raise ValueError(f"terminal_error_bound must be >= 0, got {err}")

    dim = np.asarray(m_seq[0], dtype=np.float64).shape[0]
    running = np.eye(dim)  # Phi_{h-1, h}, used at k = h-1
    per_step: list[float] = [0.0] * horizon
    for k in range(horizon - 1, -1, -1):
        dpi_dtheta = np.asarray(dpi_dtheta_seq[k], dtype=np.float64)
        b_k = np.asarray(b_seq[k], dtype=np.float64)
        term = _op_norm_sym(dpi_dtheta) * _op_norm_sym(b_k) * _op_norm_sym(running) * err
        per_step[k] = term
        m_k = np.asarray(m_seq[k], dtype=np.float64)
        running = running @ m_k
    bound = float(math.fsum(per_step))
    if not math.isfinite(bound):
        raise ValueError("truncation_bias_bound produced a non-finite bound")
    return GradientBiasReport(
        bound=bound,
        terminal_error_bound=err,
        horizon=horizon,
        per_step_terms=per_step,
        certified=True,
    )


def terminal_adjoint_error_bound(
    error_net: Network,
    input_box: Sequence[IntervalLike],
    point_residual: float,
    *,
    norm: str = "inf",
) -> float:
    r"""Sound Lipschitz-extension bound on ``sup_box ||lambda_pred(y) - lambda_true(y)||``.

    ``error_net`` must represent the caller's model of the error map
    ``e(y) = lambda_pred(y) - lambda_true(y)`` in :mod:`omnibias.verify`'s
    layer format; ``point_residual`` is ``||e(y0)||`` measured at the box
    centre ``y0`` (by the caller, e.g. by evaluating the true terminal cost
    gradient against the learned head on held-out rollouts). Returns

    .. math::
        |e(y_0)| + L \cdot \mathrm{radius}(\text{box})

    where ``L`` is :func:`omnibias.verify.lipschitz_bound` over ``input_box``
    and ``radius`` is the box's half-width in the matching induced norm.
    This is sound *given* that ``error_net`` and ``point_residual``
    genuinely model ``e``; neither is checked here.
    """
    residual = float(point_residual)
    if residual < 0.0:
        raise ValueError(f"point_residual must be >= 0, got {residual}")
    l_bound = lipschitz_bound(error_net, input_box, norm=norm)
    intervals = [Interval.from_value(v) for v in input_box]
    if norm == "inf":
        radius = max((iv.width / 2.0 for iv in intervals), default=0.0)
    elif norm == "l1":
        radius = float(math.fsum(iv.width / 2.0 for iv in intervals))
    elif norm == "l2":
        radius = math.sqrt(math.fsum((iv.width / 2.0) ** 2 for iv in intervals))
    else:
        raise ValueError(f"unknown norm {norm!r}; choose 'inf', 'l1' or 'l2'")
    bound = residual + l_bound * radius
    if not math.isfinite(bound):
        raise ValueError("terminal_adjoint_error_bound produced a non-finite bound")
    return bound


def worked_example() -> dict[str, object]:
    """G6 reference: a hand-built error net with a known Lipschitz constant."""
    from omnibias.verify import LinearLayer, Network

    weight = ((0.5, 0.0), (0.0, 0.5))
    net = Network([LinearLayer(weight=weight, bias=(0.0, 0.0))])
    box = [Interval(-0.1, 0.1), Interval(-0.1, 0.1)]
    bound = terminal_adjoint_error_bound(net, box, point_residual=0.01)
    # L=0.5 (row-sum), radius=0.1 in inf-norm -> bound = 0.01 + 0.05 = 0.06
    m = np.array([[0.8, 0.0], [0.0, 0.8]])
    dpi = np.array([[0.3, 0.0]])
    b = np.array([[1.0], [0.0]])
    report = truncation_bias_bound([dpi, dpi], [b, b], [m, m], bound)
    return {
        "terminal_error_bound": bound,
        "expected_terminal_error_bound": 0.06,
        "gradient_bias_bound": report.bound,
        "g6_earned": abs(bound - 0.06) < 1e-9 and report.certified,
    }


def gradient_bias_skill(*, n: int = 200, seed: int = 0) -> dict[str, object]:
    r"""G6: random ``(theta, y, mu)``-style draws never violate a sampled true bias.

    Builds random small linear systems where the "true" bias is computable
    exactly (a closed linear recursion with a *known* terminal perturbation),
    then checks the certified bound never falls below the realized bias.
    """
    rng = random.Random(seed)
    coverage = 0
    widths: list[float] = []
    for _ in range(n):
        dim = rng.choice((1, 2, 3))
        n_params = rng.choice((1, 2, 4))
        action_dim = rng.choice((1, 2))
        horizon = rng.randint(1, 6)
        m_seq = [
            0.7 * np.eye(dim) + 0.05 * (rng.random() - 0.5) * np.eye(dim) for _ in range(horizon)
        ]
        dpi_seq = [
            0.1 * (rng.random() + 0.1) * rng.choice((-1.0, 1.0)) * np.ones((action_dim, n_params))
            for _ in range(horizon)
        ]
        b_seq = [
            0.1 * (rng.random() + 0.1) * rng.choice((-1.0, 1.0)) * np.ones((dim, action_dim))
            for _ in range(horizon)
        ]
        err_true = np.array([rng.uniform(-1.0, 1.0) for _ in range(dim)])
        err_bound = float(np.max(np.abs(err_true)))
        report = truncation_bias_bound(dpi_seq, b_seq, m_seq, err_bound)

        running = np.eye(dim)  # Phi_{k+1}
        realized = np.zeros(n_params)
        for k in range(horizon - 1, -1, -1):
            state_vec = running.T @ err_true  # Phi_{k+1}^T @ delta_lambda_h
            action_vec = b_seq[k].T @ state_vec  # B_k^T @ (...)
            realized = realized + dpi_seq[k].T @ action_vec  # dpi/dtheta_k^T @ (...)
            running = running @ m_seq[k]
        realized_norm = float(np.max(np.abs(realized))) if realized.size else 0.0
        if report.bound >= realized_norm - 1e-9:
            coverage += 1
        widths.append(report.bound)
    return {
        "coverage": coverage / n,
        "median_bound": statistics.median(widths) if widths else math.inf,
        "g6_earned": coverage == n,
    }


__all__ = [
    "DISCLAIMER",
    "GradientBiasReport",
    "gradient_bias_skill",
    "honesty_payload",
    "terminal_adjoint_error_bound",
    "truncation_bias_bound",
    "worked_example",
]
