# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Discrete adjoint (costate) recursion for optimal control (theory 10-02).

Backend-free half of the jet-adjoint policy-optimization stack. Given a
locally linearized closed-loop transition ``y_{k+1} ~ A_k y_k + B_k u_k``
under a state-feedback policy ``u_k = pi(y_k; theta)``, the discrete
Pontryagin / adjoint recursion is

.. math::
    \lambda_k = M_k^\top \lambda_{k+1} + c_k, \qquad
    M_k = A_k + B_k \,\partial\pi/\partial y_k, \qquad
    c_k = \partial L/\partial y_k + \partial L/\partial u_k \,\partial\pi/\partial y_k

and the policy gradient assembles from the same quantities:

.. math::
    \nabla_\theta J = \sum_k \Big(\partial L/\partial u_k
    + B_k^\top \lambda_{k+1}\Big)\, \partial\pi/\partial\theta_k .

This is the classical costate recursion behind reverse-mode
backpropagation-through-time and the DDP / iLQR backward pass; it is
**not** a new algorithm. What is new is *where the pieces come from*: the
closed-loop Jacobian ``M_k`` and the parameter Jacobian
``dpi/dtheta_k`` are supplied by the caller as plain ``numpy`` arrays, and
in :mod:`omnibias.control.torch.adjoint` / :mod:`omnibias.control.jax.adjoint`
those arrays come from the exact directional-jet kernels
(``mlp_jet`` / ``layer_jet`` / ``compose_jet``), not from a finite
difference and not from truncated unrolled autodiff. This module performs
no such tower evaluation itself -- it is pure linear-algebra recursion and
never imports ``torch``, ``jax``, ``tensorflow``, or ``keras``.

:func:`discrete_riccati_sweep` is the finite-horizon time-varying LQR
Riccati backward sweep (a *matrix* generalisation of
:func:`omnibias.core.control_lqr.scalar_finite_horizon_lqr`, which is a
1-D directional restriction, not this). For an LQR problem run through
:func:`adjoint_recursion`, the analytic costate is exactly
``lambda_k = P_k y_k`` (:func:`lqr_costate_from_riccati`) -- the reference
used to certify the recursion is implemented correctly (theory 10-02 gate
G2), not a claim that a learned policy's adjoint equals ``P_k y_k`` in
general.

:func:`n_step_adjoint_bootstrap` and :func:`td_lambda_mix` implement the
TD(:math:`\lambda`)-style value-gradient bootstrap: an ``n``-step target
propagates a *learned* terminal adjoint estimate backward through ``n``
exact recursion steps, and the geometric blend over ``n`` is the same
mixing PEARL uses to learn the adjoint as a terminal correction instead of
learning the value function directly.

This is **not** a plant controller, not the infinite-horizon algebraic
Riccati / DARE, not a solved Hamilton-Jacobi-Bellman equation, and not a
claim that policy-gradient training finds a global optimum of ``J(theta)``
-- it finds a stationary point. ``theorem_prover_verified`` is not
asserted.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DISCLAIMER = (
    "discrete Pontryagin / DDP costate recursion; backend-free; not DARE, "
    "not a solved HJB, not a plant controller, not a global-optimum claim"
)

_SINGULAR = 1e-14


def honesty_payload() -> dict[str, bool]:
    return {
        "plant_controller_claimed": False,
        "algebraic_riccati_claimed": False,
        "solved_hjb_claimed": False,
        "global_optimum_claimed": False,
        "new_algorithm_claimed": False,
        "stretch_claim": False,
        "continuum_claimed": False,
        "theorem_prover_verified": False,
        "skips_chain_rule": False,
    }


def _as_matrix(name: str, value: Sequence[Sequence[float]]) -> FloatArray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out


def _as_vector(name: str, value: Sequence[float]) -> FloatArray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out


def closed_loop_jacobian(a: FloatArray, b: FloatArray, dpi_dy: FloatArray) -> FloatArray:
    r"""``M_k = A_k + B_k @ dpi/dy_k``, the closed-loop state Jacobian."""
    a_m = _as_matrix("a", a)
    b_m = _as_matrix("b", b)
    d_m = _as_matrix("dpi_dy", dpi_dy)
    out = a_m + b_m @ d_m
    if out.shape != a_m.shape:
        raise ValueError(f"closed_loop_jacobian shape mismatch: {out.shape} vs {a_m.shape}")
    return out


def total_state_cost_gradient(
    dl_dy: FloatArray, dl_du: FloatArray, dpi_dy: FloatArray
) -> FloatArray:
    r"""``c_k = dL/dy_k + dL/du_k @ dpi/dy_k`` (total derivative through the policy)."""
    y_m = _as_vector("dl_dy", dl_dy)
    u_m = _as_vector("dl_du", dl_du)
    d_m = _as_matrix("dpi_dy", dpi_dy)
    out = y_m + d_m.T @ u_m
    return out


def adjoint_step(m: FloatArray, c: FloatArray, lambda_next: FloatArray) -> FloatArray:
    r"""One backward costate step: ``lambda_k = M_k^T lambda_{k+1} + c_k``."""
    m_mat = _as_matrix("m", m)
    c_vec = _as_vector("c", c)
    lam = _as_vector("lambda_next", lambda_next)
    out: FloatArray = m_mat.T @ lam + c_vec
    if not np.all(np.isfinite(out)):
        raise ValueError("adjoint_step produced a non-finite costate")
    return out


def adjoint_recursion(
    m_seq: Sequence[FloatArray],
    c_seq: Sequence[FloatArray],
    lambda_terminal: FloatArray,
) -> list[FloatArray]:
    r"""Backward sweep. Returns ``[lambda_0, ..., lambda_H]`` with ``lambda_H`` terminal.

    ``m_seq[k]`` is ``M_k`` for ``k = 0 .. H-1``; ``c_seq`` matches in length.
    """
    if len(m_seq) != len(c_seq):
        raise ValueError(f"m_seq and c_seq must match length, got {len(m_seq)} vs {len(c_seq)}")
    lam = _as_vector("lambda_terminal", lambda_terminal)
    costates = [lam]
    for m, c in zip(reversed(m_seq), reversed(c_seq), strict=True):
        lam = adjoint_step(m, c, lam)
        costates.append(lam)
    costates.reverse()
    return costates


def hamiltonian_control_gradient(
    dl_du: FloatArray, b: FloatArray, lambda_next: FloatArray
) -> FloatArray:
    r"""``dH/du_k = dL/du_k + B_k^T lambda_{k+1}``."""
    u_vec = _as_vector("dl_du", dl_du)
    b_mat = _as_matrix("b", b)
    lam = _as_vector("lambda_next", lambda_next)
    out: FloatArray = u_vec + b_mat.T @ lam
    return out


def policy_gradient_term(h_u: FloatArray, dpi_dtheta: FloatArray) -> FloatArray:
    r"""One step's contribution to ``grad_theta J``: ``dpi/dtheta_k^T dH/du_k``."""
    h_vec = _as_vector("h_u", h_u)
    j_mat = _as_matrix("dpi_dtheta", dpi_dtheta)
    if j_mat.shape[0] != h_vec.shape[0]:
        raise ValueError(
            f"dpi_dtheta rows ({j_mat.shape[0]}) must match control dim ({h_vec.shape[0]})"
        )
    out: FloatArray = j_mat.T @ h_vec
    return out


def policy_gradient(
    dl_du_seq: Sequence[FloatArray],
    b_seq: Sequence[FloatArray],
    dpi_dtheta_seq: Sequence[FloatArray],
    lambda_seq: Sequence[FloatArray],
) -> FloatArray:
    r"""``grad_theta J = sum_k dpi/dtheta_k^T (dL/du_k + B_k^T lambda_{k+1})``.

    ``lambda_seq`` is the full costate list from :func:`adjoint_recursion`
    (length ``H+1``); ``lambda_seq[k+1]`` is used at step ``k``.
    """
    horizon = len(dl_du_seq)
    if not (len(b_seq) == len(dpi_dtheta_seq) == horizon):
        raise ValueError("dl_du_seq, b_seq, dpi_dtheta_seq must have equal length")
    if len(lambda_seq) != horizon + 1:
        raise ValueError(f"lambda_seq must have length horizon+1={horizon + 1}")
    total: FloatArray | None = None
    for k in range(horizon):
        h_u = hamiltonian_control_gradient(dl_du_seq[k], b_seq[k], lambda_seq[k + 1])
        term = policy_gradient_term(h_u, dpi_dtheta_seq[k])
        total = term if total is None else total + term
    if total is None:
        raise ValueError("policy_gradient requires horizon >= 1")
    return total


@dataclass(frozen=True)
class RiccatiSweep:
    """Backward finite-horizon time-varying LQR Riccati sweep result."""

    p_seq: list[FloatArray]
    k_seq: list[FloatArray]


def discrete_riccati_sweep(
    a_seq: Sequence[FloatArray],
    b_seq: Sequence[FloatArray],
    q_seq: Sequence[FloatArray],
    r_seq: Sequence[FloatArray],
    qf: FloatArray,
) -> RiccatiSweep:
    r"""Time-varying discrete LQR Riccati backward sweep.

    Returns ``p_seq = [P_0, ..., P_H]`` (``P_H = Qf``) and
    ``k_seq = [K_0, ..., K_{H-1}]`` with ``u_k = -K_k y_k``. Matrix
    generalisation of :func:`omnibias.core.control_lqr.scalar_finite_horizon_lqr`
    (a 1-D directional restriction); this operates on the full state.
    """
    horizon = len(a_seq)
    if not (len(b_seq) == len(q_seq) == len(r_seq) == horizon):
        raise ValueError("a_seq, b_seq, q_seq, r_seq must have equal length")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    p_next = _as_matrix("qf", qf)
    p_rev = [p_next]
    k_rev: list[FloatArray] = []
    for a, b, q, r in zip(
        reversed(a_seq), reversed(b_seq), reversed(q_seq), reversed(r_seq), strict=True
    ):
        a_m, b_m, q_m, r_m = (
            _as_matrix("a", a),
            _as_matrix("b", b),
            _as_matrix("q", q),
            _as_matrix("r", r),
        )
        s = r_m + b_m.T @ p_next @ b_m
        try:
            gain: FloatArray = np.linalg.solve(s, b_m.T @ p_next @ a_m)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Riccati solve is singular (R + B^T P B is not invertible); "
                "need R > 0 or a well-posed problem"
            ) from exc
        p_next = q_m + a_m.T @ p_next @ (a_m - b_m @ gain)
        p_next = 0.5 * (p_next + p_next.T)
        if not np.all(np.isfinite(p_next)) or not np.all(np.isfinite(gain)):
            raise ValueError("Riccati sweep produced a non-finite cost-to-go or gain")
        p_rev.append(p_next)
        k_rev.append(gain)
    return RiccatiSweep(p_seq=list(reversed(p_rev)), k_seq=list(reversed(k_rev)))


def lqr_costate_from_riccati(p_seq: Sequence[FloatArray], y_seq: Sequence[FloatArray]) -> list[FloatArray]:
    r"""Analytic LQR costate ``lambda_k = P_k y_k`` (gate G2 reference)."""
    if len(p_seq) != len(y_seq):
        raise ValueError(f"p_seq and y_seq must match length, got {len(p_seq)} vs {len(y_seq)}")
    return [_as_matrix("p_k", p) @ _as_vector("y_k", y) for p, y in zip(p_seq, y_seq, strict=True)]


def n_step_adjoint_bootstrap(
    m_seq: Sequence[FloatArray],
    c_seq: Sequence[FloatArray],
    lambda_terminal_pred: FloatArray,
) -> FloatArray:
    r"""``G_k^{(n)}``: propagate a *learned* terminal estimate back ``n = len(m_seq)`` steps.

    ``m_seq`` / ``c_seq`` are the exact per-step recursion pieces on the
    window; only the terminal value is the (possibly imperfect) bootstrap.
    """
    costates = adjoint_recursion(m_seq, c_seq, lambda_terminal_pred)
    return costates[0]


def td_lambda_mix(bootstraps: Sequence[FloatArray], td_lambda: float) -> FloatArray:
    r"""Geometric TD(:math:`\lambda`) blend of ``n``-step bootstraps ``G^{(1)}, ..., G^{(N)}``.

    .. math::
        \bar\lambda = (1-\lambda) \sum_{n=1}^{N-1} \lambda^{n-1} G^{(n)}
        + \lambda^{N-1} G^{(N)}

    ``td_lambda = 0`` recovers the pure 1-step bootstrap; ``td_lambda`` close
    to ``1`` weights long lookaheads (closer to the uncorrected recursion).
    """
    if not bootstraps:
        raise ValueError("bootstraps must be non-empty")
    lam = float(td_lambda)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"td_lambda must be in [0, 1], got {lam}")
    n_total = len(bootstraps)
    weights = [((1.0 - lam) * lam**n) for n in range(n_total - 1)]
    weights.append(lam ** (n_total - 1))
    total_weight = sum(weights)
    if n_total == 1:
        weights = [1.0]
        total_weight = 1.0
    acc: FloatArray | None = None
    for w, g in zip(weights, bootstraps, strict=True):
        term = (w / total_weight) * _as_vector("bootstrap", g)
        acc = term if acc is None else acc + term
    assert acc is not None
    if not np.all(np.isfinite(acc)):
        raise ValueError("td_lambda_mix produced a non-finite costate")
    return acc


def worked_example() -> dict[str, float | bool]:
    """G1/G2: a fixed 2-state LQR problem; recursion matches ``P_k y_k`` exactly."""
    a = np.array([[1.0, 0.1], [0.0, 1.0]])
    b = np.array([[0.0], [0.1]])
    q = np.array([[1.0, 0.0], [0.0, 0.1]])
    r = np.array([[0.5]])
    qf = np.array([[2.0, 0.0], [0.0, 2.0]])
    horizon = 6
    sweep = discrete_riccati_sweep([a] * horizon, [b] * horizon, [q] * horizon, [r] * horizon, qf)

    y = np.array([1.0, -0.5])
    y_seq = [y]
    u_seq = []
    for k in range(horizon):
        u = -sweep.k_seq[k] @ y
        u_seq.append(u)
        y = a @ y + b @ u
        y_seq.append(y)

    m_seq = [closed_loop_jacobian(a, b, -sweep.k_seq[k]) for k in range(horizon)]
    c_seq = [
        total_state_cost_gradient(q @ y_seq[k], r @ u_seq[k], -sweep.k_seq[k])
        for k in range(horizon)
    ]
    lambda_terminal = qf @ y_seq[horizon]
    lambda_seq = adjoint_recursion(m_seq, c_seq, lambda_terminal)
    reference = lqr_costate_from_riccati(sweep.p_seq, y_seq)

    max_abs_err = max(
        float(np.max(np.abs(lam - ref))) for lam, ref in zip(lambda_seq, reference, strict=True)
    )
    return {
        "max_abs_err": max_abs_err,
        "g2_earned": max_abs_err < 1e-10,
        "horizon": horizon,
    }


def adjoint_skill(*, n: int = 20, seed: int = 0) -> dict[str, object]:
    """G-style skill: random small LQR systems recover the analytic costate."""
    rng = random.Random(seed)
    errs: list[float] = []
    finite = True
    for _ in range(n):
        dim = rng.choice((1, 2, 3))
        a = np.eye(dim) + 0.1 * rng.random() * np.eye(dim)
        b = 0.5 * np.eye(dim)
        q = np.eye(dim)
        r = 0.5 * np.eye(dim)
        qf = 2.0 * np.eye(dim)
        horizon = rng.randint(2, 8)
        sweep = discrete_riccati_sweep(
            [a] * horizon, [b] * horizon, [q] * horizon, [r] * horizon, qf
        )
        y = np.array([rng.uniform(-1.0, 1.0) for _ in range(dim)])
        y_seq = [y]
        u_seq = []
        for k in range(horizon):
            u = -sweep.k_seq[k] @ y
            u_seq.append(u)
            y = a @ y + b @ u
            y_seq.append(y)
        m_seq = [closed_loop_jacobian(a, b, -sweep.k_seq[k]) for k in range(horizon)]
        c_seq = [
            total_state_cost_gradient(q @ y_seq[k], r @ u_seq[k], -sweep.k_seq[k])
            for k in range(horizon)
        ]
        lambda_seq = adjoint_recursion(m_seq, c_seq, qf @ y_seq[horizon])
        reference = lqr_costate_from_riccati(sweep.p_seq, y_seq)
        for lam, ref in zip(lambda_seq, reference, strict=True):
            if not np.all(np.isfinite(lam)):
                finite = False
            errs.append(float(np.max(np.abs(lam - ref))))
    singular = False
    try:
        bad_b = np.zeros((1, 1))
        discrete_riccati_sweep([np.eye(1)], [bad_b], [np.zeros((1, 1))], [np.zeros((1, 1))], np.eye(1))
    except ValueError as exc:
        singular = "singular" in str(exc)
    median = statistics.median(errs) if errs else math.inf
    return {
        "median_err": median,
        "all_finite": finite,
        "singular_raised": singular,
        "g2_earned": finite and median < 1e-10 and singular,
    }


__all__ = [
    "DISCLAIMER",
    "FloatArray",
    "RiccatiSweep",
    "adjoint_recursion",
    "adjoint_skill",
    "adjoint_step",
    "closed_loop_jacobian",
    "discrete_riccati_sweep",
    "hamiltonian_control_gradient",
    "honesty_payload",
    "lqr_costate_from_riccati",
    "n_step_adjoint_bootstrap",
    "policy_gradient",
    "policy_gradient_term",
    "td_lambda_mix",
    "total_state_cost_gradient",
    "worked_example",
]
