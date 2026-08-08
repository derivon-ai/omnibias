# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Causal time-marching training driver (JAX).

Bit-identical twin of :mod:`omnibias.pinn.train.torch.march` in the sense that
the window geometry, causal weights, and diagnostics are shared pure-Python /
numpy; only the optimiser step differs (hand-rolled Adam vs torch Adam).

No ``optax`` dependency: Adam is implemented with ``jax.tree_util`` so the
``omnibias-pinn[jax]`` extra stays lean.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.marching import TimeMarcher, TimeWindowSchedule
from omnibias.pinn.jax.losses.causal import (
    causal_residual_loss,
    causal_weights_from_per_bin,
)
from omnibias.pinn.train._core.causality import CausalityReport, report_causality
from omnibias.pinn.train._core.guards import (
    TrivialSolutionVerdict,
    trivial_solution_guard,
)

ResidualFn = Callable[[Any, Array], Array]
ValueFn = Callable[[Any, Array], Array]
ICFn = Callable[[Array], Array]
ApplyFn = Callable[[Any, Array], Array]
HardICFactory = Callable[[Any, Array, Array], Any]
# Factory returns (params, opt_state) after optionally wrapping params.
OptimizerFactory = Callable[[Any], tuple[Any, Any]]


@dataclass(frozen=True)
class WindowResult:
    """Outcome of one marched window."""

    window_index: int
    bounds: tuple[float, float]
    epsilon: float
    converged: bool
    exhausted: bool
    final_loss: float
    causality: CausalityReport
    steps_run: int
    seam_mse: float | None = None
    handoff_values: tuple[float, ...] | None = None


@dataclass
class MarchResult:
    """Aggregate outcome of :func:`march_solve`."""

    windows: list[WindowResult] = field(default_factory=list)
    trivial: TrivialSolutionVerdict | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    params: Any = None

    @property
    def all_converged(self) -> bool:
        return bool(self.windows) and all(w.converged for w in self.windows)


def _reshape_residual(resid: Array, n_bins: int, pb: int) -> Array:
    """Shape residual to ``(n_bins, per_bin, ...)`` then RMS over components."""
    if resid.shape[0] != n_bins * pb:
        raise ValueError(
            f"residual leading dim {resid.shape[0]} incompatible "
            f"with n_bins*per_bin={n_bins * pb}"
        )
    resid_t = resid.reshape(n_bins, pb, *resid.shape[1:])
    if resid_t.ndim > 2:
        resid_t = jnp.sqrt(
            jnp.mean(resid_t**2, axis=tuple(range(2, resid_t.ndim)))
        )
    return resid_t


def _adam_init(params: Any) -> dict[str, Any]:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.asarray(0)}


def _adam_step(
    params: Any,
    grads: Any,
    state: dict[str, Any],
    *,
    lr: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Any, dict[str, Any]]:
    t = state["t"] + 1
    m = jax.tree_util.tree_map(
        lambda mi, g: b1 * mi + (1.0 - b1) * g, state["m"], grads
    )
    v = jax.tree_util.tree_map(
        lambda vi, g: b2 * vi + (1.0 - b2) * (g * g), state["v"], grads
    )
    # Bias correction with a scalar t shared across the tree.
    bc1 = 1.0 - b1**t
    bc2 = 1.0 - b2**t

    def _update(p: Array, mi: Array, vi: Array) -> Array:
        mhat = mi / bc1
        vhat = vi / bc2
        return p - lr * mhat / (jnp.sqrt(vhat) + eps)

    params = jax.tree_util.tree_map(_update, params, m, v)
    return params, {"m": m, "v": v, "t": t}


def march_solve(
    params: Any,
    apply_fn: ApplyFn,
    residual_fn: ResidualFn,
    coordinate_spec: CoordinateSpec,
    schedule: TimeWindowSchedule,
    *,
    steps_per_window: int = 50,
    max_steps_per_window: int | None = None,
    lr: float = 1e-3,
    optimizer: OptimizerFactory | None = None,
    opt_state: Any | None = None,
    per_bin: int = 16,
    n_slice: int = 32,
    ic_values: np.ndarray | Array | None = None,
    ic_fn: ICFn | None = None,
    ic_weight: float = 1.0,
    ic_mode: str = "soft",
    hard_ic_factory: HardICFactory | None = None,
    value_fn: ValueFn | None = None,
    seed: int = 0,
    check_trivial: bool = True,
    trivial_ratio: float = 1e-3,
    trivial_mode: str = "variance",
    advance_policy: str = "gate",
    resample_every: int = 0,
    stop_on_exhaust: bool = False,
) -> MarchResult:
    """March a PINN solve window-by-window with causal residual weighting (JAX).

    Parameters
    ----------
    params
        PyTree of trainable parameters.
    apply_fn
        ``apply_fn(params, coords) -> values`` used as the default value path.
    residual_fn
        ``residual_fn(params, coords) -> residual`` with the same shape contract
        as the torch twin.
    coordinate_spec, schedule
        Shared with the torch twin.
    steps_per_window, max_steps_per_window
        Attempt size and hard cap; windows retry until the advance gate passes
        or the cap is hit.
    optimizer
        Optional factory ``params -> (params, opt_state)`` that replaces the
        default Adam initialisation (rarely needed).
    opt_state
        Optional pre-built Adam state ``{"m", "v", "t"}``.
    advance_policy
        ``"gate"`` refuses to advance an unconverged window; ``"force"``
        advances anyway.
    """
    if ic_mode not in ("soft", "hard"):
        raise ValueError(f"ic_mode must be 'soft' or 'hard', got {ic_mode!r}")
    if advance_policy not in ("gate", "force"):
        raise ValueError(
            f"advance_policy must be 'gate' or 'force', got {advance_policy!r}"
        )
    if steps_per_window < 1:
        raise ValueError(f"steps_per_window must be >= 1, got {steps_per_window}")
    if max_steps_per_window is None:
        max_steps_per_window = int(steps_per_window)
    if max_steps_per_window < steps_per_window:
        raise ValueError("max_steps_per_window must be >= steps_per_window")
    if ic_values is None and ic_fn is None:
        raise ValueError("provide ic_values or ic_fn; silent zero IC is refused")
    if ic_values is not None and ic_fn is not None:
        raise ValueError("provide only one of ic_values or ic_fn")

    marcher = TimeMarcher(
        coordinate_spec,
        schedule,
        per_bin=per_bin,
        n_slice=n_slice,
        seed=seed,
    )

    def _default_value(p: Any, coords: Array) -> Array:
        if value_fn is not None:
            return value_fn(p, coords)
        out = apply_fn(p, coords)
        if out.ndim == 1:
            return out
        return out[..., 0]

    ic_pts = jnp.asarray(marcher.initial_points())
    if ic_values is not None:
        ic0 = np.asarray(ic_values, dtype=float)
    else:
        assert ic_fn is not None
        ic0 = np.asarray(ic_fn(ic_pts), dtype=float)
    marcher.set_initial(ic0.reshape(n_slice, -1) if ic0.ndim > 1 else ic0)

    if ic_mode == "hard" and hard_ic_factory is not None:
        ic_u0 = jnp.asarray(marcher.initial_values)
        if ic_u0.ndim > 1:
            ic_u0 = ic_u0.reshape(ic_u0.shape[0], -1)[..., 0]
        params = hard_ic_factory(params, ic_pts, ic_u0)

    if optimizer is not None:
        params, opt_state = optimizer(params)
    elif opt_state is None:
        opt_state = _adam_init(params)

    result = MarchResult(
        diagnostics={
            "ic_mode": ic_mode,
            "lr": lr,
            "advance_policy": advance_policy,
            "max_steps_per_window": max_steps_per_window,
        }
    )
    last_handoff: np.ndarray | None = None
    last_pred_ic: np.ndarray | None = None
    last_window_ic: np.ndarray | None = None

    while not marcher.done:
        eps = float(marcher.epsilon)
        ic_x = jnp.asarray(marcher.initial_points())
        ic_u = jnp.asarray(marcher.initial_values)
        if ic_u.ndim > 1:
            ic_u = ic_u.reshape(ic_u.shape[0], -1)[..., 0]
        window_ic = np.asarray(ic_u, dtype=float).reshape(-1)

        final_loss = 0.0
        weights_np = np.ones(schedule.n_time_bins, dtype=float)
        L_np = np.zeros(schedule.n_time_bins, dtype=float)
        steps_run = 0
        converged = False
        pts_np = marcher.collocation()
        n_bins, pb, D = pts_np.shape
        coords = jnp.asarray(pts_np.reshape(-1, D))

        while steps_run < max_steps_per_window and not converged:
            block = min(steps_per_window, max_steps_per_window - steps_run)
            for _step_i in range(block):
                if (
                    resample_every > 0
                    and steps_run > 0
                    and steps_run % resample_every == 0
                ):
                    pts_np = marcher.collocation()
                    coords = jnp.asarray(pts_np.reshape(-1, D))

                def loss_fn(
                    p: Any,
                    *,
                    _coords: Array = coords,
                    _n_bins: int = n_bins,
                    _pb: int = pb,
                    _eps: float = eps,
                    _ic_x: Array = ic_x,
                    _ic_u: Array = ic_u,
                ) -> tuple[Array, tuple[Array, Array]]:
                    resid = residual_fn(p, _coords)
                    resid_t = _reshape_residual(resid, _n_bins, _pb)
                    loss_c, w = causal_residual_loss(
                        resid_t, epsilon=_eps, return_weights=True
                    )
                    loss = loss_c
                    if ic_mode == "soft" and ic_weight > 0.0:
                        pred_ic = _default_value(p, _ic_x)
                        loss = loss + float(ic_weight) * jnp.mean(
                            (pred_ic - _ic_u) ** 2
                        )
                    L_per = (resid_t**2).mean(
                        axis=tuple(range(1, resid_t.ndim))
                    )
                    return loss, (w, L_per)

                (loss, (w, L_per)), grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(params)
                params, opt_state = _adam_step(params, grads, opt_state, lr=lr)
                final_loss = float(loss)
                weights_np = np.asarray(w)
                L_np = np.asarray(L_per)
                steps_run += 1

            resid = residual_fn(params, coords)
            resid_t = _reshape_residual(resid, n_bins, pb)
            L_t = (resid_t**2).mean(axis=tuple(range(1, resid_t.ndim)))
            w_t = causal_weights_from_per_bin(L_t, epsilon=eps)
            L_np = np.asarray(L_t)
            weights_np = np.asarray(w_t)
            converged = bool(marcher.observe(weights_np))

        exhausted = not converged
        report = report_causality(L_np, weights_np)

        handoff_pts = jnp.asarray(marcher.handoff_points())
        handoff_vals = np.asarray(_default_value(params, handoff_pts))
        pred_ic_now = np.asarray(_default_value(params, ic_x)).reshape(-1)
        last_handoff = handoff_vals.reshape(-1)
        last_pred_ic = pred_ic_now
        last_window_ic = window_ic
        seam_mse = float(np.mean((pred_ic_now - window_ic) ** 2))

        result.windows.append(
            WindowResult(
                window_index=marcher.window_index,
                bounds=marcher.bounds,
                epsilon=eps,
                converged=converged,
                exhausted=exhausted,
                final_loss=final_loss,
                causality=report,
                steps_run=steps_run,
                seam_mse=seam_mse,
                handoff_values=tuple(float(x) for x in last_handoff),
            )
        )

        if exhausted and advance_policy == "gate":
            break
        marcher.advance(handoff_vals)
        if exhausted and stop_on_exhaust:
            break

    result.params = params
    if (
        check_trivial
        and last_pred_ic is not None
        and last_window_ic is not None
    ):
        result.trivial = trivial_solution_guard(
            last_pred_ic if trivial_mode == "variance" else last_handoff,
            last_window_ic,
            ratio_threshold=trivial_ratio,
            mode=trivial_mode,
        )
    return result


__all__ = [
    "MarchResult",
    "WindowResult",
    "march_solve",
]
