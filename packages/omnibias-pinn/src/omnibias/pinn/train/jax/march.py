# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Causal time-marching training driver (JAX).

Bit-identical twin of :mod:`omnibias.pinn.train.torch.march` in the sense that
the window geometry, causal weights, and diagnostics are shared pure-Python /
numpy; only the optimiser step differs (optax Adam vs torch Adam).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
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


@dataclass(frozen=True)
class WindowResult:
    """Outcome of one marched window."""

    window_index: int
    bounds: tuple[float, float]
    epsilon: float
    converged: bool
    final_loss: float
    causality: CausalityReport
    steps_run: int


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


def march_solve(
    params: Any,
    apply_fn: ApplyFn,
    residual_fn: ResidualFn,
    coordinate_spec: CoordinateSpec,
    schedule: TimeWindowSchedule,
    *,
    steps_per_window: int = 50,
    lr: float = 1e-3,
    per_bin: int = 16,
    n_slice: int = 32,
    ic_values: np.ndarray | Array | None = None,
    ic_fn: ICFn | None = None,
    ic_weight: float = 1.0,
    ic_mode: str = "soft",
    value_fn: ValueFn | None = None,
    seed: int = 0,
    check_trivial: bool = True,
    trivial_ratio: float = 1e-3,
) -> MarchResult:
    """March a PINN solve window-by-window with causal residual weighting (JAX).

    Parameters
    ----------
    params
        PyTree of trainable parameters.
    apply_fn
        ``apply_fn(params, coords) -> values`` used as the default value path
        and as the field handle passed to ``residual_fn``.
    residual_fn
        ``residual_fn(params, coords) -> residual`` with the same shape contract
        as the torch twin.
    coordinate_spec, schedule
        Shared with the torch twin.
    """
    if ic_mode not in ("soft", "hard"):
        raise ValueError(f"ic_mode must be 'soft' or 'hard', got {ic_mode!r}")
    if steps_per_window < 1:
        raise ValueError(f"steps_per_window must be >= 1, got {steps_per_window}")

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
    elif ic_fn is not None:
        ic0 = np.asarray(ic_fn(ic_pts), dtype=float)
    else:
        ic0 = np.zeros((n_slice,), dtype=float)
    marcher.set_initial(ic0.reshape(n_slice, -1) if ic0.ndim > 1 else ic0)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    result = MarchResult(diagnostics={"ic_mode": ic_mode, "lr": lr})
    first_ic = np.asarray(marcher.initial_values, dtype=float).reshape(-1)
    last_handoff: np.ndarray | None = None

    while not marcher.done:
        pts_np = marcher.collocation()
        n_bins, pb, D = pts_np.shape
        coords = jnp.asarray(pts_np.reshape(-1, D))
        eps = float(marcher.epsilon)
        ic_x = jnp.asarray(marcher.initial_points())
        ic_u = jnp.asarray(marcher.initial_values, dtype=coords.dtype)
        if ic_u.ndim > 1:
            ic_u = ic_u.reshape(ic_u.shape[0], -1)[..., 0]

        def loss_fn(p: Any) -> tuple[Array, tuple[Array, Array]]:
            resid = residual_fn(p, coords)
            resid_t = resid.reshape(n_bins, pb, *resid.shape[1:])
            while resid_t.ndim > 2:
                resid_t = resid_t.mean(axis=-1)
            loss_c, w = causal_residual_loss(
                resid_t, epsilon=eps, return_weights=True
            )
            loss = loss_c
            if ic_mode == "soft" and ic_weight > 0.0:
                pred_ic = _default_value(p, ic_x)
                loss = loss + float(ic_weight) * jnp.mean((pred_ic - ic_u) ** 2)
            L_per = (resid_t**2).mean(axis=tuple(range(1, resid_t.ndim)))
            return loss, (w, L_per)

        final_loss = 0.0
        weights_np = np.ones(n_bins, dtype=float)
        L_np = np.zeros(n_bins, dtype=float)

        for _step in range(steps_per_window):
            (loss, (w, L_per)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            final_loss = float(loss)
            weights_np = np.asarray(w)
            L_np = np.asarray(L_per)

        # Fresh measurement for the advance criterion.
        resid = residual_fn(params, coords)
        resid_t = resid.reshape(n_bins, pb, *resid.shape[1:])
        while resid_t.ndim > 2:
            resid_t = resid_t.mean(axis=-1)
        L_t = (resid_t**2).mean(axis=tuple(range(1, resid_t.ndim)))
        w_t = causal_weights_from_per_bin(L_t, epsilon=eps)
        L_np = np.asarray(L_t)
        weights_np = np.asarray(w_t)

        converged = bool(marcher.observe(weights_np))
        report = report_causality(L_np, weights_np)

        handoff_pts = jnp.asarray(marcher.handoff_points())
        handoff_vals = np.asarray(_default_value(params, handoff_pts))
        last_handoff = handoff_vals.reshape(-1)

        result.windows.append(
            WindowResult(
                window_index=marcher.window_index,
                bounds=marcher.bounds,
                epsilon=eps,
                converged=converged,
                final_loss=final_loss,
                causality=report,
                steps_run=steps_per_window,
            )
        )
        marcher.advance(handoff_vals)

    result.params = params
    if check_trivial and last_handoff is not None and first_ic.size > 0:
        result.trivial = trivial_solution_guard(
            last_handoff,
            first_ic,
            ratio_threshold=trivial_ratio,
            mode="energy",
        )
    return result


__all__ = [
    "MarchResult",
    "WindowResult",
    "march_solve",
]
