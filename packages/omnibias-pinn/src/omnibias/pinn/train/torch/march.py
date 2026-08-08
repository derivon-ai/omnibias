# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Causal time-marching training driver (torch).

This is the optimiser loop that
:class:`~omnibias.pinn._core.marching.TimeMarcher` was written for: per-window
collocation, Wang-Perdikaris causal weights fed back through
``marcher.observe``, warm-start handoff, and a
:class:`MarchResult` carrying per-window causality reports.

Windows retry until the advance criterion passes or ``max_steps_per_window``
is exhausted. An unconverged window is *not* silently advanced unless
``advance_policy="force"``.

``ic_mode="hard"`` expects a field that already embeds the IC (for example a
:class:`~omnibias.pinn.torch.cage.ConstrainedExpressionField` built by the
caller or returned from ``hard_ic_factory``); this driver never forges a
structural cage on its own.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.marching import TimeMarcher, TimeWindowSchedule
from omnibias.pinn.torch.losses.causal import (
    causal_residual_loss,
    causal_weights_from_per_bin,
)
from omnibias.pinn.train._core.causality import CausalityReport, report_causality
from omnibias.pinn.train._core.guards import (
    TrivialSolutionVerdict,
    trivial_solution_guard,
)
from torch import Tensor
from torch.optim import Optimizer

ResidualFn = Callable[[nn.Module, Tensor], Tensor]
ValueFn = Callable[[nn.Module, Tensor], Tensor]
ICFn = Callable[[Tensor], Tensor]
HardICFactory = Callable[[nn.Module, Tensor, Tensor], nn.Module]
OptimizerFactory = Callable[[Sequence[nn.Parameter]], Optimizer]


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
    field: nn.Module | None = None

    @property
    def all_converged(self) -> bool:
        return bool(self.windows) and all(w.converged for w in self.windows)


def _reshape_residual(resid: Tensor, n_bins: int, pb: int) -> Tensor:
    """Shape residual to ``(n_bins, per_bin, C...)`` then mean over components."""
    if resid.shape[0] == n_bins * pb:
        resid_t = resid.reshape(n_bins, pb, *resid.shape[1:])
    else:
        try:
            resid_t = resid.reshape(n_bins, pb, *resid.shape[1:])
        except RuntimeError as exc:
            raise ValueError(
                f"residual leading dim {resid.shape[0]} incompatible "
                f"with n_bins*per_bin={n_bins * pb}"
            ) from exc
    # Keep per-component structure for causal weights: average components
    # only after squaring, so opposing signs cannot cancel.
    if resid_t.dim() > 2:
        resid_t = (resid_t**2).mean(dim=tuple(range(2, resid_t.dim()))).sqrt()
    return resid_t


def march_solve(
    field: nn.Module,
    residual_fn: ResidualFn,
    coordinate_spec: CoordinateSpec,
    schedule: TimeWindowSchedule,
    *,
    steps_per_window: int = 50,
    max_steps_per_window: int | None = None,
    lr: float = 1e-3,
    optimizer: Optimizer | OptimizerFactory | None = None,
    per_bin: int = 16,
    n_slice: int = 32,
    ic_values: np.ndarray | Tensor | None = None,
    ic_fn: ICFn | None = None,
    ic_weight: float = 1.0,
    ic_mode: str = "soft",
    hard_ic_factory: HardICFactory | None = None,
    value_fn: ValueFn | None = None,
    seed: int = 0,
    dtype: torch.dtype | None = None,
    check_trivial: bool = True,
    trivial_ratio: float = 1e-3,
    trivial_mode: str = "variance",
    advance_policy: str = "gate",
    resample_every: int = 0,
    stop_on_exhaust: bool = False,
) -> MarchResult:
    """March a PINN solve window-by-window with causal residual weighting.

    Parameters
    ----------
    field
        Trainable field (``nn.Module`` exposing ``parameters()``).
    residual_fn
        ``residual_fn(field, coords) -> residual`` where ``coords`` is
        ``(n_bins * per_bin, D)``.
    coordinate_spec
        Must declare a time axis and explicit domain bounds.
    schedule
        Window ladder (geometry + annealed epsilon + advance tolerance).
    steps_per_window
        Adam steps attempted before each advance check. If the window has
        not unlocked, training continues until ``max_steps_per_window``.
    max_steps_per_window
        Hard cap on optimiser steps per window. Defaults to
        ``steps_per_window`` (single attempt) when omitted; set higher to
        allow retries until the advance gate passes.
    lr
        Learning rate when constructing the default Adam optimiser.
    optimizer
        An existing :class:`~torch.optim.Optimizer`, or a factory
        ``params -> Optimizer``. Defaults to Adam.
    per_bin, n_slice
        Forwarded to :class:`~omnibias.pinn._core.marching.TimeMarcher`.
    ic_values, ic_fn
        Required initial condition. Exactly one must be provided.
    ic_weight
        Soft IC penalty weight (ignored when ``ic_mode="hard"``).
    ic_mode
        ``"soft"`` (default) adds an MSE IC term; ``"hard"`` assumes the
        field already embeds the IC structurally (or builds one via
        ``hard_ic_factory``) and skips the penalty.
    hard_ic_factory
        Optional ``(field, ic_coords, ic_values) -> field`` used once when
        ``ic_mode="hard"`` to wrap the free network in a structural cage.
    value_fn
        ``value_fn(field, coords) -> values`` for handoff / trivial guard.
    seed
        Base seed for the marcher.
    dtype
        Collocation / IC tensor dtype.
    check_trivial
        Run a same-time trivial-solution guard on the final handoff.
    trivial_ratio, trivial_mode
        Forwarded to :func:`~omnibias.pinn.train.trivial_solution_guard`.
    advance_policy
        ``"gate"`` (default) refuses to advance an unconverged window and
        marks it ``exhausted``; ``"force"`` advances anyway (legacy smoke).
    resample_every
        If ``> 0``, redraw collocation points every ``resample_every`` steps
        within a window (adaptive resampling).
    stop_on_exhaust
        If True, stop marching after the first exhausted window.
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

    try:
        _param = next(field.parameters())
        if dtype is None:
            dtype = _param.dtype
        device = _param.device
    except StopIteration:
        if dtype is None:
            dtype = torch.get_default_dtype()
        device = torch.device("cpu")

    marcher = TimeMarcher(
        coordinate_spec,
        schedule,
        per_bin=per_bin,
        n_slice=n_slice,
        seed=seed,
    )

    def _default_value(fld: nn.Module, coords: Tensor) -> Tensor:
        if value_fn is not None:
            return value_fn(fld, coords)
        out = fld(coords) if callable(fld) else fld.forward(coords)  # type: ignore[attr-defined]
        if not isinstance(out, Tensor):
            raise TypeError(
                "field forward must return a Tensor when value_fn is omitted"
            )
        if out.ndim == 1:
            return out
        return out[..., 0]

    ic_pts = torch.as_tensor(marcher.initial_points(), dtype=dtype, device=device)
    if ic_values is not None:
        ic0 = np.asarray(
            ic_values.detach().cpu().numpy()
            if isinstance(ic_values, Tensor)
            else ic_values,
            dtype=float,
        )
    else:
        assert ic_fn is not None
        with torch.no_grad():
            ic0 = ic_fn(ic_pts).detach().cpu().numpy()
    marcher.set_initial(ic0.reshape(n_slice, -1) if ic0.ndim > 1 else ic0)

    if ic_mode == "hard" and hard_ic_factory is not None:
        ic_u0 = torch.as_tensor(
            np.asarray(marcher.initial_values, dtype=float),
            dtype=dtype,
            device=device,
        )
        if ic_u0.ndim > 1:
            ic_u0 = ic_u0.reshape(ic_u0.shape[0], -1)[..., 0]
        field = hard_ic_factory(field, ic_pts, ic_u0)

    if optimizer is None:
        opt: Optimizer = torch.optim.Adam(field.parameters(), lr=lr)
    elif callable(optimizer) and not isinstance(optimizer, Optimizer):
        opt = optimizer(list(field.parameters()))
    else:
        opt = optimizer  # type: ignore[assignment]

    result = MarchResult(
        diagnostics={
            "ic_mode": ic_mode,
            "lr": lr,
            "advance_policy": advance_policy,
            "max_steps_per_window": max_steps_per_window,
        },
        field=field,
    )
    last_handoff: np.ndarray | None = None
    last_pred_ic: np.ndarray | None = None
    last_window_ic: np.ndarray | None = None

    while not marcher.done:
        eps = float(marcher.epsilon)
        ic_x = torch.as_tensor(
            marcher.initial_points(), dtype=dtype, device=device
        )
        ic_u = torch.as_tensor(
            np.asarray(marcher.initial_values, dtype=float),
            dtype=dtype,
            device=device,
        )
        if ic_u.ndim > 1:
            ic_u = ic_u.reshape(ic_u.shape[0], -1)[..., 0]
        window_ic = ic_u.detach().cpu().numpy().reshape(-1)

        final_loss = 0.0
        weights_np = np.ones(schedule.n_time_bins, dtype=float)
        L_np = np.zeros(schedule.n_time_bins, dtype=float)
        steps_run = 0
        converged = False
        pts_np = marcher.collocation()
        n_bins, pb, D = pts_np.shape
        coords = torch.as_tensor(
            pts_np.reshape(-1, D), dtype=dtype, device=device
        )

        while steps_run < max_steps_per_window and not converged:
            block = min(steps_per_window, max_steps_per_window - steps_run)
            for _step_i in range(block):
                if (
                    resample_every > 0
                    and steps_run > 0
                    and steps_run % resample_every == 0
                ):
                    pts_np = marcher.collocation()
                    coords = torch.as_tensor(
                        pts_np.reshape(-1, D), dtype=dtype, device=device
                    )
                opt.zero_grad()
                resid = residual_fn(field, coords)
                resid_t = _reshape_residual(resid, n_bins, pb)
                loss_c, w = causal_residual_loss(
                    resid_t, epsilon=eps, return_weights=True
                )
                assert isinstance(w, Tensor)
                loss: Tensor = loss_c
                if ic_mode == "soft" and ic_weight > 0.0:
                    pred_ic = _default_value(field, ic_x)
                    loss = loss + float(ic_weight) * torch.mean((pred_ic - ic_u) ** 2)
                loss.backward()
                opt.step()
                final_loss = float(loss.detach())
                L_np = (
                    (resid_t.detach() ** 2)
                    .mean(dim=tuple(range(1, resid_t.dim())))
                    .cpu()
                    .numpy()
                )
                weights_np = w.detach().cpu().numpy()
                steps_run += 1

            # Residual diagnostics may themselves use autograd (e.g. u_t - u_xx);
            # allow the graph, then detach the reduced statistics.
            with torch.enable_grad():
                resid = residual_fn(field, coords)
                resid_t = _reshape_residual(resid, n_bins, pb)
                L_t = (resid_t**2).mean(dim=tuple(range(1, resid_t.dim())))
                w_t = causal_weights_from_per_bin(L_t, epsilon=eps)
                L_np = L_t.detach().cpu().numpy()
                weights_np = w_t.detach().cpu().numpy()
            converged = bool(marcher.observe(weights_np))

        exhausted = not converged
        report = report_causality(L_np, weights_np)

        with torch.no_grad():
            handoff_pts = torch.as_tensor(
                marcher.handoff_points(), dtype=dtype, device=device
            )
            handoff_vals = _default_value(field, handoff_pts).detach().cpu().numpy()
            # Same-time reference: opening-slice prediction vs prescribed IC.
            pred_ic_now = (
                _default_value(field, ic_x).detach().cpu().numpy().reshape(-1)
            )
        last_handoff = np.asarray(handoff_vals, dtype=float).reshape(-1)
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
            # Refuse to propagate an unconverged handoff into the next IC.
            break
        marcher.advance(handoff_vals)
        if exhausted and stop_on_exhaust:
            break

    if (
        check_trivial
        and last_pred_ic is not None
        and last_window_ic is not None
    ):
        # Same-window opening IC as reference (not the global t0 amplitude).
        result.trivial = trivial_solution_guard(
            last_pred_ic if trivial_mode == "variance" else last_handoff,
            last_window_ic,
            ratio_threshold=trivial_ratio,
            mode=trivial_mode,
        )
    result.field = field
    return result


__all__ = [
    "MarchResult",
    "WindowResult",
    "march_solve",
]
