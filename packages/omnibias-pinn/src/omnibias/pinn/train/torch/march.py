# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Causal time-marching training driver (torch).

This is the optimiser loop that
:class:`~omnibias.pinn._core.marching.TimeMarcher` was written for: per-window
collocation, Wang-Perdikaris causal weights fed back through
``marcher.observe``, warm-start handoff, and a
:class:`MarchResult` carrying per-window causality reports.

Honesty: the causality index is a measurement, not a proof of temporal
consistency. ``ic_mode="hard"`` requires the caller to wrap the field in a
:class:`~omnibias.pinn.torch.cage.ConstrainedExpressionField` (or similar)
before calling; this driver does not construct the TFC cage itself.
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

ResidualFn = Callable[[nn.Module, Tensor], Tensor]
ValueFn = Callable[[nn.Module, Tensor], Tensor]
ICFn = Callable[[Tensor], Tensor]


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

    @property
    def all_converged(self) -> bool:
        return bool(self.windows) and all(w.converged for w in self.windows)


def march_solve(
    field: nn.Module,
    residual_fn: ResidualFn,
    coordinate_spec: CoordinateSpec,
    schedule: TimeWindowSchedule,
    *,
    steps_per_window: int = 50,
    lr: float = 1e-3,
    per_bin: int = 16,
    n_slice: int = 32,
    ic_values: np.ndarray | Tensor | None = None,
    ic_fn: ICFn | None = None,
    ic_weight: float = 1.0,
    ic_mode: str = "soft",
    value_fn: ValueFn | None = None,
    seed: int = 0,
    dtype: torch.dtype | None = None,
    check_trivial: bool = True,
    trivial_ratio: float = 1e-3,
) -> MarchResult:
    """March a PINN solve window-by-window with causal residual weighting.

    Parameters
    ----------
    field
        Trainable field (``nn.Module`` exposing ``parameters()``).
    residual_fn
        ``residual_fn(field, coords) -> residual`` where ``coords`` is
        ``(n_bins * per_bin, D)`` flattened from the marcher's collocation
        layout. The residual is reshaped to ``(n_bins, per_bin)`` (or
        ``(n_bins, per_bin, ...)``) before
        :func:`~omnibias.pinn.torch.losses.causal_residual_loss`.
    coordinate_spec
        Must declare a time axis and explicit domain bounds.
    schedule
        Window ladder (geometry + annealed epsilon + advance tolerance).
    steps_per_window
        Adam steps per window before the advance check.
    lr
        Adam learning rate.
    per_bin, n_slice
        Forwarded to :class:`~omnibias.pinn._core.marching.TimeMarcher`.
    ic_values
        Optional ``(n_slice, ...)`` array of initial-condition values at
        ``marcher.initial_points()``. Used for the first window; later windows
        inherit the warm-start handoff.
    ic_fn
        Alternative to ``ic_values``: ``ic_fn(coords) -> values`` evaluated
        once at the first window's opening slice.
    ic_weight
        Soft IC penalty weight (ignored when ``ic_mode="hard"``).
    ic_mode
        ``"soft"`` (default) adds an MSE IC term; ``"hard"`` assumes the field
        already embeds the IC structurally and skips the penalty.
    value_fn
        ``value_fn(field, coords) -> values`` used for the warm-start handoff
        and the trivial-solution guard. Defaults to calling
        ``field(coords)`` / ``field.forward(coords)`` and taking the leading
        component.
    seed
        Base seed for the marcher.
    dtype
        Collocation / IC tensor dtype; defaults to the first parameter's dtype
        or ``float64``.
    check_trivial
        Run :func:`~omnibias.pinn.train._core.guards.trivial_solution_guard`
        on the final handoff against the first-window IC.
    trivial_ratio
        Threshold for the trivial-solution guard.
    """
    if ic_mode not in ("soft", "hard"):
        raise ValueError(f"ic_mode must be 'soft' or 'hard', got {ic_mode!r}")
    if steps_per_window < 1:
        raise ValueError(f"steps_per_window must be >= 1, got {steps_per_window}")

    if dtype is None:
        try:
            dtype = next(field.parameters()).dtype
        except StopIteration:
            dtype = torch.float64

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

    # Seed the first window IC.
    ic_pts_np = marcher.initial_points()
    ic_pts = torch.as_tensor(ic_pts_np, dtype=dtype)
    if ic_values is not None:
        ic0 = np.asarray(
            ic_values.detach().cpu().numpy()
            if isinstance(ic_values, Tensor)
            else ic_values,
            dtype=float,
        )
    elif ic_fn is not None:
        with torch.no_grad():
            ic0 = ic_fn(ic_pts).detach().cpu().numpy()
    else:
        ic0 = np.zeros((n_slice,), dtype=float)
    marcher.set_initial(ic0.reshape(n_slice, -1) if ic0.ndim > 1 else ic0)

    opt = torch.optim.Adam(field.parameters(), lr=lr)
    result = MarchResult(diagnostics={"ic_mode": ic_mode, "lr": lr})
    first_ic = np.asarray(marcher.initial_values, dtype=float).reshape(-1)
    last_handoff: np.ndarray | None = None

    while not marcher.done:
        pts_np = marcher.collocation()  # (n_bins, per_bin, D)
        n_bins, pb, D = pts_np.shape
        coords = torch.as_tensor(pts_np.reshape(-1, D), dtype=dtype)
        eps = float(marcher.epsilon)
        ic_x = torch.as_tensor(marcher.initial_points(), dtype=dtype)
        ic_u = torch.as_tensor(
            np.asarray(marcher.initial_values, dtype=float), dtype=dtype
        )
        if ic_u.ndim > 1:
            ic_u = ic_u.reshape(ic_u.shape[0], -1)[..., 0]

        final_loss = 0.0
        weights_np = np.ones(n_bins, dtype=float)
        L_np = np.zeros(n_bins, dtype=float)

        for _step in range(steps_per_window):
            opt.zero_grad()
            resid = residual_fn(field, coords)
            if resid.shape[0] != n_bins * pb:
                # Allow (n_bins, per_bin, ...) already.
                try:
                    resid_t = resid.reshape(n_bins, pb, *resid.shape[1:])
                except RuntimeError as exc:
                    raise ValueError(
                        f"residual leading dim {resid.shape[0]} incompatible "
                        f"with n_bins*per_bin={n_bins * pb}"
                    ) from exc
            else:
                resid_t = resid.reshape(n_bins, pb, *resid.shape[1:])
            # Reduce extra trailing dims so causal_residual_loss sees (n_t, spatial).
            while resid_t.dim() > 2:
                resid_t = resid_t.mean(dim=-1)
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

        # Refresh weights once more for the report / advance criterion.
        with torch.no_grad():
            resid = residual_fn(field, coords)
            resid_t = resid.reshape(n_bins, pb, *resid.shape[1:])
            while resid_t.dim() > 2:
                resid_t = resid_t.mean(dim=-1)
            L_t = (resid_t**2).mean(dim=tuple(range(1, resid_t.dim())))
            w_t = causal_weights_from_per_bin(L_t, epsilon=eps)
            L_np = L_t.cpu().numpy()
            weights_np = w_t.cpu().numpy()

        converged = bool(marcher.observe(weights_np))
        report = report_causality(L_np, weights_np)

        with torch.no_grad():
            handoff_pts = torch.as_tensor(marcher.handoff_points(), dtype=dtype)
            handoff_vals = _default_value(field, handoff_pts).detach().cpu().numpy()
        last_handoff = np.asarray(handoff_vals, dtype=float).reshape(-1)

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

    if check_trivial and last_handoff is not None and first_ic.size > 0:
        # Compare final handoff energy against the original IC.
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
