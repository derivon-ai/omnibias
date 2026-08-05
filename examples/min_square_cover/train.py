# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The shared solve loop: anneal ``beta`` (and the count penalty ``lambda``) while an arm
descends the soft-coverage energy, then round to a feasible discrete cover.

Architecture, initialisation (greedy warm start), candidate count ``K``, and anneal schedule
are fixed by ``seed`` and the arguments; the only per-arm degree of freedom is the optimiser.
This is the covering analogue of ``examples/binary_vs_ste/train.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from examples.min_square_cover.arms import Arm
from examples.min_square_cover.coverage import (
    closed_form_newton_step,
    finalize_cover,
    residual_vector,
    scalar_energy,
)
from examples.min_square_cover.data import Instance, greedy_cover
from examples.min_square_cover.shapes import grid_axes, init_centers, lp_init_centers


@dataclass
class CoverResult:
    """Outcome of one ``(arm, instance, seed)`` solve."""

    arm: str
    shape: str
    shape_kind: str
    warm_start: str
    beta_schedule: str
    seed: int
    side: int
    n_ones: int
    n_greedy: int
    n_active: int
    n_final: int
    n_completion: int
    feasible_before_completion: bool
    lower_bound: int
    ratio_final: float
    ratio_greedy: float
    init_energy: float
    final_energy: float
    steps: int
    squares: list[tuple[int, int]] = field(default_factory=list)
    history: list[dict[str, float]] = field(default_factory=list)


def _scalar_closure(
    axes: Sequence[Tensor],
    centers: Tensor,
    gate_logits: Tensor,
    image: Tensor,
    side: float,
    beta: float,
    *,
    loss: str,
    kappa: float,
    lam: float,
    shape_kind: str,
) -> Callable[[], Tensor]:
    def closure() -> Tensor:
        return scalar_energy(
            axes, centers, gate_logits, image, side, beta,
            loss=loss, kappa=kappa, lam=lam, shape_kind=shape_kind,
        )

    return closure


def _residual_closure(
    axes: Sequence[Tensor],
    centers: Tensor,
    gate_logits: Tensor,
    image: Tensor,
    side: float,
    beta: float,
    *,
    lam: float,
    shape_kind: str,
) -> Callable[[], Tensor]:
    def closure() -> Tensor:
        return residual_vector(
            axes, centers, gate_logits, image, side, beta, lam=lam, shape_kind=shape_kind
        )

    return closure


def _functional_residual_fn(
    axes: Sequence[Tensor],
    image: Tensor,
    side: float,
    beta: float,
    n_centers: int,
    center_shape: tuple[int, int],
    *,
    lam: float,
    shape_kind: str,
) -> Callable[[Tensor], Tensor]:
    """Residual over a flat ``[centers, gate_logits]`` vector, for the functional GN arm."""

    def residual_fn(flat: Tensor) -> Tensor:
        centers = flat[:n_centers].reshape(center_shape)
        gate_logits = flat[n_centers:]
        return residual_vector(
            axes, centers, gate_logits, image, side, beta, lam=lam, shape_kind=shape_kind
        )

    return residual_fn


def _anneal_exp(v0: float, v1: float, frac: float) -> float:
    return float(v0 * (v1 / v0) ** frac)


def solve_cover(
    arm: Arm,
    instance: Instance,
    *,
    steps: int = 150,
    beta0: float = 1.0,
    beta1: float = 8.0,
    fixed_beta: float | None = None,
    kappa: float = 4.0,
    lam0: float = 0.0,
    lam1: float = 0.15,
    k_slack: int = 2,
    init_gate: float = 2.0,
    gate_threshold: float = 0.5,
    shape_kind: str = "square",
    warm_start: str = "greedy",
    seed: int = 0,
    device: str = "cpu",
    log: bool = False,
) -> CoverResult:
    """Solve one covering instance with ``arm`` and return its :class:`CoverResult`.

    The soft cover starts from a warm start (``K = n_greedy + k_slack`` candidate squares) and is
    optimised while ``beta`` hardens ``beta0 -> beta1`` and the count penalty grows
    ``lam0 -> lam1``; the active squares are then rounded and greedily completed to a
    guaranteed-feasible discrete cover. ``shape_kind`` selects the soft occupancy surrogate
    (``"square"`` or ``"disk"``); the ``closed_form`` arm requires ``"square"``. ``warm_start``
    is ``"greedy"`` (greedy corners, gates on) or ``"lp"`` (the LP-relaxation register: the top
    fractional-weight positions become centers and gates start at the LP soft selection; falls
    back to greedy if ``omnibias-convex`` is unavailable).

    ``fixed_beta`` controls the sharpness homotopy (hypothesis H3): the default ``None`` anneals
    ``beta0 -> beta1`` over the run, while a float pins ``beta`` to that constant for every step
    (the fixed-sharpness ablation). The choice is recorded in ``CoverResult.beta_schedule`` as
    ``"anneal"`` or ``"fixed@<beta>"``.
    """
    from examples.min_square_cover.certify import area_lower_bound, lp_fractional_cover

    if arm.kind == "closed_form" and shape_kind != "square":
        raise ValueError("the closed_form arm uses the box closed-form Hessian; shape_kind must be 'square'")
    if warm_start not in ("greedy", "lp"):
        raise ValueError(f"warm_start must be 'greedy' or 'lp', got {warm_start!r}")
    torch.manual_seed(seed)
    image = instance.image.to(device)
    side_i = instance.side
    side_f = float(side_i)
    rows, cols = grid_axes(instance.shape)
    axes = (rows.to(device), cols.to(device))

    greedy = greedy_cover(instance.image, side_i)
    n_greedy = len(greedy)
    k = max(1, n_greedy + k_slack)
    lp_cover = lp_fractional_cover(instance.image, side_i) if warm_start == "lp" else None
    if lp_cover is not None:
        used_warm = "lp"
        c0, g0 = lp_init_centers(
            instance, lp_cover.positions, lp_cover.weights, k, init_gate=init_gate, seed=seed
        )
    else:
        used_warm = "greedy"  # requested greedy, or LP unavailable -> honest fallback
        c0 = init_centers(instance, greedy, k, seed=seed)
        g0 = torch.full((k,), float(init_gate))
    centers = c0.to(device).requires_grad_(True)
    gate_logits = g0.to(device).requires_grad_(True)
    optimizer = arm.make_optimizer([centers, gate_logits])

    beta_schedule = "anneal" if fixed_beta is None else f"fixed@{fixed_beta:g}"
    denom = max(steps - 1, 1)
    history: list[dict[str, float]] = []
    init_energy = float("nan")
    for step in range(steps):
        frac = step / denom
        beta = _anneal_exp(beta0, beta1, frac) if fixed_beta is None else fixed_beta
        lam = lam0 + frac * (lam1 - lam0)
        if arm.kind == "first_order":
            optimizer.zero_grad()
            energy = scalar_energy(
                axes, centers, gate_logits, image, side_f, beta,
                loss=arm.loss, kappa=kappa, lam=lam, shape_kind=shape_kind,
            )
            energy.backward()
            optimizer.step()
            e_val = float(energy.detach())
        elif arm.kind == "scalar":
            closure = _scalar_closure(
                axes, centers, gate_logits, image, side_f, beta,
                loss=arm.loss, kappa=kappa, lam=lam, shape_kind=shape_kind,
            )
            e_val = float(optimizer.step(closure))
        elif arm.kind == "functional_gn":
            n_c = centers.numel()
            residual_fn = _functional_residual_fn(
                axes, image, side_f, beta, n_c, (k, 2), lam=lam, shape_kind=shape_kind
            )
            flat0 = torch.cat([centers.detach().reshape(-1), gate_logits.detach()])
            new_flat, _info = optimizer.step(residual_fn, flat0)  # type: ignore[union-attr]
            with torch.no_grad():
                centers.copy_(new_flat[:n_c].reshape(k, 2))
                gate_logits.copy_(new_flat[n_c:])
                e_val = float(
                    scalar_energy(
                        axes, centers, gate_logits, image, side_f, beta,
                        loss="sq_hinge", kappa=kappa, lam=lam, shape_kind=shape_kind,
                    )
                )
        elif arm.kind == "closed_form":
            e_val = closed_form_newton_step(
                axes, centers, gate_logits, image, side_f, beta,
                loss=arm.loss, kappa=kappa, lam=lam,
            )
        else:
            closure = _residual_closure(
                axes, centers, gate_logits, image, side_f, beta, lam=lam, shape_kind=shape_kind
            )
            optimizer.step(closure)
            with torch.no_grad():
                e_val = float(
                    scalar_energy(
                        axes, centers, gate_logits, image, side_f, beta,
                        loss="sq_hinge", kappa=kappa, lam=lam, shape_kind=shape_kind,
                    )
                )
        if step == 0:
            init_energy = e_val
        history.append({"step": float(step), "beta": beta, "lam": lam, "energy": e_val})
        if log:
            print(f"[{arm.name:>18s} {instance.name:>10s} seed={seed}] "
                  f"step {step + 1:>3d}/{steps} beta={beta:5.2f} lam={lam:5.3f} E={e_val:.4f}")

    disc = finalize_cover(centers, gate_logits, instance, threshold=gate_threshold)
    lower_bound = area_lower_bound(image, side_i)
    return CoverResult(
        arm=arm.name,
        shape=instance.name,
        shape_kind=shape_kind,
        warm_start=used_warm,
        beta_schedule=beta_schedule,
        seed=seed,
        side=side_i,
        n_ones=instance.n_ones,
        n_greedy=n_greedy,
        n_active=disc.n_active,
        n_final=disc.n_final,
        n_completion=disc.n_completion,
        feasible_before_completion=disc.feasible_before_completion,
        lower_bound=lower_bound,
        ratio_final=disc.n_final / max(lower_bound, 1),
        ratio_greedy=n_greedy / max(lower_bound, 1),
        init_energy=init_energy,
        final_energy=history[-1]["energy"] if history else float("nan"),
        steps=steps,
        squares=disc.squares,
        history=history,
    )


__all__ = ["CoverResult", "solve_cover"]
