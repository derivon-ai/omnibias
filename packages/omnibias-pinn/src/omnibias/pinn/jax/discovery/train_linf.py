# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Residual-vector L-infinity (minimax) trainer for jax discovery.

Thin pytree wrapper around
:func:`omnibias.jax.optim.linf_minimax_minimize` -- a safeguarded
linearized-epigraph / Lawson-IRLS minimax step that directly minimises
``max_i |r_i(params)|`` rather than ``0.5 ||r(params)||^2``. This is the
**L-infinity sibling** of :mod:`omnibias.pinn.jax.discovery.train_gn`: same
``residual_fn: pytree -> Array`` convention, same ``(params, history)``
return shape, so the two trainers are interchangeable in any caller that
compares GN and minimax on the same residual.

This module makes no claim about, and does not run, any specific model,
benchmark, or certified result -- it is a generic comparison utility, kept
beside its Gauss-Newton sibling for discoverability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)
from omnibias.jax.optim import (  # noqa: E402
    LinfMinimaxConfig,
    linf_minimax_minimize,
)


@dataclass(frozen=True)
class LinfConfig:
    """L-infinity (minimax) hyper-parameters (discovery API).

    Mirrors :class:`omnibias.pinn.jax.discovery.train_gn.GNConfig`'s field
    names where the concepts line up (``steps``, trust-region schedule,
    ``accept_tol``); ``box0`` replaces GN's ``gamma`` since the epigraph /
    Lawson-IRLS direction is a full linearized minimax solve, not a damped
    normal-equation solve, so the safeguard parameter is a step-size trust
    region rather than a Levenberg-Marquardt damping.
    """

    steps: int = 50
    box0: float = 1.0
    box_decrease: float = 0.5
    box_increase: float = 1.5
    min_box: float = 1e-10
    max_box: float = 1e6
    irls_iters: int = 16
    max_line_search: int = 20
    accept_tol: float = 0.0


def linf_minimax_train(
    residual_fn: Callable[[object], Array],
    params0: object,
    *,
    config: LinfConfig | None = None,
) -> tuple[object, Array]:
    """Minimise ``max_i |r_i(params)|`` by safeguarded linearized-epigraph steps.

    Parameters
    ----------
    residual_fn
        Maps a parameter pytree to a 1-D residual vector.
    params0
        Initial parameter pytree.
    config
        Trust-region / IRLS / line-search schedule.

    Returns
    -------
    params, max_abs_history
        ``max_abs_history[k]`` is ``max|r|`` *before* step ``k``; the final
        entry is ``max|r|`` at the returned ``params`` (matching the
        before/after history convention of
        :func:`omnibias.pinn.jax.discovery.train_gn.gauss_newton_minimize`).
    """
    cfg = LinfConfig() if config is None else config
    flat0, unravel = ravel_pytree(params0)

    def r_flat(vec: Array) -> Array:
        return residual_fn(unravel(vec))

    linf_cfg = LinfMinimaxConfig(
        steps=int(cfg.steps),
        box0=float(cfg.box0),
        box_decrease=float(cfg.box_decrease),
        box_increase=float(cfg.box_increase),
        min_box=float(cfg.min_box),
        max_box=float(cfg.max_box),
        irls_iters=int(cfg.irls_iters),
        max_line_search=int(cfg.max_line_search),
        accept_tol=float(cfg.accept_tol),
    )
    flat1, history = linf_minimax_minimize(r_flat, flat0, config=linf_cfg)
    return unravel(flat1), jnp.asarray(history, dtype=jnp.float64)


__all__ = ["LinfConfig", "linf_minimax_train"]
