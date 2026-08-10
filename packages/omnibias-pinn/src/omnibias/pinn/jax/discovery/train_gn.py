# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Residual-vector Gauss-Newton with Martens-Grosse schedules (jax discovery).

Thin pytree wrapper around
:func:`omnibias.jax.optim.martens_grosse_gauss_newton_minimize`:

* damped Gauss-Newton with **QR** (non-squaring) by default
* Martens-Grosse closed-form LR / momentum via **exact** :func:`jax.jvp`
  (no finite-difference probes)

Used by DeepMind-style unstable-singularity PINNs on the Hardy-Ω residual.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from omnibias.jax.optim import (  # noqa: E402
    MartensGrosseGNConfig,
    martens_grosse_gauss_newton_minimize,
)


@dataclass(frozen=True)
class GNConfig:
    """Gauss-Newton hyper-parameters (discovery API)."""

    steps: int = 50
    gamma: float = 1e-3
    gamma_decrease: float = 0.7
    gamma_increase: float = 2.0
    min_gamma: float = 1e-8
    max_gamma: float = 1e3
    accept_tol: float = 0.0
    use_martens_grosse: bool = True
    # Non-squaring QR is the earn-path default; dense / cgls available for ablations.
    solver: str = "qr"
    seed: int = 0  # retained for API compatibility; unused (deterministic GN)


def gauss_newton_minimize(
    residual_fn: Callable[[object], Array],
    params0: object,
    *,
    config: GNConfig | None = None,
) -> tuple[object, Array]:
    """Minimise ``0.5 ||r(params)||^2`` by damped Gauss-Newton + Martens-Grosse.

    Parameters
    ----------
    residual_fn
        Maps a parameter pytree to a 1-D residual vector.
    params0
        Initial parameter pytree.
    config
        Step / damping / Martens-Grosse / solver schedule.

    Returns
    -------
    params, loss_history
    """
    cfg = GNConfig() if config is None else config
    flat0, unravel = ravel_pytree(params0)

    def r_flat(vec: Array) -> Array:
        return residual_fn(unravel(vec))

    solver = str(cfg.solver)
    if solver not in ("dense", "qr", "cgls"):
        raise ValueError(f"solver must be 'dense', 'qr', or 'cgls', got {solver!r}")

    mg_cfg = MartensGrosseGNConfig(
        steps=int(cfg.steps),
        damping=float(cfg.gamma),
        damping_decrease=float(cfg.gamma_decrease),
        damping_increase=float(cfg.gamma_increase),
        min_damping=float(cfg.min_gamma),
        max_damping=float(cfg.max_gamma),
        accept_tol=float(cfg.accept_tol),
        use_martens_grosse=bool(cfg.use_martens_grosse),
        solver=solver,  # type: ignore[arg-type]
    )
    flat1, losses = martens_grosse_gauss_newton_minimize(r_flat, flat0, config=mg_cfg)
    return unravel(flat1), jnp.asarray(losses, dtype=jnp.float64)


__all__ = ["GNConfig", "gauss_newton_minimize"]
