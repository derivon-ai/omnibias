# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Residual-vector Gauss-Newton for jax discovery.

Thin pytree wrapper around the JAX second-order drivers:

* :func:`omnibias.jax.optim.martens_grosse_gauss_newton_minimize` -- damped
  Gauss-Newton (QR by default) plus Martens–Grosse closed-form LR / momentum
  via exact :func:`jax.jvp`
* :func:`omnibias.jax.optim.cubic_regularized_gauss_newton_minimize` -- ARC on
  the PSD Gauss-Newton model (``method="cubic"``)

Used by DeepMind-style unstable-singularity PINNs on the Hardy-Ω residual.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)
from omnibias.jax.optim import (  # noqa: E402
    CubicRegularizedGNConfig,
    MartensGrosseGNConfig,
    cubic_regularized_gauss_newton_minimize,
    martens_grosse_gauss_newton_minimize,
)

GNMethod = Literal["martens_grosse", "cubic"]


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
    method: GNMethod = "martens_grosse"
    cubic_sigma: float = 1.0
    # None = full parameter dimension (do not truncate ARC on wide hops).
    krylov_dim: int | None = None


def gauss_newton_minimize(
    residual_fn: Callable[[object], Array],
    params0: object,
    *,
    config: GNConfig | None = None,
) -> tuple[object, Array]:
    """Minimise ``0.5 ||r(params)||^2`` by damped or cubic-regularised Gauss-Newton.

    Parameters
    ----------
    residual_fn
        Maps a parameter pytree to a 1-D residual vector.
    params0
        Initial parameter pytree.
    config
        Step / damping / Martens-Grosse / cubic / solver schedule.

    Returns
    -------
    params, loss_history
    """
    cfg = GNConfig() if config is None else config
    flat0, unravel = ravel_pytree(params0)

    def r_flat(vec: Array) -> Array:
        return residual_fn(unravel(vec))

    method = str(cfg.method)
    if method == "cubic":
        cubic_cfg = CubicRegularizedGNConfig(
            steps=int(cfg.steps),
            sigma=float(cfg.cubic_sigma),
            krylov_dim=cfg.krylov_dim,
        )
        flat1, losses = cubic_regularized_gauss_newton_minimize(
            r_flat, flat0, config=cubic_cfg
        )
        return unravel(flat1), jnp.asarray(losses, dtype=jnp.float64)
    if method != "martens_grosse":
        raise ValueError(
            f"method must be 'martens_grosse' or 'cubic', got {method!r}"
        )

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


__all__ = ["GNConfig", "GNMethod", "gauss_newton_minimize"]
