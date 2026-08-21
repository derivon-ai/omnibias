# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Optional bounded-step residual fit that packs an :class:`Observation`.

Default CI does not import this module. Symbolic never imports it at
module load. The fit is a proposer; snap stays the accept gate.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from omnibias.core.proof.observe import Observation
from omnibias.symbolic.ingest import pack_residual


def fit_residual_and_pack(
    observation: Observation,
    *,
    steps: int = 2,
) -> Observation:
    """Two-step Gauss-Newton on a 1-D identity residual, then pack values."""

    import jax.numpy as jnp
    from omnibias.pinn.jax.discovery.train_gn import GNConfig, gauss_newton_minimize

    if "y" not in observation.jet_names or "yp" not in observation.jet_names:
        return observation
    y_index = observation.jet_names.index("y")
    yp_index = observation.jet_names.index("yp")
    y = jnp.asarray(
        [float(Fraction(row[y_index])) for row in observation.jet_rows],
        dtype=jnp.float64,
    )
    yp = jnp.asarray(
        [float(Fraction(row[yp_index])) for row in observation.jet_rows],
        dtype=jnp.float64,
    )

    def residual_fn(params: Any) -> Any:
        scale = params["scale"]
        return yp - scale * (1.0 - y * y)

    params0 = {"scale": jnp.asarray(1.0, dtype=jnp.float64)}
    fitted, _history = gauss_newton_minimize(
        residual_fn,
        params0,
        config=GNConfig(steps=int(steps), gamma=1e-3),
    )
    residual = residual_fn(fitted)
    values = [float(item) for item in residual.tolist()]
    return pack_residual(values, observation=observation)
