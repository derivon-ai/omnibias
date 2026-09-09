# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Deterministic JAX discovery harnesses for self-similar blow-up profiles.

Ships:

* periodic CCF (:mod:`omnibias.pinn.jax.discovery.ccf`)
* line / compactified CCF (:mod:`omnibias.pinn.jax.discovery.ccf_line`)
* signed-hat homotopy (:mod:`omnibias.pinn.jax.discovery.ccf_hat_homotopy`)
* signed PirateNet hat (:mod:`omnibias.pinn.jax.discovery.pirate_hat`)
* funnel ``lambda`` inference (:mod:`omnibias.pinn.jax.discovery.funnel`)
* Gauss-Newton trainer (:mod:`omnibias.pinn.jax.discovery.train_gn`)
* L-infinity / minimax trainer (:mod:`omnibias.pinn.jax.discovery.train_linf`)
* multi-stage correction (:mod:`omnibias.pinn.jax.discovery.multistage`)
* CAP export (:mod:`omnibias.pinn.jax.discovery.cap`)

Import explicitly::

    from omnibias.pinn.jax.discovery import ccf, ccf_line, cap, funnel, multistage, pirate_hat
"""

from __future__ import annotations

from omnibias.pinn.jax.discovery import (
    boussinesq,
    cap,
    ccf,
    ccf_hat_homotopy,
    ccf_line,
    ccf_vorticity,
    euler3d_axisym,
    funnel,
    ipm,
    lambda_laws,
    multistage,
    ns_core,
    phase5_beyond,
    pipeline,
    pirate_hat,
    polish_mp,
    spectrum,
    train_gn,
    train_linf,
)

__all__ = [
    "boussinesq",
    "cap",
    "ccf",
    "ccf_hat_homotopy",
    "ccf_line",
    "ccf_vorticity",
    "euler3d_axisym",
    "funnel",
    "ipm",
    "lambda_laws",
    "multistage",
    "ns_core",
    "phase5_beyond",
    "pipeline",
    "pirate_hat",
    "polish_mp",
    "spectrum",
    "train_gn",
    "train_linf",
]
