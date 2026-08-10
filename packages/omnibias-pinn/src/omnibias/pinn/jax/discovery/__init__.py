# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Deterministic JAX discovery harnesses for self-similar blow-up profiles.

Ships:

* periodic CCF (:mod:`omnibias.pinn.jax.discovery.ccf`)
* line / compactified CCF (:mod:`omnibias.pinn.jax.discovery.ccf_line`)
* funnel ``lambda`` inference (:mod:`omnibias.pinn.jax.discovery.funnel`)
* Gauss-Newton trainer (:mod:`omnibias.pinn.jax.discovery.train_gn`)
* multi-stage correction (:mod:`omnibias.pinn.jax.discovery.multistage`)
* CAP export (:mod:`omnibias.pinn.jax.discovery.cap`)

Import explicitly::

    from omnibias.pinn.jax.discovery import ccf, ccf_line, cap, funnel, multistage
"""

from __future__ import annotations

from omnibias.pinn.jax.discovery import (
    boussinesq,
    cap,
    ccf,
    ccf_line,
    ccf_vorticity,
    euler3d_axisym,
    funnel,
    ipm,
    lambda_laws,
    multistage,
    phase5_beyond,
    pipeline,
    polish_mp,
    spectrum,
    train_gn,
)

__all__ = [
    "boussinesq",
    "cap",
    "ccf",
    "ccf_line",
    "ccf_vorticity",
    "euler3d_axisym",
    "funnel",
    "ipm",
    "lambda_laws",
    "multistage",
    "phase5_beyond",
    "pipeline",
    "polish_mp",
    "spectrum",
    "train_gn",
]
