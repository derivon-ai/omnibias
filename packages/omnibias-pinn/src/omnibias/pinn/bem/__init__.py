# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""BEM-Net (theory 02-06, gated).

Linear constant-coefficient homogeneous only. PDE exact off-surface;
the boundary condition is approximated. No Yang-Mills / NS claim.
"""

from __future__ import annotations

from omnibias.pinn.bem._core import (
    KernelSpec,
    Surface,
    green_laplace_2d,
    half_plane_dtn,
    pde_residual_off_surface,
    poisson_pair_dictionary,
    single_layer,
)

__all__ = [
    "BEMNet",
    "KernelSpec",
    "Surface",
    "green_laplace_2d",
    "half_plane_dtn",
    "pde_residual_off_surface",
    "poisson_pair_dictionary",
    "single_layer",
]


def __getattr__(name: str) -> object:
    if name == "BEMNet":
        from omnibias.pinn.bem.torch import BEMNet

        return BEMNet
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
