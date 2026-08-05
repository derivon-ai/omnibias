# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Prebuilt PDE residuals for the jax backend.

Bit-parity twin of :mod:`omnibias.pinn.torch.equations`. Both face same
attribute DSL through :class:`FieldState`, so equation code is shared
in spirit even when the host framework differs.
"""

from __future__ import annotations

from omnibias.pinn.jax.equations._types import (
    BiharmonicOutput,
    BurgersOutput,
    CCFOutput,
    CHOutput,
    FredholmOutput,
    HeatOutput,
    KSOutput,
    NavierStokesOutput,
    VolterraOutput,
)
from omnibias.pinn.jax.equations.biharmonic import Biharmonic, biharmonic
from omnibias.pinn.jax.equations.burgers import Burgers, burgers
from omnibias.pinn.jax.equations.cahn_hilliard import (
    CahnHilliard,
    GinzburgLandauPotential,
    Potential,
    cahn_hilliard,
)
from omnibias.pinn.jax.equations.cordoba_cordoba_fontelos import (
    CordobaCordobaFontelos,
    ccf_residual_samples,
    cordoba_cordoba_fontelos,
)
from omnibias.pinn.jax.equations.heat import Heat, heat
from omnibias.pinn.jax.equations.integral import (
    CausalKernelFn,
    Fredholm,
    KernelFn,
    Volterra,
    fredholm,
    fredholm_residual_samples,
    volterra,
    volterra_residual_samples,
)
from omnibias.pinn.jax.equations.kuramoto_sivashinsky import (
    KuramotoSivashinsky,
    kuramoto_sivashinsky,
)
from omnibias.pinn.jax.equations.navier_stokes import (
    NavierStokes,
    navier_stokes,
)

__all__ = [
    "Biharmonic",
    "BiharmonicOutput",
    "Burgers",
    "BurgersOutput",
    "CCFOutput",
    "CHOutput",
    "CahnHilliard",
    "CausalKernelFn",
    "CordobaCordobaFontelos",
    "Fredholm",
    "FredholmOutput",
    "GinzburgLandauPotential",
    "Heat",
    "HeatOutput",
    "KSOutput",
    "KernelFn",
    "KuramotoSivashinsky",
    "NavierStokes",
    "NavierStokesOutput",
    "Potential",
    "Volterra",
    "VolterraOutput",
    "biharmonic",
    "burgers",
    "cahn_hilliard",
    "ccf_residual_samples",
    "cordoba_cordoba_fontelos",
    "fredholm",
    "fredholm_residual_samples",
    "heat",
    "kuramoto_sivashinsky",
    "navier_stokes",
    "volterra",
    "volterra_residual_samples",
]
