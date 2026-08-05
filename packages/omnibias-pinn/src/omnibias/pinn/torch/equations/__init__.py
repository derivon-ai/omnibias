# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Prebuilt PDE residuals for the torch backend.

Each equation has two faces:

1. **Class form** (preferred -- readable, configurable)::

       ns = equations.NavierStokes(viscosity=1e-3, form="primitive_3d")
       out = ns(state)            # NavierStokesOutput
       loss = (out.residual ** 2).sum(dim=-1).mean()

2. **Function shortcut** (stateless one-liner)::

       out = equations.navier_stokes(state, viscosity=1e-3, form="primitive_3d")

Both produce identical numerics. Equations consume the
:class:`FieldState` attribute DSL via ``state.ops``, so they work
identically over any field type (one-layer / spectral / chebyshev /
caged) and on either backend.
"""

from __future__ import annotations

from omnibias.pinn.torch.equations._types import (
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
from omnibias.pinn.torch.equations.biharmonic import Biharmonic, biharmonic
from omnibias.pinn.torch.equations.burgers import Burgers, burgers
from omnibias.pinn.torch.equations.cahn_hilliard import (
    CahnHilliard,
    GinzburgLandauPotential,
    Potential,
    cahn_hilliard,
)
from omnibias.pinn.torch.equations.cordoba_cordoba_fontelos import (
    CordobaCordobaFontelos,
    ccf_residual_samples,
    cordoba_cordoba_fontelos,
)
from omnibias.pinn.torch.equations.heat import Heat, heat
from omnibias.pinn.torch.equations.integral import (
    CausalKernelFn,
    Fredholm,
    KernelFn,
    Volterra,
    fredholm,
    fredholm_residual_samples,
    volterra,
    volterra_residual_samples,
)
from omnibias.pinn.torch.equations.kuramoto_sivashinsky import (
    KuramotoSivashinsky,
    kuramoto_sivashinsky,
)
from omnibias.pinn.torch.equations.navier_stokes import (
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
