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
    CCFCompactifiedOutput,
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
from omnibias.pinn.torch.equations.ccf_compactified import (
    CordobaCordobaFontelosCompactified,
    alpha_from_lambda,
    apply_envelope,
    ccf_compactified_residual_samples,
    ccf_hardy_residual_samples,
    compactified_grid,
    compactify_y,
    compactify_y_lambda,
    compactify_y_rational,
    cordoba_cordoba_fontelos_compactified,
    decay_envelope,
    decompactify_q,
    dq_dy,
    hardy_even,
    hardy_even_deriv,
    hardy_odd,
    hardy_odd_deriv,
    hardy_profile,
    hilbert_transform_truncated_line,
    residual_weight,
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
    "CCFCompactifiedOutput",
    "CCFOutput",
    "CHOutput",
    "CahnHilliard",
    "CausalKernelFn",
    "CordobaCordobaFontelos",
    "CordobaCordobaFontelosCompactified",
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
    "alpha_from_lambda",
    "apply_envelope",
    "biharmonic",
    "burgers",
    "cahn_hilliard",
    "ccf_compactified_residual_samples",
    "ccf_hardy_residual_samples",
    "ccf_residual_samples",
    "compactified_grid",
    "compactify_y",
    "compactify_y_lambda",
    "compactify_y_rational",
    "cordoba_cordoba_fontelos",
    "cordoba_cordoba_fontelos_compactified",
    "decay_envelope",
    "decompactify_q",
    "dq_dy",
    "fredholm",
    "fredholm_residual_samples",
    "hardy_even",
    "hardy_even_deriv",
    "hardy_odd",
    "hardy_odd_deriv",
    "hardy_profile",
    "heat",
    "hilbert_transform_truncated_line",
    "kuramoto_sivashinsky",
    "navier_stokes",
    "residual_weight",
    "volterra",
    "volterra_residual_samples",
]
