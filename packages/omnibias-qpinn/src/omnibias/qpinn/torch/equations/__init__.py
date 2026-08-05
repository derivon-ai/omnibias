# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Prebuilt PDE residuals for the torch backend (quantum-physics PDEs).

Every equation has two faces:

1. **Class form** (preferred -- readable, configurable)::

       eq = equations.TDSE(hbar=1.0, mass=1.0, potential=harmonic_V)
       out = eq(state)                              # TDSEOutput
       loss = (out.residual ** 2).sum(dim=-1).mean()

2. **Function shortcut** (stateless one-liner)::

       out = equations.tdse(state, hbar=1.0, mass=1.0, potential=harmonic_V)

Both produce identical numerics. Equations consume the
:class:`FieldState` attribute DSL via ``state.ops``, so they work
identically over any ``omnibias.pinn.torch.fields.*`` field type
(one-layer / spectral / chebyshev / caged) and have JAX twins under
:mod:`omnibias.qpinn.jax.equations` with bit-identical numerics.
"""

from __future__ import annotations

from omnibias.qpinn.torch.equations._types import (
    NLSOutput,
    RotatingNLSOutput,
    TDSEOutput,
    TISEOutput,
)
from omnibias.qpinn.torch.equations.dirac import Dirac, DiracOutput, dirac
from omnibias.qpinn.torch.equations.helmholtz import (
    Helmholtz,
    HelmholtzOutput,
    helmholtz,
)
from omnibias.qpinn.torch.equations.klein_gordon import (
    KleinGordon,
    KleinGordonOutput,
    klein_gordon,
)
from omnibias.qpinn.torch.equations.nls import NLS, nls
from omnibias.qpinn.torch.equations.rotating_nls import RotatingNLS, rotating_nls
from omnibias.qpinn.torch.equations.tdse import TDSE, tdse
from omnibias.qpinn.torch.equations.tise import TISE, tise

__all__ = [
    "Dirac",
    "DiracOutput",
    "Helmholtz",
    "HelmholtzOutput",
    "KleinGordon",
    "KleinGordonOutput",
    "NLS",
    "NLSOutput",
    "RotatingNLS",
    "RotatingNLSOutput",
    "TDSE",
    "TDSEOutput",
    "TISE",
    "TISEOutput",
    "dirac",
    "helmholtz",
    "klein_gordon",
    "nls",
    "rotating_nls",
    "tdse",
    "tise",
]
