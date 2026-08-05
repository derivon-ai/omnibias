# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Prebuilt PDE residuals for the jax backend (quantum-physics PDEs).

Every equation here has bit-identical numerics with its
:mod:`omnibias.qpinn.torch.equations` twin (cross-backend parity is
checked in ``tests/cross_backend/``).
"""

from __future__ import annotations

from omnibias.qpinn.jax.equations._types import (
    NLSOutput,
    RotatingNLSOutput,
    TDSEOutput,
    TISEOutput,
)
from omnibias.qpinn.jax.equations.dirac import Dirac, DiracOutput, dirac
from omnibias.qpinn.jax.equations.helmholtz import (
    Helmholtz,
    HelmholtzOutput,
    helmholtz,
)
from omnibias.qpinn.jax.equations.klein_gordon import (
    KleinGordon,
    KleinGordonOutput,
    klein_gordon,
)
from omnibias.qpinn.jax.equations.nls import NLS, nls
from omnibias.qpinn.jax.equations.rotating_nls import RotatingNLS, rotating_nls
from omnibias.qpinn.jax.equations.tdse import TDSE, tdse
from omnibias.qpinn.jax.equations.tise import TISE, tise

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
