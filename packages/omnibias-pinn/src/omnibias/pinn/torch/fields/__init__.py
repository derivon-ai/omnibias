# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Typed PINN fields for the torch backend.

Each field is a ``torch.nn.Module`` whose ``__call__(coords)`` returns a
:class:`omnibias.pinn.FieldState`. The state primes the lazy
:class:`SigmaCache`, exposes the :class:`ComponentSpec` /
:class:`CoordinateSpec` metadata, and routes attribute access into the
torch ops dispatch module.
"""

from __future__ import annotations

from omnibias.pinn.torch.fields.base import FieldBase
from omnibias.pinn.torch.fields.chebyshev import ChebyshevVectorField
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.pinn.torch.fields.spectral import SpectralVectorField

__all__ = [
    "ChebyshevVectorField",
    "FieldBase",
    "OneLayerVectorField",
    "SpectralVectorField",
]
