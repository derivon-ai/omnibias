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

from omnibias.pinn.torch.fields.attention import (
    AttentionVectorField,
    build_attention_vector_field,
)
from omnibias.pinn.torch.fields.base import FieldBase
from omnibias.pinn.torch.fields.chebyshev import ChebyshevVectorField
from omnibias.pinn.torch.fields.fbpinn import (
    FBPINNField,
    build_fbpinn_field,
    window_centers_1d,
)
from omnibias.pinn.torch.fields.jet_mlp import (
    FourierFeatureVectorField,
    JetMLPVectorField,
    build_fourier_feature_vector_field,
    build_jet_mlp_vector_field,
    make_siren_vector_field,
)
from omnibias.pinn.torch.fields.multiscale import (
    AdaptiveJetMLPVectorField,
    MscaleVectorField,
    build_adaptive_jet_mlp_vector_field,
    build_mscale_vector_field,
)
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.pinn.torch.fields.spectral import SpectralVectorField

__all__ = [
    "AdaptiveJetMLPVectorField",
    "AttentionVectorField",
    "ChebyshevVectorField",
    "FBPINNField",
    "FieldBase",
    "FourierFeatureVectorField",
    "JetMLPVectorField",
    "MscaleVectorField",
    "OneLayerVectorField",
    "SpectralVectorField",
    "build_adaptive_jet_mlp_vector_field",
    "build_attention_vector_field",
    "build_fbpinn_field",
    "build_fourier_feature_vector_field",
    "build_jet_mlp_vector_field",
    "build_mscale_vector_field",
    "make_siren_vector_field",
    "window_centers_1d",
]
