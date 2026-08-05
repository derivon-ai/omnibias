# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Reference JAX architectures built on the omnibias closed-form jets.

* :class:`~omnibias.jax.architectures.pinn.JetMLP` -- deep, arbitrary-order
  physics-informed network whose input derivatives come from the multivariate-jet
  kernel (bit-identical twin of the torch ``JetMLP``).
* :class:`~omnibias.jax.architectures.pinn.FourierFeatureMLP` -- sin-encoded random
  Fourier-feature front end for spectral-bias mitigation, still fully closed-form.
* :func:`~omnibias.jax.architectures.pinn.make_siren` -- SIREN built as a ``sin``
  :class:`JetMLP` with exact arbitrary-order derivatives.
* :class:`~omnibias.jax.architectures.hardbc.HardConstraintField` -- wraps a network
  as ``u = g + b N`` so a boundary/initial condition holds exactly by construction
  (no boundary loss term), staying closed form via the jet-level Leibniz product.
"""

from omnibias.jax.architectures.hardbc import (
    AffineFactor,
    AffineLift,
    BoundaryMask,
    HardConstraintField,
    dirichlet_interval,
    homogeneous_box,
    initial_value,
)
from omnibias.jax.architectures.pinn import (
    FourierFeatureMLP,
    JetMLP,
    make_fourier_feature_mlp,
    make_jet_mlp,
    make_siren,
)

__all__ = [
    "AffineFactor",
    "AffineLift",
    "BoundaryMask",
    "FourierFeatureMLP",
    "HardConstraintField",
    "JetMLP",
    "dirichlet_interval",
    "homogeneous_box",
    "initial_value",
    "make_fourier_feature_mlp",
    "make_jet_mlp",
    "make_siren",
]
