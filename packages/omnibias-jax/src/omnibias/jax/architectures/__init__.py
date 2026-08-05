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
* :class:`~omnibias.jax.architectures.multiscale.AdaptiveJetMLP` /
  :class:`~omnibias.jax.architectures.multiscale.MscaleMLP` -- the two constructions
  that put the frequency knob *inside* the network: a trainable activation slope
  ``sigma(n a z)`` (built from the ``tempered`` combinator, so the tower stays
  exact) and the MscaleDNN band mixture ``u(x) = sum_j f_j(alpha_j x)``.
* :class:`~omnibias.jax.architectures.hardbc.HardConstraintField` -- wraps a network
  as ``u = g + b N`` so a boundary/initial condition holds exactly by construction
  (no boundary loss term), staying closed form via the jet-level Leibniz product.
* :class:`~omnibias.jax.architectures.attention.AttentionJetMLP` -- the first
  *non-local* block on the substrate: a softmax mixture over a trainable memory,
  whose coordinate derivatives stay closed form through ``jet_attention``.
"""

from omnibias.jax.architectures.attention import (
    AttentionJetMLP,
    make_attention_jet_mlp,
)
from omnibias.jax.architectures.hardbc import (
    AffineFactor,
    AffineLift,
    BoundaryMask,
    HardConstraintField,
    dirichlet_interval,
    homogeneous_box,
    initial_value,
)
from omnibias.jax.architectures.multiscale import (
    AdaptiveActivation,
    AdaptiveJetMLP,
    MscaleMLP,
    make_adaptive_activation,
    make_adaptive_jet_mlp,
    make_mscale_mlp,
)
from omnibias.jax.architectures.pinn import (
    FourierFeatureMLP,
    JetMLP,
    make_fourier_feature_mlp,
    make_jet_mlp,
    make_siren,
)

__all__ = [
    "AdaptiveActivation",
    "AdaptiveJetMLP",
    "AffineFactor",
    "AffineLift",
    "AttentionJetMLP",
    "BoundaryMask",
    "FourierFeatureMLP",
    "HardConstraintField",
    "JetMLP",
    "MscaleMLP",
    "dirichlet_interval",
    "homogeneous_box",
    "initial_value",
    "make_adaptive_activation",
    "make_adaptive_jet_mlp",
    "make_attention_jet_mlp",
    "make_fourier_feature_mlp",
    "make_jet_mlp",
    "make_mscale_mlp",
    "make_siren",
]
