# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Activation dictionary: registry plus the activation families.

Importing this module triggers registration of every activation in
``smooth``, ``proximal``, ``classical``, ``nqs``, ``trigonometric``,
``piecewise`` (hard non-smooth, almost-everywhere towers), and ``tempered``
(smooth beta-tempered surrogates). Use :func:`get_activation` or
:func:`list_activations` to access them by name.
"""

# Importing for side-effect: each module registers its activations. The
# ``tempered`` surrogates import the base specs they temper (softplus, sigmoid,
# tanh) directly, so the registration dependency is resolved by Python's import
# machinery regardless of the listing order here.
from omnibias.torch.activations import (  # noqa: F401
    classical,
    nqs,
    piecewise,
    proximal,
    smooth,
    tempered,
    trigonometric,
)
from omnibias.torch.activations.registry import (
    ActivationSpec,
    get_activation,
    is_registered,
    list_activations,
    register_activation,
)

__all__ = [
    "ActivationSpec",
    "get_activation",
    "is_registered",
    "list_activations",
    "register_activation",
]
