# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Activation dictionary (Keras backend): registry plus activation families.

Importing this module triggers registration of every activation in
``smooth``, ``proximal``, ``classical``, ``trigonometric``, ``nqs``,
``piecewise`` (hard non-smooth, almost-everywhere towers), and ``tempered``
(smooth beta-tempered surrogates). Use :func:`get_activation` or
:func:`list_activations` to access them.
"""

from omnibias.keras.activations import (  # noqa: F401
    classical,
    nqs,
    piecewise,
    proximal,
    smooth,
    tempered,
    trigonometric,
)
from omnibias.keras.activations.registry import (
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
