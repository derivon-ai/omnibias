# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Physics-informed neural networks built on omnibias closed-form derivatives.

``omnibias-pinn`` ships typed PINN fields, closed-form differential
operators, hard-conservation cage layers, conditioned losses, prebuilt
PDE residuals, and diagnostics for the Riccati class of base activations
(``tanh``, ``sigmoid``, ``softplus``, ``gaussian``, ``exp``).

Public API split:

- :mod:`omnibias.pinn` (this module) -- backend-agnostic schemas and
  small re-export of the most-used types. The top-level subpackages
  :mod:`omnibias.pinn.torch` and :mod:`omnibias.pinn.jax` carry the
  backend implementations (fields, ops, cage, losses, equations,
  diagnostics) with bit-identical numerics.
- :mod:`omnibias.pinn._core` -- the schemas (``CoordinateSpec``,
  ``ComponentSpec``, ``FieldState``, ...) that drive both backends.
- :mod:`omnibias.pinn.solver` -- an **alpha** mesh-free solver for coupled
  systems of PDEs (``System`` / ``Domain`` / canonical problem builders plus
  torch/jax collocation and spectral method-of-lines drivers), folded in from
  the former standalone ``omnibias-pde`` package. Imported on demand; not part
  of the eager ``omnibias.pinn`` API.

Backend selection
-----------------

Importing ``omnibias.pinn.torch`` requires ``omnibias-pinn[torch]``;
importing ``omnibias.pinn.jax`` requires ``omnibias-pinn[jax]``. The
backend subpackages are *not* imported eagerly here, so installing only
one extra is sufficient.

Example
-------

.. code-block:: python

    from omnibias.pinn import CoordinateSpec, ComponentSpec
    from omnibias.pinn.torch.fields import OneLayerVectorField

    field = OneLayerVectorField(
        coordinate_spec=CoordinateSpec(("x", "y", "t")),
        components=ComponentSpec(("u", "v"), groups={"velocity": ("u", "v")}),
        hidden=64,
        base="tanh",
    )
    state = field(coords)              # FieldState
    state.u.lap                        # Delta u, closed form
    state.velocity.curl                # nabla x u, closed form

.. important::

    **Bit-parity with the PyTorch twin requires 64-bit JAX** --
    ``jax.config.update("jax_enable_x64", True)`` before the first JAX array is
    created (or ``JAX_ENABLE_X64=1``). JAX otherwise truncates to ``float32``
    while PyTorch uses ``float64``, so the twins stay internally consistent but
    agree only to ``float32`` tolerance. Where a value feeds a threshold, a
    rounding step or an ``argmax``, that is enough to change the decision rather
    than just the last digits. See :mod:`omnibias.jax.precision`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.pinn._core import (
    ComponentSpec,
    ComponentView,
    CoordinateSpec,
    EquationSpec,
    FieldBase,
    FieldState,
    IncompressibilityPolicy,
    ResidualPolicy,
    SigmaCache,
    VectorView,
    ops_registry,
    registry,
)
from omnibias.pinn.extensions import (
    register_lim_along,
    unregister_lim_along,
)

try:
    __version__ = _pkg_version("omnibias-pinn")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "ComponentSpec",
    "ComponentView",
    "CoordinateSpec",
    "EquationSpec",
    "FieldBase",
    "FieldState",
    "IncompressibilityPolicy",
    "ResidualPolicy",
    "SigmaCache",
    "VectorView",
    "__lineage__",
    "__version__",
    "ops_registry",
    "register_lim_along",
    "registry",
    "unregister_lim_along",
]
