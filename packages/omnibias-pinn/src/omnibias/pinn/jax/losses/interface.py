# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Interface residuals for domain decomposition (JAX twin).

Bit-parity twin of :mod:`omnibias.pinn.torch.losses.interface`. Two subdomain
fields, one seam, and the two conditions that glue them:

.. math::

   [\![u]\!] = u_+ - u_-,
   \qquad
   [\![k \partial_n u]\!] = k_+ \nabla u_+ \cdot n - k_- \nabla u_- \cdot n,

plus, optionally, XPINN's residual continuity ``r_+ - r_-``. Value continuity
alone is not enough: it is cheap to satisfy and says nothing about whether the
pieces exchange the right flux, which is what makes a decomposition
conservative and what carries a genuine kink when ``k_+ != k_-``.

Orientation follows :class:`~omnibias.pinn._core.interface.Interface`: the
``plus`` state is the field on the side the normal points into (``n.x > c``).

The geometry is shared, backend-free numpy in
:mod:`omnibias.pinn._core.interface`, so both backends sample *the same*
interface points and the parity tests compare like with like.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.interface import Interface, InterfaceSpec
from omnibias.pinn._core.state import FieldState


class InterfaceOutput(NamedTuple):
    """Jumps across an interface, one column per component.

    ``residual_jump`` is ``None`` unless PDE residuals were supplied -- it is
    XPINN's third condition, and it only means anything when both sides are
    discretising the *same* equation.

    ``diag`` holds **arrays**, not Python floats, matching the equation modules:
    reading a traced value out to a float would make the whole residual
    un-``jit``-able, and the seam loss belongs inside the jitted step. Call
    ``float(...)`` on a diagnostic at the point where it is logged.
    """

    value_jump: Array
    flux_jump: Array
    residual_jump: Array | None
    diag: dict[str, Array]


def _normal_vector(
    interface: Interface | InterfaceSpec | Sequence[float] | Array,
    state: FieldState,
) -> Array:
    """Resolve the interface normal to a ``(D,)`` unit array for ``state``."""
    if isinstance(interface, InterfaceSpec):
        raw: Sequence[float] | Array = interface.interface.normal
    elif isinstance(interface, Interface):
        raw = interface.normal
    else:
        raw = interface
        if not isinstance(raw, Array) and not any(float(v) != 0.0 for v in raw):
            raise ValueError("normal must be non-zero (it orients the interface)")
    n = jnp.asarray(raw, dtype=state.coords.dtype).reshape(-1)
    d = state.coordinate_spec.ndim
    if n.shape[0] != d:
        raise ValueError(
            f"normal has {n.shape[0]} entries but the coordinate spec is {d}-D "
            f"({state.coordinate_spec.axes}); the normal lives in the full "
            f"coordinate space, so a purely spatial interface has 0 in the time slot"
        )
    unit: Array = n / jnp.linalg.norm(n)
    return unit


def normal_derivative(
    state: FieldState,
    name: str,
    *,
    normal: Interface | InterfaceSpec | Sequence[float] | Array,
) -> Array:
    """``dU_name/dn = grad u . n`` of shape ``(B,)``.

    The gradient is taken over **all** coordinate axes, time included, so an
    interface may be oriented in space-time (a temporal seam between marching
    windows is an interface too). A spatial interface simply carries a zero in
    the time slot of its normal.
    """
    n = _normal_vector(normal, state)
    axes = tuple(range(state.coordinate_spec.ndim))
    g = state.ops.gradient(state, name, axes=axes)  # (B, D)
    out: Array = g @ n
    return out


def normal_flux(
    state: FieldState,
    names: Sequence[str],
    *,
    normal: Interface | InterfaceSpec | Sequence[float] | Array,
    conductivity: float = 1.0,
) -> Array:
    """``k dU/dn`` for each component, shape ``(B, len(names))``."""
    cols = [normal_derivative(state, nm, normal=normal) for nm in names]
    return float(conductivity) * jnp.stack(cols, axis=-1)


def value_jump(
    state_plus: FieldState, state_minus: FieldState, names: Sequence[str]
) -> Array:
    """``u_+ - u_-`` for each component, shape ``(B, len(names))``."""
    cols = [
        state_plus.ops.value(state_plus, nm) - state_minus.ops.value(state_minus, nm)
        for nm in names
    ]
    return jnp.stack(cols, axis=-1)


def flux_jump(
    state_plus: FieldState,
    state_minus: FieldState,
    names: Sequence[str],
    *,
    normal: Interface | InterfaceSpec | Sequence[float] | Array,
    conductivity: tuple[float, float] = (1.0, 1.0),
) -> Array:
    """``k_+ dU_+/dn - k_- dU_-/dn``, shape ``(B, len(names))``.

    Both sides are differentiated along the *same* normal, so a smooth field
    split by a fictitious interface gives exactly zero, while a genuine material
    jump ``k_+ != k_-`` gives the kink the physics demands.
    """
    k_plus, k_minus = (float(v) for v in conductivity)
    a = normal_flux(state_plus, names, normal=normal, conductivity=k_plus)
    b = normal_flux(state_minus, names, normal=normal, conductivity=k_minus)
    return a - b


def _resolve(
    interface: Interface | InterfaceSpec,
    conductivity: tuple[float, float] | None,
) -> tuple[Interface, tuple[float, float]]:
    """Geometry and the ``(k_+, k_-)`` pair, with an explicit override winning."""
    if isinstance(interface, InterfaceSpec):
        geom = interface.interface
        k = interface.conductivity if conductivity is None else conductivity
    else:
        geom = interface
        k = (1.0, 1.0) if conductivity is None else conductivity
    return geom, (float(k[0]), float(k[1]))


def interface_residual(
    state_plus: FieldState,
    state_minus: FieldState,
    interface: Interface | InterfaceSpec,
    *,
    names: Sequence[str] | None = None,
    conductivity: tuple[float, float] | None = None,
    residuals: tuple[Array, Array] | None = None,
) -> InterfaceOutput:
    """Both interface conditions at once, plus XPINN's residual continuity.

    ``state_plus`` and ``state_minus`` must be the two subdomain fields
    evaluated at the **same** interface points (see
    :func:`~omnibias.pinn._core.interface.interface_points`); the jump is
    pointwise, so a mismatch in the point sets is a silent wrong answer rather
    than an error, and the op checks the batch sizes for exactly that reason.

    ``conductivity`` overrides the pair carried by an
    :class:`~omnibias.pinn._core.interface.InterfaceSpec`; passing a bare
    :class:`~omnibias.pinn._core.interface.Interface` defaults it to ``(1, 1)``.
    """
    geom, k = _resolve(interface, conductivity)
    if names is None:
        names = state_plus.components.names
    if state_plus.coords.shape[0] != state_minus.coords.shape[0]:
        raise ValueError(
            f"the two sides must be evaluated at the same interface points: got "
            f"{state_plus.coords.shape[0]} and {state_minus.coords.shape[0]} points"
        )
    jump = value_jump(state_plus, state_minus, names)
    flux = flux_jump(state_plus, state_minus, names, normal=geom, conductivity=k)
    res_jump = None if residuals is None else residuals[0] - residuals[1]
    diag = {
        "max_abs_value_jump": jnp.abs(jump).max(),
        "max_abs_flux_jump": jnp.abs(flux).max(),
        "mean_sq_value_jump": (jump**2).mean(),
        "mean_sq_flux_jump": (flux**2).mean(),
    }
    if res_jump is not None:
        diag["mean_sq_residual_jump"] = (res_jump**2).mean()
    return InterfaceOutput(
        value_jump=jump, flux_jump=flux, residual_jump=res_jump, diag=diag
    )


def interface_loss(
    out: InterfaceOutput,
    *,
    weights: tuple[float, float] = (1.0, 1.0),
    residual_weight: float = 1.0,
) -> Array:
    """Weighted mean-square of the jumps: ``w_v |[u]|^2 + w_f |[k du/dn]|^2``.

    The two terms have different units whenever ``k`` does, which is why they
    carry separate weights rather than being summed raw; an
    :class:`~omnibias.pinn._core.interface.InterfaceSpec` can hold the balance
    alongside the geometry, and a
    :class:`~omnibias.pinn._core.weighting.LossWeighter` can tune it during
    training like any other term.
    """
    w_value, w_flux = (float(v) for v in weights)
    if w_value < 0.0 or w_flux < 0.0 or residual_weight < 0.0:
        raise ValueError(
            f"interface weights must be non-negative, got "
            f"{(w_value, w_flux, float(residual_weight))}"
        )
    total = w_value * (out.value_jump**2).mean() + w_flux * (out.flux_jump**2).mean()
    if out.residual_jump is not None and residual_weight:
        total = total + float(residual_weight) * (out.residual_jump**2).mean()
    return total


__all__ = [
    "InterfaceOutput",
    "flux_jump",
    "interface_loss",
    "interface_residual",
    "normal_derivative",
    "normal_flux",
    "value_jump",
]
