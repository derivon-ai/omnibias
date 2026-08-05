# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Split-real encoding of a complex wavefunction.

For v0.0.1 every quantum-PINN field encodes a complex wavefunction
:math:`\psi : \Omega \to \mathbb{C}` as two real-valued components

.. math::

    \psi(x) = \psi_R(x) + i\,\psi_I(x),

stored as two named components in a single :class:`ComponentSpec` group.
The helpers below build that spec and provide pointwise / operator
wrappers that the equation residuals consume.

Why split-real and not native complex?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real-channel encoding lets the existing :mod:`omnibias.pinn` fields
(``OneLayerVectorField``, ``SpectralVectorField``, ...) be used as-is
without any change to ``ComponentSpec`` or the ``ops`` dispatch surface.
Both backends already vectorise over independent component channels, so
``laplacian(psi_re)`` and ``laplacian(psi_im)`` are computed in the same
sweep as the channel dimension. Native complex-valued
:class:`ComponentSpec` is a future v0.0.2 enhancement.

Convention
~~~~~~~~~~

- Group name: the user-facing wavefunction name (default ``"psi"``).
- Component names: ``f"{name}_re"`` and ``f"{name}_im"``.
- Order: real first, imaginary second.

Example
~~~~~~~

.. code-block:: python

    from omnibias.qpinn._core.complex import make_psi_components, psi_value

    components = make_psi_components(name="psi")
    # components.names    -> ("psi_re", "psi_im")
    # components.groups   -> (("psi", ("psi_re", "psi_im")),)
    psi_re, psi_im = psi_value(state, "psi")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from omnibias.pinn._core.components import ComponentSpec

if TYPE_CHECKING:  # pragma: no cover -- typing-only import
    from omnibias.pinn._core.state import FieldState

T = TypeVar("T")


def make_psi_components(
    name: str = "psi",
    *,
    extra_groups: dict[str, tuple[str, ...]] | None = None,
) -> ComponentSpec:
    """Build a :class:`ComponentSpec` carrying a complex wavefunction.

    Parameters
    ----------
    name
        Wavefunction name. The two real components are ``f"{name}_re"``
        and ``f"{name}_im"`` and they are bundled under the group
        ``name`` so callers can refer to the wavefunction as a single
        object.
    extra_groups
        Optional additional groups on the same spec (e.g. an outer
        ``{"all": ("psi_re", "psi_im")}`` if the user wants a second
        alias). Each member must be a known component.

    Returns
    -------
    ComponentSpec
        Frozen spec with two real components and a group of length two.

    Raises
    ------
    ValueError
        If ``name`` is empty or contains a ``_re`` / ``_im`` suffix
        (which would collide with the auto-generated component names).

    Examples
    --------
    >>> components = make_psi_components(name="psi")
    >>> components.names
    ('psi_re', 'psi_im')
    >>> components.group_members("psi")
    ('psi_re', 'psi_im')
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string, got {name!r}")
    if name.endswith("_re") or name.endswith("_im"):
        raise ValueError(
            f"name {name!r} must not end with '_re' or '_im' "
            "(that would collide with the auto-generated component names)"
        )
    re_name = f"{name}_re"
    im_name = f"{name}_im"
    groups: dict[str, tuple[str, ...]] = {name: (re_name, im_name)}
    if extra_groups is not None:
        for g_name, members in extra_groups.items():
            if g_name == name:
                raise ValueError(
                    f"extra_groups[{g_name!r}] collides with the wavefunction group"
                )
            for m in members:
                if m not in (re_name, im_name):
                    raise ValueError(
                        f"extra_groups[{g_name!r}] member {m!r} not in "
                        f"({re_name!r}, {im_name!r})"
                    )
            groups[g_name] = tuple(members)
    return ComponentSpec(names=(re_name, im_name), groups=groups)


def _resolve_psi_group(state: FieldState[T], group: str) -> tuple[str, str]:
    """Return ``(re_name, im_name)`` for a wavefunction group on this state.

    Validates that ``group`` is a group on ``state.components`` carrying
    exactly two members in the canonical ``(re, im)`` order.
    """
    components = state.components
    if not components.is_group(group):
        raise KeyError(
            f"{group!r} is not a group on this FieldState; known groups: "
            f"{tuple(g for g, _ in components.groups)!r}"
        )
    members = components.group_members(group)
    if len(members) != 2:
        raise ValueError(
            f"wavefunction group {group!r} must have exactly 2 components "
            f"(re, im); got {len(members)}: {members!r}"
        )
    re_name, im_name = members
    return re_name, im_name


def is_psi_group(state: FieldState[Any], group: str) -> bool:
    """Return whether ``group`` is a valid two-component wavefunction group."""
    components = state.components
    if not components.is_group(group):
        return False
    members = components.group_members(group)
    return len(members) == 2


def psi_value(
    state: FieldState[T], group: str = "psi",
) -> tuple[T, T]:
    """Return the ``(psi_re, psi_im)`` value pair on ``state``.

    Parameters
    ----------
    state
        :class:`FieldState` produced by an ``omnibias.pinn`` field.
    group
        Name of the wavefunction group. Default ``"psi"``.

    Returns
    -------
    (psi_re, psi_im)
        Two tensors / arrays of shape ``(B,)`` (matches the backend
        convention for scalar-component values).

    Notes
    -----
    This is a thin wrapper around ``state.ops.value(state, name)`` for
    each of the two channels. The wavefunction is the *value* level; for
    derivatives use :func:`apply_kinetic` or call ``state.ops.derivative``
    directly on the underlying component names.
    """
    re_name, im_name = _resolve_psi_group(state, group)
    re = state.ops.value(state, re_name)
    im = state.ops.value(state, im_name)
    return re, im


def psi_density(
    state: FieldState[T], group: str = "psi",
) -> T:
    r"""Return the probability density ``|psi|^2 = psi_re^2 + psi_im^2``.

    Parameters
    ----------
    state
        :class:`FieldState` carrying the wavefunction group.
    group
        Name of the wavefunction group. Default ``"psi"``.

    Returns
    -------
    Tensor / array
        Same backend type as ``psi_re``; shape ``(B,)``.
    """
    re, im = psi_value(state, group)
    return re * re + im * im  # type: ignore[operator,no-any-return]


def psi_phase(
    state: FieldState[T], group: str = "psi", *, atan2: Callable[[T, T], T] | None = None,
) -> T:
    r"""Return the phase ``arg(psi) = atan2(psi_im, psi_re)`` in ``[-pi, pi]``.

    Parameters
    ----------
    state
        :class:`FieldState` carrying the wavefunction group.
    group
        Name of the wavefunction group.
    atan2
        Backend-specific two-arg arctangent. Required because there is no
        canonical pure-Python ``atan2`` for tensor types. Use
        ``torch.atan2`` (torch) or ``jax.numpy.arctan2`` (jax).

    Returns
    -------
    Tensor / array
        Phase angle, shape ``(B,)``.
    """
    if atan2 is None:
        raise TypeError(
            "psi_phase requires the backend-specific atan2 callable; pass "
            "torch.atan2 or jax.numpy.arctan2"
        )
    re, im = psi_value(state, group)
    return atan2(im, re)


def apply_kinetic(
    state: FieldState[T],
    *,
    group: str = "psi",
    hbar: float = 1.0,
    mass: float = 1.0,
) -> tuple[T, T]:
    r"""Apply the kinetic operator ``T = -hbar^2 / (2 m) * Laplacian``.

    Parameters
    ----------
    state
        :class:`FieldState` carrying the wavefunction group.
    group
        Name of the wavefunction group.
    hbar
        Planck constant. Default 1.0 (atomic units).
    mass
        Particle mass. Default 1.0 (atomic units / electron mass).

    Returns
    -------
    (T_re, T_im)
        Two tensors / arrays giving the real and imaginary parts of
        :math:`-\hbar^2/(2m)\,\nabla^2\psi`.

    Notes
    -----
    The kinetic operator commutes with complex conjugation (no ``i``
    factor), so it acts on the two channels independently.
    """
    if mass <= 0:
        raise ValueError(f"mass must be > 0, got {mass}")
    re_name, im_name = _resolve_psi_group(state, group)
    coeff = -0.5 * hbar * hbar / mass
    lap_re = state.ops.laplacian(state, re_name)
    lap_im = state.ops.laplacian(state, im_name)
    return coeff * lap_re, coeff * lap_im


def apply_potential(
    state: FieldState[T],
    *,
    group: str = "psi",
    potential: Callable[[FieldState[T]], T] | None = None,
) -> tuple[T, T]:
    r"""Apply the local potential operator ``V(x) * psi``.

    Parameters
    ----------
    state
        :class:`FieldState` carrying the wavefunction group.
    group
        Name of the wavefunction group.
    potential
        Callable ``V(state) -> Tensor of shape (B,)``. If ``None``, the
        free-particle case ``V = 0`` is used and a zero tensor is
        returned.

    Returns
    -------
    (V_re, V_im)
        Pointwise ``V * psi_re``, ``V * psi_im``.

    Notes
    -----
    The potential is treated as real-valued (Hermitian by assumption).
    For a complex-valued potential (e.g. absorbing boundary), pass a
    callable that returns ``V_re`` only and compose with a custom
    imaginary-V residual; native complex-V support is deferred to
    v0.0.2.
    """
    re, im = psi_value(state, group)
    if potential is None:
        zero = 0.0 * re  # type: ignore[operator]
        return zero, zero  # type: ignore[return-value]
    V = potential(state)
    return V * re, V * im  # type: ignore[operator,return-value]


def apply_hamiltonian(
    state: FieldState[T],
    *,
    group: str = "psi",
    hbar: float = 1.0,
    mass: float = 1.0,
    potential: Callable[[FieldState[T]], T] | None = None,
) -> tuple[T, T]:
    r"""Apply the Schrodinger Hamiltonian ``H = T + V`` to ``psi``.

    Equivalent to ``apply_kinetic(...) + apply_potential(...)`` channel
    by channel. Provided as a single entry point because the kinetic
    and potential terms are always summed in the residual.

    Parameters
    ----------
    state
        :class:`FieldState` carrying the wavefunction group.
    group
        Wavefunction group name. Default ``"psi"``.
    hbar
        Planck constant. Default 1.0.
    mass
        Particle mass. Default 1.0.
    potential
        Callable ``V(state) -> Tensor of shape (B,)``. Default ``None``
        (free particle).

    Returns
    -------
    (H_re, H_im)
        Real and imaginary parts of ``H psi``.

    Examples
    --------
    Time-dependent Schrodinger residual ``i hbar psi_t - H psi``:

    .. code-block:: python

        psi_re_t = state.ops.derivative(state, "psi_re", axis="t", order=1)
        psi_im_t = state.ops.derivative(state, "psi_im", axis="t", order=1)
        H_re, H_im = apply_hamiltonian(state, hbar=1.0, mass=1.0,
                                       potential=lambda s: 0.5 * s.coords[..., 0]**2)
        res_re = -hbar * psi_im_t - H_re
        res_im =  hbar * psi_re_t - H_im
    """
    T_re, T_im = apply_kinetic(state, group=group, hbar=hbar, mass=mass)
    V_re, V_im = apply_potential(state, group=group, potential=potential)
    return T_re + V_re, T_im + V_im  # type: ignore[operator,return-value]


def apply_angular_momentum_z(
    state: FieldState[T],
    *,
    group: str = "psi",
    hbar: float = 1.0,
    x_axis: int | str = 0,
    y_axis: int | str = 1,
) -> tuple[T, T]:
    r"""Apply the planar angular-momentum operator.

    :math:`L_z = -i\hbar\,(x\,\partial_y - y\,\partial_x)`. For
    :math:`\psi = \psi_R + i\psi_I` this returns

    .. math::

        \mathrm{Re}(L_z\psi) &= \hbar\,(x\,\partial_y\psi_I - y\,\partial_x\psi_I),\\
        \mathrm{Im}(L_z\psi) &= -\hbar\,(x\,\partial_y\psi_R - y\,\partial_x\psi_R).

    The eigenfunctions :math:`(x+iy)^m` satisfy :math:`L_z\psi = m\hbar\,\psi`.
    """
    re_name, im_name = _resolve_psi_group(state, group)
    xi = state.coordinate_spec.axis_index(x_axis)
    yi = state.coordinate_spec.axis_index(y_axis)
    if xi == yi:
        raise ValueError(f"x_axis ({x_axis}) and y_axis ({y_axis}) must differ")
    x = state.coords[..., xi]
    y = state.coords[..., yi]
    dx_re = state.ops.derivative(state, re_name, axis=xi, order=1)
    dy_re = state.ops.derivative(state, re_name, axis=yi, order=1)
    dx_im = state.ops.derivative(state, im_name, axis=xi, order=1)
    dy_im = state.ops.derivative(state, im_name, axis=yi, order=1)
    lz_re = hbar * (x * dy_im - y * dx_im)
    lz_im = -hbar * (x * dy_re - y * dx_re)
    return lz_re, lz_im  # type: ignore[return-value]


def _angular_double(state: FieldState[Any], name: str, xi: int, yi: int) -> Any:
    r"""``A^2 f`` with ``A = x d_y - y d_x``: the bare squared angular derivative.

    :math:`A^2 f = x^2 f_{yy} + y^2 f_{xx} - 2xy\,f_{xy} - x f_x - y f_y`.
    """
    x = state.coords[..., xi]
    y = state.coords[..., yi]
    f_x = state.ops.derivative(state, name, axis=xi, order=1)
    f_y = state.ops.derivative(state, name, axis=yi, order=1)
    f_xx = state.ops.derivative(state, name, axis=xi, order=2)
    f_yy = state.ops.derivative(state, name, axis=yi, order=2)
    f_xy = state.ops.mixed_partial(state, name, (xi, yi), (1, 1))
    return x * x * f_yy + y * y * f_xx - 2.0 * x * y * f_xy - x * f_x - y * f_y


def apply_angular_momentum_squared(
    state: FieldState[T],
    *,
    group: str = "psi",
    hbar: float = 1.0,
    x_axis: int | str = 0,
    y_axis: int | str = 1,
) -> tuple[T, T]:
    r"""Apply :math:`L_z^2 = -\hbar^2 (x\,\partial_y - y\,\partial_x)^2`.

    In two dimensions this is the total angular momentum :math:`L^2`. The
    eigenfunctions :math:`(x+iy)^m` satisfy :math:`L_z^2\psi = m^2\hbar^2\,\psi`.
    """
    re_name, im_name = _resolve_psi_group(state, group)
    xi = state.coordinate_spec.axis_index(x_axis)
    yi = state.coordinate_spec.axis_index(y_axis)
    if xi == yi:
        raise ValueError(f"x_axis ({x_axis}) and y_axis ({y_axis}) must differ")
    coeff = -hbar * hbar
    lz2_re = coeff * _angular_double(state, re_name, xi, yi)
    lz2_im = coeff * _angular_double(state, im_name, xi, yi)
    return lz2_re, lz2_im  # type: ignore[return-value]


__all__ = [
    "apply_angular_momentum_squared",
    "apply_angular_momentum_z",
    "apply_hamiltonian",
    "apply_kinetic",
    "apply_potential",
    "is_psi_group",
    "make_psi_components",
    "psi_density",
    "psi_phase",
    "psi_value",
]
