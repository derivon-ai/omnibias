# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Spinor structure and Pauli / Dirac gamma-matrix utilities.

A 2-spinor :math:`\psi^a` (Weyl) or a 4-spinor :math:`\psi^a` (Dirac)
has every entry independently complex-valued. We encode an
:math:`n`-spinor as :math:`2n` real components in a single
:class:`ComponentSpec` group:

- component channel names follow ``{group}_{a}_{re,im}`` for
  :math:`a = 0, \dots, n-1`,
- a sub-group ``{group}_{a}`` carries the real / imaginary channels for
  the :math:`a`-th spinor index (and is therefore a wavefunction group
  in the sense of :mod:`omnibias.qpinn._core.complex`).

Conventions
-----------

We adopt the **mostly-minus metric** (Peskin & Schroeder) :math:`\eta_{\mu\nu} =
\text{diag}(+1, -1, -1, -1)` so that the Dirac matrices satisfy

.. math::

    \{\gamma^\mu, \gamma^\nu\} = 2 \eta^{\mu\nu} \mathbb{1}_4,

i.e. :math:`(\gamma^0)^2 = +I_4` and :math:`(\gamma^i)^2 = -I_4` for
:math:`i \in \{1, 2, 3\}`.

Two representations are provided:

- ``"dirac"`` (also called the *standard* representation), used for
  non-relativistic limits.
- ``"weyl"`` (also called the *chiral* representation), used for
  massless / high-energy limits.

Both share the same Pauli matrices :math:`\sigma_x, \sigma_y, \sigma_z`
embedded in the 4-block structure; only the placement differs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from omnibias.pinn._core.components import ComponentSpec

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState

T = TypeVar("T")

#: Identity 2x2.
_I2: np.ndarray = np.eye(2, dtype=np.complex128)
#: Pauli matrix sigma_x.
PAULI_X: np.ndarray = np.array([[0, 1], [1, 0]], dtype=np.complex128)
#: Pauli matrix sigma_y.
PAULI_Y: np.ndarray = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
#: Pauli matrix sigma_z.
PAULI_Z: np.ndarray = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def pauli_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sigma_x, sigma_y, sigma_z)`` as ``(2, 2)`` complex arrays."""
    return PAULI_X, PAULI_Y, PAULI_Z


def _block(top_left, top_right, bottom_left, bottom_right) -> np.ndarray:
    """Glue four 2x2 matrices into a 4x4 matrix."""
    out = np.zeros((4, 4), dtype=np.complex128)
    out[:2, :2] = top_left
    out[:2, 2:] = top_right
    out[2:, :2] = bottom_left
    out[2:, 2:] = bottom_right
    return out


def gamma_matrices(
    representation: Literal["dirac", "weyl"] = "dirac",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Return ``(gamma_0, gamma_1, gamma_2, gamma_3)`` for a given representation.

    The convention here is the mostly-minus metric :math:`\eta = \text{diag}
    (+1, -1, -1, -1)` and the Dirac (or Weyl) representations as in
    Peskin & Schroeder.

    Parameters
    ----------
    representation
        One of:

        - ``"dirac"`` (standard): :math:`\gamma^0 =
          \text{diag}(I_2, -I_2)`, :math:`\gamma^i = \begin{pmatrix} 0
          & \sigma_i \\ -\sigma_i & 0 \end{pmatrix}`.
        - ``"weyl"`` (chiral): :math:`\gamma^0 = \begin{pmatrix} 0 & I_2
          \\ I_2 & 0 \end{pmatrix}`, :math:`\gamma^i = \begin{pmatrix} 0
          & \sigma_i \\ -\sigma_i & 0 \end{pmatrix}`.

    Returns
    -------
    tuple of 4 ndarrays
        Each of shape ``(4, 4)`` with ``dtype=complex128``.

    Raises
    ------
    ValueError
        If ``representation`` is not one of ``"dirac"``, ``"weyl"``.

    Examples
    --------
    >>> g0, g1, g2, g3 = gamma_matrices("dirac")
    >>> g0 @ g0  # gamma_0^2 = I for the standard representation
    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])
    """
    zero = np.zeros((2, 2), dtype=np.complex128)
    if representation == "dirac":
        g0 = _block(_I2, zero, zero, -_I2)
    elif representation == "weyl":
        g0 = _block(zero, _I2, _I2, zero)
    else:
        raise ValueError(
            f"unknown representation {representation!r}; expected 'dirac' or 'weyl'"
        )
    g1 = _block(zero, PAULI_X, -PAULI_X, zero)
    g2 = _block(zero, PAULI_Y, -PAULI_Y, zero)
    g3 = _block(zero, PAULI_Z, -PAULI_Z, zero)
    return g0, g1, g2, g3


def gamma5(
    representation: Literal["dirac", "weyl"] = "dirac",
) -> np.ndarray:
    r"""Return :math:`\gamma^5 = i\gamma^0\gamma^1\gamma^2\gamma^3`.

    For the chiral (Weyl) representation this is diagonal:
    :math:`\gamma^5 = \text{diag}(-I_2, +I_2)`. The 0 / 5 product gives
    the chirality projector :math:`P_L = (1 - \gamma^5)/2`.
    """
    g0, g1, g2, g3 = gamma_matrices(representation)
    return 1j * g0 @ g1 @ g2 @ g3


def make_spinor_components(
    name: str = "spinor",
    *,
    n_components: int = 4,
) -> ComponentSpec:
    r"""Build a :class:`ComponentSpec` carrying an :math:`n`-spinor.

    Parameters
    ----------
    name
        Spinor group name. Component channels are
        ``f"{name}_{a}_re"`` and ``f"{name}_{a}_im"`` for
        ``a = 0, ..., n_components - 1``. Each component-pair is
        bundled under the sub-group ``f"{name}_{a}"`` so it is a valid
        wavefunction group for
        :func:`omnibias.qpinn._core.complex.psi_value`.
    n_components
        Number of spinor indices. Default ``4`` (Dirac); use ``2`` for
        a Weyl 2-spinor.

    Returns
    -------
    ComponentSpec
        Frozen spec with ``2 * n_components`` real channels.

    Raises
    ------
    ValueError
        If ``name`` is empty or ``n_components < 1``.

    Examples
    --------
    >>> spec = make_spinor_components(name="psi", n_components=2)
    >>> spec.names
    ('psi_0_re', 'psi_0_im', 'psi_1_re', 'psi_1_im')
    >>> spec.group_members("psi_0")
    ('psi_0_re', 'psi_0_im')
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string, got {name!r}")
    if n_components < 1:
        raise ValueError(
            f"n_components must be >= 1, got {n_components}"
        )
    channel_names: list[str] = []
    groups: dict[str, tuple[str, ...]] = {}
    for a in range(n_components):
        re_name = f"{name}_{a}_re"
        im_name = f"{name}_{a}_im"
        channel_names.append(re_name)
        channel_names.append(im_name)
        groups[f"{name}_{a}"] = (re_name, im_name)
    groups[name] = tuple(channel_names)
    return ComponentSpec(names=tuple(channel_names), groups=groups)


def spinor_value(
    state: FieldState[T],
    group: str,
    component_idx: int,
) -> tuple[T, T]:
    """Return ``(re, im)`` of the ``component_idx``-th spinor component."""
    re_name = f"{group}_{component_idx}_re"
    im_name = f"{group}_{component_idx}_im"
    re = state.ops.value(state, re_name)
    im = state.ops.value(state, im_name)
    return re, im


def pauli_dot(
    state: FieldState[T],
    *,
    A: tuple[Callable[[FieldState], T], Callable[[FieldState], T], Callable[[FieldState], T]],
    spinor_group: str = "spinor",
) -> tuple[tuple[T, T], tuple[T, T]]:
    r"""Apply :math:`\sigma\cdot A` to a 2-spinor on ``state``.

    With :math:`A = (A_x, A_y, A_z)` and the wavefunction
    :math:`\psi = (\psi_0, \psi_1)`, the result is

    .. math::

        (\sigma\cdot A)\psi
        = \big(A_z\,\psi_0 + (A_x - i\,A_y)\,\psi_1,\;
              (A_x + i\,A_y)\,\psi_0 - A_z\,\psi_1\big).

    Splitting into real / imaginary parts gives the four-tuple this
    function returns.

    Parameters
    ----------
    state
        :class:`FieldState` carrying the spinor group.
    A
        Three callables, each ``A_i(state) -> Tensor`` returning the
        corresponding vector-field component on the collocation points.
    spinor_group
        Name of the 2-spinor group. Must be a 4-channel
        :class:`ComponentSpec` group built by
        :func:`make_spinor_components` with ``n_components=2``.

    Returns
    -------
    ((out_0_re, out_0_im), (out_1_re, out_1_im))
        Real / imaginary parts of each output spinor component.
    """
    A_x = A[0](state)
    A_y = A[1](state)
    A_z = A[2](state)
    psi_re_0, psi_im_0 = spinor_value(state, spinor_group, 0)
    psi_re_1, psi_im_1 = spinor_value(state, spinor_group, 1)
    out_re_0 = A_x * psi_re_1 + A_y * psi_im_1 + A_z * psi_re_0
    out_im_0 = A_x * psi_im_1 - A_y * psi_re_1 + A_z * psi_im_0
    out_re_1 = A_x * psi_re_0 - A_y * psi_im_0 - A_z * psi_re_1
    out_im_1 = A_x * psi_im_0 + A_y * psi_re_0 - A_z * psi_im_1
    return (out_re_0, out_im_0), (out_re_1, out_im_1)


def apply_gamma_matrix(
    state: FieldState[T],
    *,
    matrix: np.ndarray,
    spinor_group: str = "spinor",
) -> tuple[tuple[T, T], ...]:
    r"""Apply a 4x4 complex matrix ``matrix`` to a 4-spinor.

    Given a constant 4x4 complex matrix :math:`M^a_{\ b} = M^{re}_{ab} +
    i M^{im}_{ab}` and a 4-spinor :math:`\psi^b = \psi^b_R + i \psi^b_I`
    encoded as 8 real channels, this computes
    :math:`(M\psi)^a` in split-real form:

    .. math::

        (M\psi)^a_R &= \sum_b\big(M^{re}_{ab}\psi^b_R - M^{im}_{ab}\psi^b_I\big),\\
        (M\psi)^a_I &= \sum_b\big(M^{re}_{ab}\psi^b_I + M^{im}_{ab}\psi^b_R\big).

    Parameters
    ----------
    state
        :class:`FieldState` carrying the spinor group.
    matrix
        ``(4, 4)`` complex numpy array. The function does **not** copy
        it onto the backend's device; it just reads the entries. For
        the Dirac / Weyl gamma matrices use :func:`gamma_matrices`.
    spinor_group
        Spinor group name; the spec must have
        ``n_components = 4``.

    Returns
    -------
    tuple of four ``(re, im)`` pairs
        One pair per output spinor component, total 8 backend tensors.
    """
    if matrix.shape != (4, 4):
        raise ValueError(
            f"matrix must be (4, 4); got {matrix.shape}"
        )
    psi_re = [None] * 4
    psi_im = [None] * 4
    for b in range(4):
        psi_re[b], psi_im[b] = spinor_value(state, spinor_group, b)

    out: list[tuple[T, T]] = []
    for a in range(4):
        re_acc: T | None = None
        im_acc: T | None = None
        for b in range(4):
            entry = matrix[a, b]
            entry_re = float(entry.real)
            entry_im = float(entry.imag)
            re_contrib = entry_re * psi_re[b] - entry_im * psi_im[b]
            im_contrib = entry_re * psi_im[b] + entry_im * psi_re[b]
            re_acc = re_contrib if re_acc is None else re_acc + re_contrib
            im_acc = im_contrib if im_acc is None else im_acc + im_contrib
        assert re_acc is not None
        assert im_acc is not None
        out.append((re_acc, im_acc))
    return tuple(out)


def gamma_partial_psi(
    state: FieldState[T],
    *,
    spinor_group: str = "spinor",
    representation: Literal["dirac", "weyl"] = "dirac",
) -> tuple[tuple[T, T], ...]:
    r"""Compute :math:`\sum_\mu \gamma^\mu \partial_\mu \psi`.

    The result is a 4-spinor (8 split-real channels). The temporal
    partial is taken along the coordinate spec's ``time_axis`` and the
    spatial partials along its spatial axes (in the order
    ``coordinate_spec.spatial_axes``).

    Parameters
    ----------
    state
        :class:`FieldState`. Must have at least one spatial axis. If
        a time axis is missing, only the spatial terms contribute.
    spinor_group
        Spinor group name; n_components must be 4.
    representation
        One of ``"dirac"`` or ``"weyl"``.

    Returns
    -------
    tuple of four ``(re, im)`` pairs
        :math:`(\gamma^\mu \partial_\mu \psi)^a` for ``a = 0, ..., 3``.

    Notes
    -----
    Spatial indices used by :math:`\gamma^i` are 1, 2, 3 corresponding
    to the first, second, and third spatial axes of the
    :attr:`CoordinateSpec` (i.e. *not* mapped by name). For a domain
    with only :math:`x, y` (no :math:`z`), :math:`\gamma^3 \partial_3 = 0`
    and is dropped from the sum.
    """
    gammas = gamma_matrices(representation)
    time = state.coordinate_spec.time_axis
    spatial = state.coordinate_spec.spatial_axes

    if time is not None:
        re_acc, im_acc = _gamma_partial_one_axis(
            state, gammas[0], time_axis=time, spinor_group=spinor_group,
        )
        out_pairs = [(re_acc[a], im_acc[a]) for a in range(4)]
    else:
        out_pairs = None  # type: ignore[assignment]

    for mu in range(3):
        if mu >= len(spatial):
            break
        re_contrib, im_contrib = _gamma_partial_one_axis(
            state, gammas[mu + 1], time_axis=spatial[mu],
            spinor_group=spinor_group,
        )
        if out_pairs is None:
            out_pairs = [(re_contrib[a], im_contrib[a]) for a in range(4)]
        else:
            out_pairs = [
                (out_pairs[a][0] + re_contrib[a], out_pairs[a][1] + im_contrib[a])
                for a in range(4)
            ]
    if out_pairs is None:
        raise ValueError(
            "gamma_partial_psi requires at least one axis (spatial or time)"
        )
    return tuple(out_pairs)


def _gamma_partial_one_axis(
    state: FieldState[T],
    matrix: np.ndarray,
    *,
    time_axis: int | str,
    spinor_group: str,
) -> tuple[list[T], list[T]]:
    """Helper: apply a 4x4 matrix to ``partial_axis psi`` and return list."""
    psi_re_d = [None] * 4
    psi_im_d = [None] * 4
    for b in range(4):
        re_name = f"{spinor_group}_{b}_re"
        im_name = f"{spinor_group}_{b}_im"
        psi_re_d[b] = state.ops.derivative(state, re_name, axis=time_axis, order=1)
        psi_im_d[b] = state.ops.derivative(state, im_name, axis=time_axis, order=1)

    re_out: list[T] = []
    im_out: list[T] = []
    for a in range(4):
        re_acc: T | None = None
        im_acc: T | None = None
        for b in range(4):
            entry = matrix[a, b]
            er = float(entry.real)
            ei = float(entry.imag)
            re_contrib = er * psi_re_d[b] - ei * psi_im_d[b]
            im_contrib = er * psi_im_d[b] + ei * psi_re_d[b]
            re_acc = re_contrib if re_acc is None else re_acc + re_contrib
            im_acc = im_contrib if im_acc is None else im_acc + im_contrib
        assert re_acc is not None
        assert im_acc is not None
        re_out.append(re_acc)
        im_out.append(im_acc)
    return re_out, im_out


__all__ = [
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "apply_gamma_matrix",
    "gamma5",
    "gamma_matrices",
    "gamma_partial_psi",
    "make_spinor_components",
    "pauli_dot",
    "pauli_matrices",
    "spinor_value",
]
