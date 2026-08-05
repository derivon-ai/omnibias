# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Interfaces between subdomains: the geometry, and how to sample on one.

Domain decomposition (XPINN / cPINN) splits a hard problem into easy patches and
then has to **glue them back together**. The glue lives on a codimension-1 set --
a curve in 2-D, a surface in 3-D -- and it is two conditions, not one:

.. math::

   [\![u]\!] = u_a - u_b = 0,
   \qquad
   [\![k \partial_n u]\!] = k_a\, \nabla u_a \cdot n - k_b\, \nabla u_b \cdot n = 0.

Continuity of the *value* alone is not enough. A field can be continuous and
still leak mass or energy across the seam; it is the second condition -- normal
**flux** continuity -- that makes the pieces add up to a solution of the original
problem, and it is the one that carries a jump when the two sides have different
material coefficients (``k_a != k_b``), which is exactly the case decomposition
is usually there to handle.

This module is the geometry half, deliberately backend-free numpy:

* :class:`Interface` -- an oriented hyperplane ``{x : n.x = c}`` with unit
  normal, so its signed distance *is* a distance and its normal is directly
  usable as the ``n`` in ``d/dn``.
* :func:`interface_points` -- points drawn **on** that hyperplane and inside the
  box, which is the part that is easy to get subtly wrong. Sampling the box and
  keeping what is "close to" the interface gives points that are near it, not on
  it, and the residual then measures a jump plus a discretisation error.
* :func:`split_by_interface` -- which side each point of a set falls on, for
  routing collocation to the patch that owns it.

The residual half -- ``value_jump`` / ``flux_jump`` on two :class:`FieldState`\ s
-- is per backend, in ``omnibias.pinn.{torch,jax}.losses.interface``.

An :class:`Interface` also links the decomposition to
:class:`~omnibias.pinn.partition.torch.field.PartitionedField`:
:meth:`Interface.from_split` reads a partition gate's zero set, so the seam you
sample on is *the same* seam the partition of unity blends across, rather than a
second hyperplane that happens to be nearby.

References
----------
Jagtap & Karniadakis, *Extended Physics-Informed Neural Networks (XPINNs)*,
Commun. Comput. Phys. 28(5), 2002-2041 (2020). Jagtap, Kharazmi & Karniadakis,
*Conservative PINNs on discrete domains* (cPINN), CMAME 365, 113028 (2020).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from omnibias.pinn._core.coords import CoordinateSpec


@dataclass(frozen=True)
class Interface:
    r"""An oriented hyperplane ``{x : n.x = c}`` separating two subdomains.

    The normal is stored **unit-length** (the constructor rescales ``normal``
    and ``offset`` together, which leaves the point set unchanged), so
    :meth:`signed_distance` is a true signed distance and :attr:`unit_normal`
    can be handed straight to a normal-derivative op.

    Orientation matters and is fixed by the sign of the normal: the *plus* side
    is ``n.x > c``. Whichever field you call ``a`` in a jump must be the field
    on the plus side, or the flux jump comes out with the wrong sign.

    Parameters
    ----------
    normal:
        Any non-zero vector orthogonal to the interface; normalised on
        construction.
    offset:
        The threshold ``c``, rescaled with ``normal``.
    label:
        Free-form name, carried into diagnostics so a multi-interface problem
        reports something better than an index.

    Examples
    --------
    >>> iface = Interface.from_axis(ndim=2, axis=0, value=0.5)
    >>> iface.unit_normal.tolist()
    [1.0, 0.0]
    >>> float(iface.signed_distance(np.array([[0.9, 0.0]]))[0])
    0.4
    """

    normal: tuple[float, ...]
    offset: float
    label: str = "interface"

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=float).reshape(-1)
        if n.size == 0:
            raise ValueError("normal must be a non-empty vector")
        norm = float(np.linalg.norm(n))
        if not norm > 0.0:
            raise ValueError("normal must be non-zero (it orients the interface)")
        if not np.all(np.isfinite(n)) or not np.isfinite(self.offset):
            raise ValueError("normal and offset must be finite")
        object.__setattr__(self, "normal", tuple(float(v) for v in n / norm))
        object.__setattr__(self, "offset", float(self.offset) / norm)

    # -- geometry -------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Ambient dimension ``D``; the interface itself is ``D - 1``."""
        return len(self.normal)

    @property
    def unit_normal(self) -> np.ndarray:
        """``(D,)`` unit normal, pointing towards the plus side."""
        return np.asarray(self.normal, dtype=float)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """``n.x - c`` for each row of ``points``; positive on the plus side."""
        x = np.asarray(points, dtype=float)
        self._check_ndim(x)
        d: np.ndarray = x @ self.unit_normal - self.offset
        return d

    def side(self, points: np.ndarray) -> np.ndarray:
        """``+1`` on the plus side, ``-1`` on the minus side (ties go plus)."""
        return np.where(self.signed_distance(points) >= 0.0, 1, -1)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Orthogonal projection of ``points`` onto the interface."""
        x = np.asarray(points, dtype=float)
        d = self.signed_distance(x)
        foot: np.ndarray = x - d[:, None] * self.unit_normal[None, :]
        return foot

    def contains(self, points: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
        """Whether each point lies on the interface to within ``tol``."""
        hit: np.ndarray = np.abs(self.signed_distance(points)) <= tol
        return hit

    def tangent_basis(self) -> np.ndarray:
        """``(D, D-1)`` orthonormal basis of the interface's tangent space.

        A 1-D domain has no tangent directions -- its "interface" is a single
        point -- so the basis is legitimately empty there.
        """
        d = self.ndim
        q, _ = np.linalg.qr(np.column_stack([self.unit_normal, np.eye(d)]))
        return np.asarray(q[:, 1:d], dtype=float)

    def flip(self) -> Interface:
        """The same point set with the opposite orientation."""
        return Interface(
            normal=tuple(-v for v in self.normal),
            offset=-self.offset,
            label=self.label,
        )

    # -- constructors ---------------------------------------------------

    @classmethod
    def from_axis(
        cls, *, ndim: int, axis: int, value: float, label: str | None = None
    ) -> Interface:
        """An axis-aligned interface ``{x : x_axis = value}``."""
        if ndim < 1:
            raise ValueError(f"ndim must be >= 1, got {ndim}")
        if not 0 <= axis < ndim:
            raise IndexError(f"axis {axis} out of range for ndim {ndim}")
        n = np.zeros(ndim)
        n[axis] = 1.0
        name = label if label is not None else f"x{axis}={value:g}"
        return cls(normal=tuple(n), offset=float(value), label=name)

    @classmethod
    def from_spec(
        cls,
        coordinate_spec: CoordinateSpec,
        *,
        axis: int | str,
        value: float,
        label: str | None = None,
    ) -> Interface:
        """An axis-aligned interface named by axis, e.g. ``axis="x"``."""
        idx = coordinate_spec.axis_index(axis)
        name = label if label is not None else f"{coordinate_spec.axes[idx]}={value:g}"
        return cls.from_axis(
            ndim=coordinate_spec.ndim, axis=idx, value=value, label=name
        )

    @classmethod
    def from_split(
        cls,
        split_dirs: Sequence[Sequence[float]] | np.ndarray,
        split_thresh: Sequence[float] | np.ndarray,
        *,
        row: int = 0,
        label: str | None = None,
    ) -> Interface:
        """The zero set of one partition gate ``sigmoid(beta (W_row.x - t_row))``.

        The gate is exactly ``1/2`` on this hyperplane and saturates away from
        it, so as ``beta -> inf`` it *is* the seam a
        :class:`~omnibias.pinn.partition.torch.field.PartitionedField` hardens
        towards. Sampling here means the interface residual and the partition
        blend are talking about the same geometry.
        """
        w = np.asarray(split_dirs, dtype=float)
        t = np.asarray(split_thresh, dtype=float).reshape(-1)
        if w.ndim != 2:
            raise ValueError(f"split_dirs must be (depth, D), got shape {w.shape}")
        if t.shape[0] != w.shape[0]:
            raise ValueError(
                f"split_thresh must have one entry per split row: "
                f"expected {w.shape[0]}, got {t.shape[0]}"
            )
        if not 0 <= row < w.shape[0]:
            raise IndexError(f"row {row} out of range for {w.shape[0]} splits")
        name = label if label is not None else f"split{row}"
        return cls(normal=tuple(w[row]), offset=float(t[row]), label=name)

    def _check_ndim(self, x: np.ndarray) -> None:
        if x.ndim != 2:
            raise ValueError(f"points must be 2-D (N, D), got shape {x.shape}")
        if x.shape[1] != self.ndim:
            raise ValueError(
                f"points have D={x.shape[1]} but interface is {self.ndim}-D"
            )


@dataclass(frozen=True)
class InterfaceSpec:
    """An :class:`Interface` plus the material coefficients on either side.

    ``conductivity`` is the ``k`` of the flux ``k dU/dn``: a diffusivity, a
    permeability, a thermal conductivity, whatever the problem calls it. The
    pair is what makes the flux condition non-trivial -- with ``k_plus`` equal
    to ``k_minus`` the flux jump is just a normal-derivative jump.

    ``weights`` scale the value and flux terms when the two are summed into one
    scalar; keeping them here rather than at the call site means a problem with
    several interfaces carries its own balance.
    """

    interface: Interface
    conductivity: tuple[float, float] = (1.0, 1.0)
    weights: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        k = tuple(float(v) for v in self.conductivity)
        w = tuple(float(v) for v in self.weights)
        if len(k) != 2:
            raise ValueError(f"conductivity must be (k_plus, k_minus), got {k}")
        if len(w) != 2:
            raise ValueError(f"weights must be (value, flux), got {w}")
        if any(v < 0.0 for v in w):
            raise ValueError(f"weights must be non-negative, got {w}")
        object.__setattr__(self, "conductivity", k)
        object.__setattr__(self, "weights", w)

    @property
    def label(self) -> str:
        return self.interface.label


def _box(bounds: Sequence[tuple[float, float]] | np.ndarray) -> np.ndarray:
    b = np.asarray(bounds, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"bounds must be ((lo, hi), ...), got shape {b.shape}")
    if np.any(b[:, 1] <= b[:, 0]):
        raise ValueError(f"every bound needs hi > lo, got {b.tolist()}")
    return b


def _inside(x: np.ndarray, box: np.ndarray, shrink: float) -> np.ndarray:
    lo, hi = box[:, 0], box[:, 1]
    pad = shrink * (hi - lo)
    keep: np.ndarray = np.all(
        (x >= lo + pad - 1e-12) & (x <= hi - pad + 1e-12), axis=1
    )
    return keep


def interface_points(
    interface: Interface,
    bounds: Sequence[tuple[float, float]] | np.ndarray,
    *,
    n_points: int,
    method: str = "random",
    seed: int = 0,
    shrink: float = 0.0,
) -> np.ndarray:
    r"""``(n_points, D)`` points lying **on** ``interface`` and inside the box.

    Points are drawn in the interface's own tangent coordinates and mapped back,
    so ``n.x - c`` is zero to round-off rather than to a tolerance. That
    distinction is the entire reason this function exists: an interface residual
    evaluated on points that are merely *near* the seam measures the jump plus
    however much the solution varies over the gap, and no amount of training
    drives that second part to zero.

    Parameters
    ----------
    bounds:
        ``((lo, hi), ...)`` per axis -- the box the interface cuts through.
    method:
        ``"random"`` draws uniformly (rejection against the box);
        ``"grid"`` lays down a deterministic lattice and subsamples it evenly,
        which is reproducible and gives the flat spacing a convergence study
        wants.
    shrink:
        Fraction of each axis to stay clear of the box faces, in ``[0, 0.5)``.
        Useful when the interface runs into a boundary and you would rather the
        interface and boundary conditions not fight over the same points.

    Raises
    ------
    ValueError
        If the interface misses the box entirely, rather than silently
        returning points that are not on it.
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    if not 0.0 <= shrink < 0.5:
        raise ValueError(f"shrink must be in [0, 0.5), got {shrink}")
    box = _box(bounds)
    d = interface.ndim
    if box.shape[0] != d:
        raise ValueError(f"bounds have D={box.shape[0]} but interface is {d}-D")

    center = 0.5 * (box[:, 0] + box[:, 1])
    origin = interface.project(center[None, :])[0]
    if d == 1:
        if not _inside(origin[None, :], box, shrink)[0]:
            raise ValueError(
                f"interface {interface.label!r} at x={origin[0]:g} lies outside "
                f"the box {box.tolist()}"
            )
        return np.repeat(origin[None, :], n_points, axis=0)

    basis = interface.tangent_basis()  # (D, D-1)
    radius = float(np.linalg.norm(box[:, 1] - box[:, 0]))  # covers the box
    accepted = _sample_tangent(
        interface, origin, basis, box, radius, n_points, method, seed, shrink
    )
    # One projection pass removes the round-off the mapping introduced.
    return interface.project(accepted)


def _sample_tangent(
    interface: Interface,
    origin: np.ndarray,
    basis: np.ndarray,
    box: np.ndarray,
    radius: float,
    n_points: int,
    method: str,
    seed: int,
    shrink: float,
) -> np.ndarray:
    """Draw ``n_points`` in-box points parameterised by the tangent basis."""
    k = basis.shape[1]
    if method == "random":
        rng = np.random.default_rng(seed)
        out: list[np.ndarray] = []
        found = 0
        for _ in range(64):
            s = rng.uniform(-0.5 * radius, 0.5 * radius, size=(4 * n_points, k))
            x = origin[None, :] + s @ basis.T
            keep = x[_inside(x, box, shrink)]
            if keep.size:
                out.append(keep)
                found += keep.shape[0]
            if found >= n_points:
                return np.concatenate(out, axis=0)[:n_points]
        raise ValueError(
            f"interface {interface.label!r} barely meets the box {box.tolist()}: "
            f"only {found} of {n_points} sampled points landed inside"
        )
    if method == "grid":
        per_axis = max(2, int(np.ceil((8 * n_points) ** (1.0 / k))))
        axis = np.linspace(-0.5 * radius, 0.5 * radius, per_axis)
        mesh = np.meshgrid(*([axis] * k), indexing="ij")
        s = np.stack([m.reshape(-1) for m in mesh], axis=1)
        x = origin[None, :] + s @ basis.T
        keep = x[_inside(x, box, shrink)]
        if keep.shape[0] < n_points:
            raise ValueError(
                f"interface {interface.label!r} meets the box {box.tolist()} in "
                f"too few lattice points ({keep.shape[0]} < {n_points}); use "
                f"method='random' or ask for fewer points"
            )
        take = np.linspace(0, keep.shape[0] - 1, n_points).round().astype(int)
        return np.asarray(keep[take], dtype=float)
    raise ValueError(f"unknown method {method!r}; use 'random' or 'grid'")


def split_by_interface(
    interface: Interface, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``points`` into ``(plus_side, minus_side)`` about the interface.

    The complement of :func:`interface_points`: it routes ordinary collocation
    to the patch that owns it, so each subfield trains on its own subdomain and
    only the interface residual sees both.
    """
    x = np.asarray(points, dtype=float)
    interface._check_ndim(x)
    plus = interface.signed_distance(x) >= 0.0
    return x[plus], x[~plus]


__all__ = [
    "Interface",
    "InterfaceSpec",
    "interface_points",
    "split_by_interface",
]
