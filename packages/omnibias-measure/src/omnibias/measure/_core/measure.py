# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The :class:`Measure` abstraction (pure Python + numpy).

A :class:`Measure` is a *discrete* (atomic / quadrature) measure on
``R^d``: a finite set of ``nodes`` carrying non-negative ``weights`` (the
mass at each node). It generalizes ``omnibias.fields``'s ``QuadratureSpec``
(nodes + weights over a box) to arbitrary supports and arbitrary total mass,
and adds the measure-algebra operations that a measure integral needs:

* :meth:`Measure.pushforward` -- the image measure ``T_# mu`` under a map
  ``T``; ``int g d(T_# mu) = int (g o T) dmu``.
* :meth:`Measure.product` -- the product measure ``mu (x) nu``.
* :meth:`Measure.reweight` -- multiply the weights by a density (the
  Radon-Nikodym derivative ``d nu / d mu``), the change-of-measure /
  importance-sampling primitive.
* :meth:`Measure.normalize` -- rescale to a probability measure.

The constructors reuse ``omnibias.fields``'s battle-tested quadrature rules
(Gauss-Legendre / Gauss-Hermite / Monte-Carlo) rather than reimplementing
them, so a Lebesgue measure on a box, a Gaussian measure, or a Monte-Carlo
measure are one call away. Because nodes and weights are generated once in
numpy ``float64``, the torch and jax twins that consume a :class:`Measure`
produce bit-identical values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from omnibias.fields._core.quadrature import (
    QuadratureSpec,
    gauss_hermite,
    gauss_legendre,
    monte_carlo,
)

#: A numpy density callable ``nodes (n, d) -> (n,)`` used by :meth:`Measure.reweight`.
DensityFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True, eq=False)
class Measure:
    """A finite discrete measure ``sum_i weights_i * delta_{nodes_i}`` on ``R^d``.

    Parameters
    ----------
    nodes
        Float64 array of shape ``(n_nodes, dim)`` -- the atoms / collocation
        points.
    weights
        Float64 array of shape ``(n_nodes,)`` -- the non-negative mass at each
        node. A probability measure has ``sum(weights) == 1``; a Lebesgue box
        quadrature has ``sum(weights) == volume``.
    name
        Human-readable label (e.g. ``"lebesgue"``, ``"gaussian"``).
    support
        Optional ``(lo, hi)`` box per axis, when the measure has bounded
        rectangular support.
    """

    nodes: NDArray[np.float64]
    weights: NDArray[np.float64]
    name: str = "measure"
    support: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.nodes.ndim != 2:
            raise ValueError(
                f"nodes must be 2D (n_nodes, dim), got shape {self.nodes.shape}"
            )
        if self.weights.ndim != 1 or self.weights.shape[0] != self.nodes.shape[0]:
            raise ValueError(
                "weights must be 1D with one entry per node; "
                f"got weights {self.weights.shape}, nodes {self.nodes.shape}"
            )

    @property
    def dim(self) -> int:
        """Ambient dimension of the support."""
        return int(self.nodes.shape[1])

    @property
    def n_nodes(self) -> int:
        """Number of atoms."""
        return int(self.nodes.shape[0])

    @property
    def total_mass(self) -> float:
        """``mu(R^d) = sum_i weights_i``."""
        return float(np.sum(self.weights))

    def pushforward(
        self,
        transform: Callable[[NDArray[np.float64]], ArrayLike],
        *,
        name: str | None = None,
    ) -> Measure:
        r"""Return the image measure ``T_# mu`` under ``transform`` ``T``.

        For any integrable ``g`` on the target space,
        :math:`\int g \, d(T_\# \mu) = \int (g \circ T)\, d\mu`, so the
        pushforward is represented by mapping the nodes through ``T`` and
        keeping the weights. ``T`` may change the dimension.
        """
        new_nodes = np.asarray(transform(self.nodes), dtype=np.float64)
        if new_nodes.ndim == 1:
            new_nodes = new_nodes[:, None]
        if new_nodes.shape[0] != self.n_nodes:
            raise ValueError(
                "transform must map (n_nodes, dim) -> (n_nodes, m); got "
                f"{new_nodes.shape} for n_nodes={self.n_nodes}"
            )
        return Measure(
            new_nodes,
            np.array(self.weights, dtype=np.float64, copy=True),
            name=name if name is not None else f"{self.name}#pushforward",
            support=None,
        )

    def product(self, other: Measure, *, name: str | None = None) -> Measure:
        r"""Return the product measure ``self (x) other`` on ``R^{d1+d2}``.

        Nodes are the Cartesian product of the two node sets (concatenated on
        the coordinate axis) and weights are the outer product, so
        :math:`\int f \, d(\mu \otimes \nu)` integrates ``f`` over the product
        support with the product weights.
        """
        n1, n2 = self.n_nodes, other.n_nodes
        a = np.repeat(self.nodes, n2, axis=0)  # (n1*n2, d1)
        b = np.tile(other.nodes, (n1, 1))  # (n1*n2, d2)
        nodes = np.concatenate([a, b], axis=1)
        weights = (self.weights[:, None] * other.weights[None, :]).reshape(-1)
        support: tuple[tuple[float, float], ...] | None = None
        if self.support is not None and other.support is not None:
            support = tuple(self.support) + tuple(other.support)
        return Measure(
            nodes.astype(np.float64),
            weights.astype(np.float64),
            name=name if name is not None else f"{self.name}(x){other.name}",
            support=support,
        )

    def reweight(
        self,
        density: DensityFn | ArrayLike,
        *,
        name: str | None = None,
    ) -> Measure:
        r"""Return the measure with weights multiplied by ``density``.

        ``density`` is the Radon-Nikodym derivative :math:`d\nu/d\mu` evaluated
        at the nodes (a callable ``nodes -> (n,)`` or a length-``n`` array). The
        result ``nu`` satisfies :math:`\int f \, d\nu = \int f \cdot (d\nu/d\mu)
        \, d\mu`; with a likelihood ratio ``p/q`` this is the importance-sampling
        reweighting of a proposal ``q`` toward a target ``p``.
        """
        if callable(density):
            factor = np.asarray(density(self.nodes), dtype=np.float64)
        else:
            factor = np.asarray(density, dtype=np.float64)
        if factor.shape != (self.n_nodes,):
            raise ValueError(
                f"density must be shape ({self.n_nodes},), got {factor.shape}"
            )
        if np.any(factor < 0.0):
            raise ValueError("reweight density must be non-negative")
        return Measure(
            np.array(self.nodes, dtype=np.float64, copy=True),
            self.weights * factor,
            name=name if name is not None else f"{self.name}#reweight",
            support=self.support,
        )

    def normalize(self, *, name: str | None = None) -> Measure:
        """Return the probability measure ``mu / mu(R^d)`` (weights sum to 1)."""
        mass = self.total_mass
        if not mass > 0.0:
            raise ValueError(f"cannot normalize a measure of mass {mass}")
        return Measure(
            np.array(self.nodes, dtype=np.float64, copy=True),
            self.weights / mass,
            name=name if name is not None else f"{self.name}#normalized",
            support=self.support,
        )

    def __repr__(self) -> str:
        return (
            f"Measure(name={self.name!r}, dim={self.dim}, "
            f"n_nodes={self.n_nodes}, total_mass={self.total_mass:.6g})"
        )


def from_quadrature(spec: QuadratureSpec, *, name: str | None = None) -> Measure:
    """Lift an ``omnibias.fields`` :class:`QuadratureSpec` into a :class:`Measure`."""
    return Measure(
        np.asarray(spec.nodes, dtype=np.float64),
        np.asarray(spec.weights, dtype=np.float64),
        name=name if name is not None else spec.rule,
        support=spec.bounds,
    )


def lebesgue(bounds: object, points_per_axis: int | tuple[int, ...]) -> Measure:
    """Lebesgue measure on a box, discretized by tensor-product Gauss-Legendre.

    The weights sum to the box volume, so ``lebesgue_integral(f, lebesgue(...))``
    approximates ``int_box f dx`` (exact for per-axis polynomials of degree
    ``<= 2 * points_per_axis - 1``).
    """
    return from_quadrature(gauss_legendre(bounds, points_per_axis), name="lebesgue")


def gaussian(
    points_per_axis: int | tuple[int, ...],
    *,
    mean: float | tuple[float, ...] = 0.0,
    scale: float | tuple[float, ...] = 1.0,
    dim: int | None = None,
) -> Measure:
    """Gaussian probability measure ``N(mean, diag(scale**2))`` via Gauss-Hermite.

    Weights sum to 1, so ``lebesgue_integral(f, gaussian(...))`` is the
    expectation ``E_{x ~ N}[f(x)]``.
    """
    return from_quadrature(
        gauss_hermite(points_per_axis, mean=mean, scale=scale, dim=dim),
        name="gaussian",
    )


def uniform_mc(bounds: object, n: int, *, seed: int = 0) -> Measure:
    """Uniform measure on a box, discretized by seeded Monte-Carlo sampling."""
    return from_quadrature(monte_carlo(bounds, n, seed=seed), name="uniform_mc")


def empirical(
    samples: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    name: str = "empirical",
) -> Measure:
    """Empirical measure ``sum_i w_i delta_{x_i}`` from samples.

    ``samples`` is ``(n,)`` or ``(n, d)``. With ``weights=None`` every atom gets
    mass ``1/n`` (the empirical probability measure); otherwise the given
    non-negative weights are used verbatim.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"samples must be (n,) or (n, d), got shape {x.shape}")
    n = x.shape[0]
    if weights is None:
        w = np.full((n,), 1.0 / n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"weights must be shape ({n},), got {w.shape}")
        if np.any(w < 0.0):
            raise ValueError("empirical weights must be non-negative")
    return Measure(x, w, name=name)


def counting(points: ArrayLike, *, name: str = "counting") -> Measure:
    """Counting measure: unit mass at each of the given points.

    For a counting measure ``int f dmu = sum_i f(x_i)`` is a plain sum -- the
    measure-integral generalization that makes "sum" and "integral" the same
    operation with different weights.
    """
    x = np.asarray(points, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return Measure(x, np.ones((x.shape[0],), dtype=np.float64), name=name)


def dirac(point: ArrayLike, *, name: str = "dirac") -> Measure:
    """Dirac point mass ``delta_{point}`` (a single atom of unit mass)."""
    x = np.atleast_1d(np.asarray(point, dtype=np.float64))
    return Measure(x[None, :], np.ones((1,), dtype=np.float64), name=name)


__all__ = [
    "DensityFn",
    "Measure",
    "counting",
    "dirac",
    "empirical",
    "from_quadrature",
    "gaussian",
    "lebesgue",
    "uniform_mc",
]
