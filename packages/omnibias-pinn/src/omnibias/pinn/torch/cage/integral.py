# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Global-integral conservation cage for the torch backend.

Every other cage in omnibias enforces a *pointwise* identity -- ``div u = 0``
holds at each ``x`` because ``u`` is a curl. A conserved *integral* is a
different kind of constraint: it couples the whole domain, so no algebraic
rearrangement at a point can enforce it. The one lever that does is a global
rescaling, and it works whenever the conserved density is **homogeneous**:

.. math::

    I[\lambda \tilde u] = \lambda^p\, I[\tilde u]
    \quad\Longrightarrow\quad
    \lambda = \left(\frac{C}{I[\tilde u]}\right)^{1/p}
    \quad\text{gives}\quad
    I[\lambda \tilde u] = C .

:class:`IntegralConservationField` is that cage, with ``p`` the ``degree``
argument:

* ``degree=1`` -- ``I = int sum_c u_c dx``: total mass, charge, or probability
  of a real density. ``lambda = C / I``.
* ``degree=2`` -- ``I = int sum_c u_c^2 dx``: the squared :math:`L^2` norm.
  ``lambda = sqrt(C / I)``, which is exactly
  :class:`omnibias.qpinn.torch.cage.NormConservationField` -- that cage is the
  ``degree=2``, two-component, wavefunction-shaped special case of this one.

Two things are generalised away from the qpinn cage. The **density** is no
longer hard-wired to :math:`|\psi|^2`, and the **quadrature** is a
:class:`~omnibias.fields.QuadratureSpec` rather than a hand-rolled pair of
arrays, so the same cage works on a Gauss-Legendre box in any dimension, a
Gauss-Hermite weight on an unbounded domain, or a seeded Monte-Carlo sample --
that is the "domain-neutral" part.

Because ``lambda`` is a single scalar with no ``x`` dependence, *every*
derivative is scaled by the same factor, so the closed-form tower survives
intact: ``D^alpha u = lambda D^alpha (tilde u)``. The cage costs one extra base
forward pass (on the quadrature nodes) per evaluation.

Honesty: the constraint holds **to quadrature accuracy**, not to machine
precision. ``I`` is a finite sum, so what is exact is
``sum_q w_q rho(u(x_q)) == C``; the continuum integral differs by the rule's
own error. That is a real limitation and it is the same one
``NormConservationField`` has -- use a rule that resolves the field, and check
with :func:`~omnibias.fields.torch.ops.integrate` on a finer rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.cage.incompressible import _CageFieldBase
from omnibias.pinn.torch.fields.base import FieldBase, _import_torch_ops
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.quadrature import QuadratureSpec


class IntegralConservationField(_CageFieldBase):
    r"""Hard conservation of a homogeneous integral, by global rescaling.

    Exposes the same component names as ``base``; the conserved components come
    out multiplied by the scalar ``lambda = (total / I)^(1/degree)`` and the
    rest pass through untouched.

    Parameters
    ----------
    base:
        Field carrying the conserved components (and any passthrough ones).
    rule:
        Quadrature the integral is evaluated on. Its ``dim`` must match the
        coordinate spec, and its nodes are held fixed for the whole run.
    conserved:
        Component names entering the density. Their sum of ``degree``-th powers
        is the integrand.
    total:
        The conserved value ``C``. Must be positive.
    degree:
        Homogeneity ``p`` of the density. ``1`` for a mass-like linear density,
        ``2`` for an energy- or norm-like quadratic one.
    dtype:
        Dtype for the quadrature buffers; must match the base field's
        parameters. Defaults to the framework default dtype.

    Raises
    ------
    ValueError
        If the measured integral admits no real rescaling: zero at any degree
        (the conserved components vanish on the nodes), or negative at an even
        degree. The cage refuses rather than returning a silent ``nan``. A
        negative integral at an *odd* degree is fine -- the root keeps the sign.
    """

    quadrature_nodes: Tensor
    quadrature_weights: Tensor

    def __init__(
        self,
        *,
        base: FieldBase,
        rule: QuadratureSpec,
        conserved: tuple[str, ...],
        total: float = 1.0,
        degree: int = 2,
        dtype: torch.dtype | None = None,
    ) -> None:
        if not conserved:
            raise ValueError("conserved must name at least one component")
        for name in conserved:
            if not base.components.is_component(name):
                raise ValueError(
                    f"conserved component {name!r} not in base components "
                    f"{base.components.names!r}"
                )
        if len(set(conserved)) != len(conserved):
            raise ValueError(f"conserved names must be unique, got {conserved!r}")
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        if total <= 0.0:
            raise ValueError(f"total must be > 0, got {total}")
        if rule.dim != base.coordinate_spec.ndim:
            raise ValueError(
                f"quadrature dim {rule.dim} != coordinate_spec.ndim "
                f"{base.coordinate_spec.ndim}"
            )

        passthrough = tuple(n for n in base.components.names if n not in conserved)
        super().__init__(
            base=base,
            velocity_names=tuple(conserved),
            passthrough_names=passthrough,
            groups={"conserved": tuple(conserved)},
        )
        self.total = float(total)
        self.degree = int(degree)
        self.rule = rule
        dt = torch.get_default_dtype() if dtype is None else dtype
        self.register_buffer(
            "quadrature_nodes", torch.as_tensor(rule.nodes, dtype=dt)
        )
        self.register_buffer(
            "quadrature_weights", torch.as_tensor(rule.weights, dtype=dt)
        )

    def integral(self, state: FieldState) -> Tensor:
        """The conserved integral of the *caged* field: ``C``, to quadrature error.

        Recomputed from the cage's own output rather than read off ``lambda``,
        so it is a genuine check of the constraint and not a restatement of it.
        """
        scale = state.extra["_conservation_scale"]
        return self._raw_integral() * scale**self.degree

    def _scale_from(self, raw: Tensor) -> Tensor:
        r"""``lambda`` solving ``lambda^degree * raw == total``.

        Real for a negative ``raw`` only when ``degree`` is odd, where the root
        keeps the sign: ``lambda = sign(r) |r|^(1/p)`` for ``r = total / raw``.
        An even degree squares any sign away, so a non-positive ``raw`` there
        has no real solution and is an error rather than a ``nan``.
        """
        value = float(raw.detach())
        if value == 0.0 or (value < 0.0 and self.degree % 2 == 0):
            raise ValueError(
                f"measured integral {value:.6g} admits no real rescaling to "
                f"total={self.total} at degree={self.degree}. Zero means the "
                f"conserved components vanish on the quadrature nodes; negative "
                f"at even degree cannot happen with non-negative weights. "
                f"Re-initialise the base field, or map its output through a "
                f"positive activation."
            )
        ratio = self.total / raw
        return torch.sign(ratio) * torch.abs(ratio) ** (1.0 / self.degree)

    def _raw_integral(self) -> Tensor:
        """``sum_q w_q sum_c u_c(x_q)^degree`` on the *un*-scaled base field."""
        quad_state = self.base.evaluate(self.quadrature_nodes)
        density = None
        for name in self.velocity_names:
            term = quad_state.ops.value(quad_state, name) ** self.degree
            density = term if density is None else density + term
        assert density is not None  # velocity_names is non-empty by construction
        return (self.quadrature_weights * density).sum()

    def evaluate(self, coords: Tensor) -> FieldState[Tensor]:
        if coords.dim() != 2:
            raise ValueError(
                f"coords must be 2D (B, D), got shape {tuple(coords.shape)}"
            )
        if coords.shape[-1] != self.coordinate_spec.ndim:
            raise ValueError(
                f"coords last dim {coords.shape[-1]} != coordinate_spec.ndim "
                f"{self.coordinate_spec.ndim}"
            )
        raw = self._raw_integral()
        scale = self._scale_from(raw)
        inner_state = self.base.evaluate(coords)
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_torch_ops(),
            sigma_cache=inner_state.sigma_cache,
            extra={"_cage_inner_state": inner_state, "_conservation_scale": scale},
        )

    # ----- dispatched value / derivative / mixed_partial ------------------
    # lambda has no x dependence, so all three are the base's answer times it.

    def value_component(self, state: FieldState, name: str) -> Tensor:
        inner = state.extra["_cage_inner_state"]
        v = inner.ops.value(inner, name)
        return self._scaled(state, name, v)

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1
    ) -> Tensor:
        if order == 0:
            return self.value_component(state, name)
        inner = state.extra["_cage_inner_state"]
        d = inner.ops.derivative(inner, name, axis=axis, order=order)
        return self._scaled(state, name, d)

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        inner = state.extra["_cage_inner_state"]
        d = inner.ops.mixed_partial(inner, name, axes, orders)
        return self._scaled(state, name, d)

    def _scaled(self, state: FieldState, name: str, value: Tensor) -> Tensor:
        if name in self.velocity_names:
            scale: Tensor = state.extra["_conservation_scale"]
            return value * scale
        if name in self.passthrough_names:
            return value
        raise KeyError(
            f"{name!r} is neither a conserved component {self.velocity_names!r} "
            f"nor a passthrough {self.passthrough_names!r}"
        )


def make_integral_conservation_field(
    *,
    base: FieldBase,
    rule: QuadratureSpec,
    conserved: tuple[str, ...],
    total: float = 1.0,
    degree: int = 2,
    dtype: torch.dtype | None = None,
) -> IntegralConservationField:
    """Build an :class:`IntegralConservationField`; see it for the arguments."""
    return IntegralConservationField(
        base=base,
        rule=rule,
        conserved=conserved,
        total=total,
        degree=degree,
        dtype=dtype,
    )


__all__ = [
    "IntegralConservationField",
    "make_integral_conservation_field",
]
