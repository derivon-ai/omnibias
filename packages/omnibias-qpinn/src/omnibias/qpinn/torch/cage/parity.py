# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Hard parity-projection cage (torch backend).

For a Hamiltonian :math:`\hat H` that commutes with reflection through
the chosen *mirror axis* :math:`x_a`, every non-degenerate eigenstate
has a definite parity :math:`\varepsilon \in \{+1, -1\}`. The
parity-projected combination

.. math::

    \psi_\pm(x) = \frac{1}{\sqrt 2}\,\big[\,u(x) \pm u(R_a\,x)\,\big]

isolates the symmetric (+) and antisymmetric (-) sectors. Here
:math:`R_a\,x` denotes the coordinate vector with sign of axis
:math:`a` flipped (only ``axis = mirror_axis`` is reflected, all other
axes pass through). For a 1D problem on :math:`x \in \mathbb R` (the
NH3 inversion-tunneling demo) this collapses to
:math:`R\,x = -x`.

This cage gives the parity-projected wavefunction *and all its
derivatives* in closed form via the chain rule. Each derivative along
the mirror axis flips the sign of the mirror-image evaluation;
derivatives along non-mirror axes pass through unchanged:

.. math::

    \frac{\partial^{n} \psi_\pm}{\partial x_a^{n}}(x)
        \;=\; \frac{1}{\sqrt 2}\,\Big[
            u^{(n,a)}(x) + \varepsilon\,(-1)^{n}\,u^{(n,a)}(R_a x)
        \Big],

with the natural multilinear extension for mixed partials. This
preserves the omnibias closed-form derivative path -- every call into
the cage routes value / derivative / mixed_partial through *two*
inner forward passes (one at ``coords``, one at ``R_a coords``) and
combines them with the parity sign. No autograd Hessian round-trip is
needed.

By construction the cage output is *exactly* in the chosen parity
sector, so the variational Rayleigh quotient cannot find a wavefunction
of the wrong parity (the failure mode that caused 6/15 NH3 v=0
sign-flips in the v0.0.2a1 benchmark batch). It is the hard-constraint
analogue of the *soft* output-level parity projection used by the
v0.0.2a1 NH3 solver.

Usage
-----

.. code-block:: python

    from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
    from omnibias.pinn._core.coords import CoordinateSpec
    from omnibias.pinn._core.components import ComponentSpec
    from omnibias.qpinn.torch.cage.parity import ParityProjectedField

    coord = CoordinateSpec(("Q",))
    spec = ComponentSpec(("psi_re", "psi_im"), groups={"psi": ("psi_re", "psi_im")})
    base = OneLayerVectorField(
        coordinate_spec=coord, components=spec, hidden=64, base="gaussian",
        bias_init="normal",
    )
    cage_even = ParityProjectedField(base=base, parity="even", mirror_axis=0)
    cage_odd  = ParityProjectedField(base=base, parity="odd",  mirror_axis=0)

    coords = torch.linspace(-5, 5, 401, dtype=torch.float64).unsqueeze(-1)
    state_even = cage_even(coords)
    # state_even.ops.value(state_even, "psi_re") is symmetric in Q by construction
    # state_even.ops.laplacian(state_even, "psi_re") is symmetric in Q too
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.cage.incompressible import _CageFieldBase
from omnibias.pinn.torch.fields.base import FieldBase, _import_torch_ops
from torch import Tensor

_INV_SQRT2: float = 0.7071067811865475244008443621048


def _parity_sign(parity: str) -> float:
    p = parity.lower()
    if p in ("+", "even", "symmetric", "+1"):
        return +1.0
    if p in ("-", "odd", "antisymmetric", "-1"):
        return -1.0
    raise ValueError(f"unknown parity {parity!r}; use 'even' or 'odd'")


class ParityProjectedField(_CageFieldBase):
    r"""Hard parity-projection cage for symmetric / antisymmetric eigenstates.

    Parameters
    ----------
    base
        Underlying :class:`FieldBase`. Any field type works (one-layer,
        spectral, chebyshev, ...); the only requirement is that the
        chosen ``projected_names`` are real-valued *channels* of the
        base. Pass-through components (e.g. an unprojected pressure,
        density, or auxiliary potential) are forwarded unchanged.
    parity
        ``"even"`` (symmetric, eps = +1) or ``"odd"`` (antisymmetric,
        eps = -1).
    mirror_axis
        Index of the mirror axis on ``base.coordinate_spec``. Default
        ``0``. For the 1D NH3 tunneling problem the only axis is the
        umbrella coordinate ``Q``, so ``mirror_axis = 0``.
    projected_names
        Names of the base components to parity-project. If ``None``,
        every base component is projected. Names not in this list are
        passed through.
    groups
        Optional :class:`ComponentSpec` groups for the cage. Defaults
        to the base's groups (so e.g. a ``"psi"`` wavefunction group on
        the base is preserved on the cage's spec).

    Notes
    -----
    The cage's :class:`ComponentSpec` is identical to the base's --
    component names and groups are preserved; only the *values and
    derivatives* of the projected components are changed.

    On every call to :meth:`evaluate`, the cage performs **two** base
    forward passes (at ``coords`` and at ``R_a coords``). The cost
    is therefore 2x a single forward pass; in exchange the parity is
    enforced *by construction* and the symmetric / antisymmetric eigenstates
    can be variationally trained without any sign-flip failure mode.

    The mirror reflection only flips ``mirror_axis``; all other axes
    pass through unchanged. For the multi-D case
    (e.g. parity in :math:`x` for a 2D ground state) the chain rule
    handles mixed partials per Leibniz: each derivative along the
    mirror axis contributes a :math:`(-1)` factor on the mirror-image
    evaluation, derivatives along non-mirror axes pass through.

    Examples
    --------
    See :mod:`omnibias.qpinn.torch.cage.parity` for a worked NH3
    example.
    """

    parity_sign_value: float
    mirror_axis: int

    def __init__(
        self,
        *,
        base: FieldBase,
        parity: str,
        mirror_axis: int = 0,
        projected_names: Sequence[str] | None = None,
        groups: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        if not isinstance(base, FieldBase):
            raise TypeError(
                f"base must be a FieldBase, got {type(base).__name__}"
            )
        if mirror_axis < 0 or mirror_axis >= base.coordinate_spec.ndim:
            raise ValueError(
                f"mirror_axis={mirror_axis} out of range for "
                f"coordinate_spec.ndim={base.coordinate_spec.ndim}"
            )
        if projected_names is None:
            projected_names = tuple(base.components.names)
        else:
            projected_names = tuple(projected_names)
            for n in projected_names:
                if not base.components.is_component(n):
                    raise ValueError(
                        f"projected_name {n!r} not in base components "
                        f"{base.components.names!r}"
                    )
        passthrough = tuple(
            n for n in base.components.names if n not in projected_names
        )
        if groups is None:
            groups = {
                name: tuple(members) for name, members in base.components.groups
            }
        super().__init__(
            base=base,
            velocity_names=projected_names,
            passthrough_names=passthrough,
            groups=groups,
        )
        self.parity_sign_value = _parity_sign(parity)
        self.mirror_axis = int(mirror_axis)

    def _mirror_coords(self, coords: Tensor) -> Tensor:
        """Return ``coords`` with the sign of ``mirror_axis`` flipped."""
        out = coords.clone()
        out[..., self.mirror_axis] = -out[..., self.mirror_axis]
        return out

    def evaluate(self, coords: Tensor) -> FieldState[Tensor]:
        if coords.dim() != 2:
            raise ValueError(
                f"coords must be 2D (B, D), got shape {tuple(coords.shape)}"
            )
        if coords.shape[-1] != self.coordinate_spec.ndim:
            raise ValueError(
                f"coords last dim {coords.shape[-1]} != "
                f"coordinate_spec.ndim {self.coordinate_spec.ndim}"
            )
        state_pos = self.base.evaluate(coords)
        state_neg = self.base.evaluate(self._mirror_coords(coords))
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_torch_ops(),
            sigma_cache=state_pos.sigma_cache,
            extra={
                "_parity_state_pos": state_pos,
                "_parity_state_neg": state_neg,
            },
        )

    # ----- dispatched value / derivative / mixed_partial --------------

    def _is_projected(self, name: str) -> bool:
        return name in self.velocity_names

    def _is_passthrough(self, name: str) -> bool:
        return name in self.passthrough_names

    def value_component(self, state: FieldState, name: str) -> Tensor:
        s_pos = state.extra["_parity_state_pos"]
        s_neg = state.extra["_parity_state_neg"]
        v_pos = s_pos.ops.value(s_pos, name)
        if self._is_projected(name):
            v_neg = s_neg.ops.value(s_neg, name)
            return _INV_SQRT2 * (v_pos + self.parity_sign_value * v_neg)
        if self._is_passthrough(name):
            return v_pos
        raise KeyError(
            f"{name!r} is neither a projected component "
            f"{self.velocity_names!r} nor a passthrough "
            f"{self.passthrough_names!r}"
        )

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1,
    ) -> Tensor:
        if order == 0:
            return self.value_component(state, name)
        s_pos = state.extra["_parity_state_pos"]
        s_neg = state.extra["_parity_state_neg"]
        d_pos = s_pos.ops.derivative(s_pos, name, axis=axis, order=order)
        if self._is_projected(name):
            d_neg = s_neg.ops.derivative(s_neg, name, axis=axis, order=order)
            sign_neg = self.parity_sign_value
            if axis == self.mirror_axis and (order % 2) == 1:
                sign_neg = -sign_neg
            return _INV_SQRT2 * (d_pos + sign_neg * d_neg)
        if self._is_passthrough(name):
            return d_pos
        raise KeyError(name)

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        s_pos = state.extra["_parity_state_pos"]
        s_neg = state.extra["_parity_state_neg"]
        d_pos = s_pos.ops.mixed_partial(s_pos, name, axes, orders)
        if self._is_projected(name):
            d_neg = s_neg.ops.mixed_partial(s_neg, name, axes, orders)
            # Each derivative step along the mirror axis flips the sign of
            # the mirror-image evaluation. Non-mirror-axis derivatives
            # pass through unchanged. Compute the total mirror-axis
            # parity from the sum of orders along ``mirror_axis``.
            mirror_order = 0
            for a, o in zip(axes, orders, strict=False):
                if a == self.mirror_axis:
                    mirror_order += int(o)
            sign_neg = self.parity_sign_value
            if (mirror_order % 2) == 1:
                sign_neg = -sign_neg
            return _INV_SQRT2 * (d_pos + sign_neg * d_neg)
        if self._is_passthrough(name):
            return d_pos
        raise KeyError(name)


def make_parity_projected_field(
    *,
    base: FieldBase,
    parity: str,
    mirror_axis: int = 0,
    projected_names: Sequence[str] | None = None,
) -> ParityProjectedField:
    """Functional builder for :class:`ParityProjectedField`."""
    return ParityProjectedField(
        base=base, parity=parity, mirror_axis=mirror_axis,
        projected_names=projected_names,
    )


__all__ = [
    "ParityProjectedField",
    "make_parity_projected_field",
]
