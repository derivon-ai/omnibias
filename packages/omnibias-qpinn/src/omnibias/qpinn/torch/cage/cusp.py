# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Hard Kato electron-nucleus cusp cage (torch backend).

Torch twin of :mod:`omnibias.qpinn.jax.cage.cusp`; see that module for the full
mathematical description. The cage multiplies the wavefunction by the
strictly-positive cusp factor

.. math::

   C(r) = \exp\!\Bigl(\sum_a u_a(s_a)\Bigr),
   \qquad
   u_a(s) = -\frac{Z_a\,s}{1 + b_a\,s},
   \qquad
   s_a = \lVert r - R_a\rVert,

whose radial slope ``u_a'(0) = -Z_a`` reproduces the Kato electron-nucleus
cusp by construction. The caged value / gradient / Laplacian are assembled by
the closed-form Leibniz product rule combining the elementary derivatives of
``C`` with the base field's closed-form ``omnibias`` derivatives -- no autodiff.
``b_a = 0`` recovers the exact hydrogenic Slater factor ``exp(-Z_a s)``.
"""

from __future__ import annotations

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.cage.incompressible import _CageFieldBase
from omnibias.pinn.torch.fields.base import FieldBase, _import_torch_ops
from torch import Tensor

_EPS2 = 1e-30


class NuclearCuspField(_CageFieldBase):
    r"""Multiplicative electron-nucleus cusp cage ``psi = C(r) psi_base(r)`` (torch).

    Parameters
    ----------
    base
        Base :class:`FieldBase` carrying a wavefunction group.
    nuclei
        Nuclear positions ``(n_nuc, n_spatial)`` in the base's spatial frame.
    charges
        Nuclear charges ``Z_a`` ``(n_nuc,)``.
    rates
        Pade denominator rates ``b_a``; scalar (broadcast) or ``(n_nuc,)``.
        ``0`` gives the exact hydrogenic Slater factor ``exp(-Z_a s)``.
    psi_group
        Wavefunction group name. Default ``"psi"``.
    """

    nuclei: Tensor
    charges: Tensor
    rates: Tensor

    def __init__(
        self,
        *,
        base: FieldBase,
        nuclei: Tensor,
        charges: Tensor,
        rates: Tensor | float = 0.0,
        psi_group: str = "psi",
    ) -> None:
        if not base.components.is_group(psi_group):
            raise ValueError(
                f"base does not have a wavefunction group {psi_group!r}; "
                "build it with omnibias.qpinn.make_psi_components"
            )
        members = base.components.group_members(psi_group)
        n_spatial = base.coordinate_spec.n_spatial
        nuclei = torch.as_tensor(nuclei)
        charges = torch.as_tensor(charges, dtype=nuclei.dtype)
        if nuclei.dim() != 2 or nuclei.shape[-1] != n_spatial:
            raise ValueError(
                f"nuclei must be (n_nuc, {n_spatial}); got {tuple(nuclei.shape)}"
            )
        if charges.dim() != 1 or charges.shape[0] != nuclei.shape[0]:
            raise ValueError(
                f"charges must be (n_nuc,) = ({nuclei.shape[0]},); "
                f"got {tuple(charges.shape)}"
            )
        if isinstance(rates, int | float):
            rates_t = torch.full(
                (nuclei.shape[0],), float(rates), dtype=nuclei.dtype
            )
        else:
            rates_t = torch.as_tensor(rates, dtype=nuclei.dtype)
        if rates_t.shape != (nuclei.shape[0],):
            raise ValueError(
                f"rates must be scalar or (n_nuc,); got {tuple(rates_t.shape)}"
            )
        passthrough = tuple(n for n in base.components.names if n not in members)
        super().__init__(
            base=base,
            velocity_names=members,
            passthrough_names=passthrough,
            groups={psi_group: members},
        )
        self.psi_group_name = psi_group
        self.spatial_axis_indices = tuple(
            base.coordinate_spec.axis_index(a)
            for a in base.coordinate_spec.spatial_axes
        )
        self.register_buffer("nuclei", nuclei)
        self.register_buffer("charges", charges)
        self.register_buffer("rates", rates_t)

    def _coord_to_spatial(self, axis: int) -> int | None:
        for k, a in enumerate(self.spatial_axis_indices):
            if a == axis:
                return k
        return None

    def _cusp_derivs(self, coords: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""Closed-form ``(C, grad C, Hess C)`` on the spatial coordinates."""
        idx = list(self.spatial_axis_indices)
        r_s = coords[:, idx]  # (B, n_s)
        diff = r_s[:, None, :] - self.nuclei[None, :, :]  # (B, n_nuc, n_s)
        s = torch.sqrt(torch.sum(diff * diff, dim=-1) + _EPS2)  # (B, n_nuc)
        shat = diff / s[..., None]
        z = self.charges[None, :]
        denom = 1.0 + self.rates[None, :] * s
        up = -z / (denom * denom)  # u'(s) -> -Z at s=0
        upp = 2.0 * z * self.rates[None, :] / (denom * denom * denom)  # u''(s)

        grad_g = torch.sum(up[..., None] * shat, dim=1)  # (B, n_s)
        n_s = len(idx)
        eye = torch.eye(n_s, dtype=coords.dtype, device=coords.device)
        outer = shat[..., :, None] * shat[..., None, :]  # (B, n_nuc, n_s, n_s)
        radial = (up / s)[..., None, None] * (eye[None, None, :, :] - outer)
        hess_g = torch.sum(upp[..., None, None] * outer + radial, dim=1)

        g = torch.sum(-z * s / denom, dim=1)  # (B,) log C
        c = torch.exp(g)
        grad_c = c[:, None] * grad_g
        hess_c = c[:, None, None] * (
            grad_g[:, :, None] * grad_g[:, None, :] + hess_g
        )
        return c, grad_c, hess_c

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
        inner_state = self.base.evaluate(coords)
        c, grad_c, hess_c = self._cusp_derivs(coords)
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_torch_ops(),
            sigma_cache=inner_state.sigma_cache,
            extra={
                "_cage_inner_state": inner_state,
                "_cusp_C": c,
                "_cusp_gradC": grad_c,
                "_cusp_hessC": hess_c,
            },
        )

    def value_component(self, state: FieldState, name: str) -> Tensor:
        inner = state.extra["_cage_inner_state"]
        v = inner.ops.value(inner, name)
        if name in self.velocity_names:
            return state.extra["_cusp_C"] * v
        if name in self.passthrough_names:
            return v
        raise KeyError(
            f"{name!r} is neither a caged wavefunction component "
            f"{self.velocity_names!r} nor a passthrough "
            f"{self.passthrough_names!r}"
        )

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1,
    ) -> Tensor:
        if order == 0:
            return self.value_component(state, name)
        inner = state.extra["_cage_inner_state"]
        if name in self.passthrough_names:
            return inner.ops.derivative(inner, name, axis=axis, order=order)
        if name not in self.velocity_names:
            raise KeyError(name)

        c = state.extra["_cusp_C"]
        k = self._coord_to_spatial(axis)
        b0 = inner.ops.value(inner, name)
        b1 = inner.ops.derivative(inner, name, axis=axis, order=1)
        c1 = state.extra["_cusp_gradC"][:, k] if k is not None else 0.0
        if order == 1:
            return c1 * b0 + c * b1
        if order == 2:
            b2 = inner.ops.derivative(inner, name, axis=axis, order=2)
            c2 = state.extra["_cusp_hessC"][:, k, k] if k is not None else 0.0
            return c2 * b0 + 2.0 * c1 * b1 + c * b2
        raise NotImplementedError(
            "NuclearCuspField supplies closed-form derivatives up to order 2 "
            f"(the Schrodinger kinetic operator); got order={order}."
        )

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        inner = state.extra["_cage_inner_state"]
        if name in self.passthrough_names:
            return inner.ops.mixed_partial(inner, name, axes, orders)
        if name not in self.velocity_names:
            raise KeyError(name)

        folded: dict[int, int] = {}
        for a, o in zip(axes, orders, strict=False):
            folded[a] = folded.get(a, 0) + int(o)
        total = sum(folded.values())
        if total == 0:
            return self.value_component(state, name)
        if total == 1:
            (a,) = folded.keys()
            return self.derivative(state, name, axis=a, order=1)
        if total == 2:
            if len(folded) == 1:
                (a,) = folded.keys()
                return self.derivative(state, name, axis=a, order=2)
            (ai, aj) = folded.keys()
            c = state.extra["_cusp_C"]
            grad_c = state.extra["_cusp_gradC"]
            hess_c = state.extra["_cusp_hessC"]
            ki = self._coord_to_spatial(ai)
            kj = self._coord_to_spatial(aj)
            b0 = inner.ops.value(inner, name)
            bi = inner.ops.derivative(inner, name, axis=ai, order=1)
            bj = inner.ops.derivative(inner, name, axis=aj, order=1)
            bij = inner.ops.mixed_partial(inner, name, (ai, aj), (1, 1))
            ci = grad_c[:, ki] if ki is not None else 0.0
            cj = grad_c[:, kj] if kj is not None else 0.0
            cij = hess_c[:, ki, kj] if (ki is not None and kj is not None) else 0.0
            return cij * b0 + ci * bj + cj * bi + c * bij
        raise NotImplementedError(
            "NuclearCuspField supplies closed-form mixed partials up to total "
            f"order 2; got total order {total}."
        )


def make_nuclear_cusp_field(
    *,
    base: FieldBase,
    nuclei: Tensor,
    charges: Tensor,
    rates: Tensor | float = 0.0,
    psi_group: str = "psi",
) -> NuclearCuspField:
    r"""Build a :class:`NuclearCuspField` (torch backend)."""
    return NuclearCuspField(
        base=base,
        nuclei=nuclei,
        charges=charges,
        rates=rates,
        psi_group=psi_group,
    )


def nuclear_cusp_slope(field: NuclearCuspField, atom: int = 0) -> Tensor:
    r"""The realised cusp slope ``u_a'(0) = -Z_a`` for nucleus ``atom``."""
    return -field.charges[atom]


__all__ = [
    "NuclearCuspField",
    "make_nuclear_cusp_field",
    "nuclear_cusp_slope",
]
