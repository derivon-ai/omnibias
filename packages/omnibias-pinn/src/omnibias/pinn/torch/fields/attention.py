# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Non-local attention PINN field with closed-form coordinate derivatives (torch).

Every other field on the substrate is *local*: ``u(x)`` is a chain of elementwise
activations applied to affine maps of ``x``, so the value at a point never sees
the rest of the domain except through the shared weights.
:class:`AttentionVectorField` breaks that. It routes the coordinates through a
softmax mixture over a trainable memory,

.. math::

    u(x) = W_o\,\Big[\mathrm{softmax}\big(\beta\,q(x) K^\top\big) V + q(x)\Big]
    + b_o,

so every output couples to every memory slot through the shared softmax
denominator -- a learned, differentiable soft partition of the domain with one
local model ``V_j`` per region.

Why this is the closed-form story and not a bolted-on module
------------------------------------------------------------
:mod:`omnibias.hopfield` already differentiates this block's log-sum-exp core in
closed form, but with respect to the **scores**. A PDE residual needs ``d/dx``,
which that Jacobian does not give.
:func:`omnibias.torch.jet_mv.jet_attention` supplies exactly the missing piece by
pushing the *query jet* through the block: softmax factors into ``exp`` (a
registered activation), a linear sum, a reciprocal (tower ``(-1)^k k!
u^{-(k+1)}``) and a jet product, each of which is exact. So ``D^alpha u(x)`` is
closed form at arbitrary order and there is no ``torch.autograd.grad`` in the
differential operator -- the same guarantee the local fields carry.

The field is an ordinary ``jet_mlp``-tagged field, inheriting the jet cache, the
gradient / Hessian / polylaplacian fast paths and the whole operator surface from
:class:`~omnibias.pinn.torch.fields.jet_mlp._JetFieldBase`.

Reading the attention
---------------------
:meth:`AttentionVectorField.attention_weights` returns the per-point partition of
unity over the memory, which is the diagnostic that makes the field interpretable:
it says which slots the model consults at ``x``. As ``beta -> inf`` the mixture
hardens into a crisp assignment -- the *feasibility* sense of collapse used
throughout the repo, not the founding ``delta -> 0`` one.
"""

from __future__ import annotations

import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.jet_mlp import _JetFieldBase
from omnibias.torch.activations.registry import ActivationSpec
from omnibias.torch.architectures.attention import AttentionJetMLP
from torch import Tensor


class AttentionVectorField(_JetFieldBase):
    r"""PINN field whose value at ``x`` is a non-local mixture over a trainable memory.

    Parameters
    ----------
    coordinate_spec, components:
        Input-axis / output-channel metadata, as for every omnibias PINN field.
    hidden, depth, base, jet_order, dtype:
        As for :class:`~omnibias.pinn.torch.fields.JetMLPVectorField`; ``hidden``
        is also the query / key width.
    memory:
        Number of memory slots. Think of it as the number of regions the field may
        specialise to; it is *soft*, so slots overlap and the count is only an
        upper bound on the partition's resolution.
    value_dim:
        Width of the value slots, defaulting to ``hidden``. Must equal ``hidden``
        when ``residual`` is set.
    beta:
        Initial inverse temperature; larger is a sharper partition.
    learnable_temperature:
        Train ``beta`` with the weights. It enters the jet as a scale on the keys,
        so derivatives stay closed form while it moves.
    residual:
        Add the query to the attention output (the transformer skip). Keeps a
        local path alongside the non-local one; on by default because a pure
        attention readout is confined to the convex hull of the value slots.
    net:
        Optional pre-built architecture; when given the shape arguments are ignored.
    """

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        hidden: int = 64,
        depth: int = 2,
        base: str | ActivationSpec[Tensor] = "tanh",
        memory: int = 16,
        value_dim: int | None = None,
        beta: float = 1.0,
        learnable_temperature: bool = False,
        residual: bool = True,
        jet_order: int = 2,
        net: AttentionJetMLP | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if net is None:
            net = AttentionJetMLP(
                coordinate_spec.ndim,
                hidden,
                out_dim=components.n_components,
                depth=depth,
                base=base,
                memory=memory,
                value_dim=value_dim,
                beta=beta,
                learnable_temperature=learnable_temperature,
                residual=residual,
                dtype=dtype,
            )
        net.to(dtype)
        super().__init__(
            coordinate_spec=coordinate_spec,
            components=components,
            net=net,
            jet_order=jet_order,
        )

    @property
    def memory(self) -> int:
        """Number of memory slots."""
        net = self.net
        assert isinstance(net, AttentionJetMLP)
        return int(net.memory)

    @property
    def beta(self) -> Tensor:
        """Current inverse temperature of the softmax mixture."""
        net = self.net
        assert isinstance(net, AttentionJetMLP)
        return net.beta

    def attention_weights(self, coords: Tensor) -> Tensor:
        """Per-point partition of unity over the memory, shape ``(B, memory)``."""
        net = self.net
        assert isinstance(net, AttentionJetMLP)
        return net.attention_weights(coords)


def build_attention_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    hidden: int = 64,
    depth: int = 2,
    base: str | ActivationSpec[Tensor] = "tanh",
    memory: int = 16,
    value_dim: int | None = None,
    beta: float = 1.0,
    learnable_temperature: bool = False,
    residual: bool = True,
    jet_order: int = 2,
    seed: int | None = 0,
    dtype: torch.dtype = torch.float64,
) -> AttentionVectorField:
    """Seeded convenience builder for an :class:`AttentionVectorField`."""
    if seed is not None:
        torch.manual_seed(seed)
    return AttentionVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        hidden=hidden,
        depth=depth,
        base=base,
        memory=memory,
        value_dim=value_dim,
        beta=beta,
        learnable_temperature=learnable_temperature,
        residual=residual,
        jet_order=jet_order,
        dtype=dtype,
    )


__all__ = [
    "AttentionVectorField",
    "build_attention_vector_field",
]
