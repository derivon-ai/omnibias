# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Heterogeneous multi-pack Birkhoff unit (torch; theory 01-01).

Closed-form evaluation of

    F(z) = sum_g c_g * sigma^(n_g)(z + mu_g)

via the activation fastpath. With ``share_means=True`` (default), packs that
share a mean in the :class:`~omnibias.core.multipack.MultiPackSpec` share one
mean Parameter -- one activation call per distinct mean. Founding bias
collapse ``delta -> 0`` only -- no temperature collapse.
"""

from __future__ import annotations

from omnibias.core.multipack import MultiPackSpec, PackSpec
from omnibias.torch.activations.registry import ActivationSpec, get_activation

import torch
import torch.nn as nn
from torch import Tensor


def multipack_response(
    z: Tensor,
    means: Tensor,
    weights: Tensor,
    orders: tuple[int, ...],
    spec: ActivationSpec[Tensor],
    *,
    mean_index: tuple[int, ...] | None = None,
) -> Tensor:
    """Evaluate ``sum_g c_g sigma^(n_g)(z + mu_g)`` with a closed-form tower.

    ``orders`` must be a Python tuple of static ints. When ``mean_index`` is
    given, pack ``g`` reads ``means[mean_index[g]]`` (shared-mean layout);
    otherwise ``means`` has length ``G`` and pack ``g`` reads ``means[g]``.
    """
    if spec.fastpath is None:
        raise NotImplementedError(
            f"Activation {spec.name!r} has no closed-form derivative kernel"
        )
    G = len(orders)
    if int(weights.shape[-1]) != G:
        raise ValueError("weights length must equal number of packs")
    if mean_index is None:
        if int(means.shape[-1]) != G:
            raise ValueError("means length must equal number of packs")
        index = tuple(range(G))
    else:
        if len(mean_index) != G:
            raise ValueError("mean_index length must equal number of packs")
        index = mean_index
    for n in orders:
        if n < 0:
            raise ValueError(f"order must be >= 0, got {n}")

    fp = spec.fastpath
    # One fastpath call per distinct mean slot, then fan out orders.
    slots = sorted(set(index))
    slot_u: dict[int, Tensor] = {s: z + means[s] for s in slots}
    out: Tensor | None = None
    for g, n in enumerate(orders):
        term = weights[g] * fp(slot_u[index[g]], n)
        out = term if out is None else out + term
    assert out is not None
    return out


class MultiPackUnit(nn.Module):
    """Channel-wise multi-pack Birkhoff unit.

    Parameters
    ----------
    num_channels:
        Number of independent channels; ``z`` has shape ``(..., num_channels)``.
    spec:
        :class:`~omnibias.core.multipack.MultiPackSpec` describing orders,
        initial means, and initial outer weights.
    base:
        Activation name or :class:`~omnibias.torch.activations.ActivationSpec`.
    learnable_means / learnable_weights:
        Whether means / outer weights are ``nn.Parameter`` or frozen buffers.
    share_means:
        If true (default), packs with equal means in ``spec`` share one mean
        Parameter (one activation evaluation per distinct mean).
    dtype:
        Parameter dtype; ``None`` resolves to ``torch.get_default_dtype()``.
    """

    act_spec: ActivationSpec[Tensor]
    means: Tensor
    weights: Tensor

    def __init__(
        self,
        num_channels: int,
        spec: MultiPackSpec,
        *,
        base: str | ActivationSpec[Tensor] = "sigmoid",
        learnable_means: bool = True,
        learnable_weights: bool = True,
        share_means: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        self.num_channels = int(num_channels)
        self.pack_spec = spec
        self.orders: tuple[int, ...] = tuple(int(p.order) for p in spec.packs)
        self.share_means = bool(share_means)
        self.act_spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        if self.act_spec.fastpath is None:
            raise NotImplementedError(
                f"Activation {self.act_spec.name!r} has no closed-form derivative kernel"
            )
        probe = torch.zeros(1, dtype=torch.get_default_dtype())
        for n in self.orders:
            try:
                self.act_spec.fastpath(probe, n)
            except NotImplementedError:
                raise
            except Exception as exc:  # pragma: no cover
                raise NotImplementedError(
                    f"Activation {self.act_spec.name!r} cannot reach order {n}"
                ) from exc

        dt = torch.get_default_dtype() if dtype is None else dtype
        if share_means:
            distinct = spec.distinct_means
            mean_to_slot = {m: i for i, m in enumerate(distinct)}
            self.mean_index: tuple[int, ...] = tuple(
                mean_to_slot[p.mean] for p in spec.packs
            )
            means0 = torch.tensor(list(distinct), dtype=dt)
        else:
            self.mean_index = tuple(range(len(spec.packs)))
            means0 = torch.tensor([p.mean for p in spec.packs], dtype=dt)
        weights0 = torch.tensor([p.weight for p in spec.packs], dtype=dt)
        if learnable_means:
            self.means = nn.Parameter(means0)
        else:
            self.register_buffer("means", means0)
        if learnable_weights:
            self.weights = nn.Parameter(weights0)
        else:
            self.register_buffer("weights", weights0)

    def forward(self, z: Tensor) -> Tensor:
        """``z`` shape ``(..., num_channels)`` -> same shape."""
        if z.shape[-1] != self.num_channels:
            raise ValueError(
                f"expected z[..., {self.num_channels}], got shape {tuple(z.shape)}"
            )
        return multipack_response(
            z,
            self.means,
            self.weights,
            self.orders,
            self.act_spec,
            mean_index=self.mean_index,
        )


BirkhoffOMBU = MultiPackUnit

__all__ = [
    "BirkhoffOMBU",
    "MultiPackSpec",
    "MultiPackUnit",
    "PackSpec",
    "multipack_response",
]
