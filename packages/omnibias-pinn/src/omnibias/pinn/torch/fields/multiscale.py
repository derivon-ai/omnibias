# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Multi-scale PINN fields: adaptive slopes and MscaleDNN band mixtures (torch).

Phase 1 gave the field substrate a deep field type
(:mod:`omnibias.pinn.torch.fields.jet_mlp`) and, with
:class:`~omnibias.pinn.torch.fields.FourierFeatureVectorField`, the *input-lifting*
cure for spectral bias. This module adds the two cures that put the frequency
knob **inside** the network, and both keep the derivative tower exact:

* :class:`AdaptiveJetMLPVectorField` -- a deep field whose activation slope is
  trainable (Jagtap et al. 2020). The slope is not a bolted-on module with a
  hand-written derivative: it is the temperature of
  :func:`omnibias.core.spec.tempered`, so ``sigma_a^{(k)}(z) = (n a)^k
  sigma^{(k)}(n a z)`` comes from the base activation's own tower at every order.
* :class:`MscaleVectorField` -- the MscaleDNN band mixture ``u(x) = sum_j
  f_j(alpha_j x)`` (Liu, Cai & Xu 2020). Each band sees the input pre-scaled, so
  the part of the target that is high-frequency for one band is low-frequency for
  another. The mixture's jet is the sum of the per-band jets, so it is still
  exact closed form.

Both are ordinary ``jet_mlp``-tagged fields: they inherit the hidden-jet cache
(with the live affine readout applied per call), the gradient / Hessian /
polylaplacian fast paths and the whole operator surface from
:class:`~omnibias.pinn.torch.fields.jet_mlp._JetFieldBase`, and need no new
dispatch tag.

Picking the bands
-----------------
Band scales are usually guessed as the ladder ``1, 2, 4, 8, ...``
(:func:`~omnibias.pinn._core.multiscale.geometric_bands`).
:func:`~omnibias.pinn._core.multiscale.suggest_frequency_bands`
measures them instead, from the power spectrum of a sampled solution -- see
:mod:`omnibias.pinn._core.multiscale`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.jet_mlp import _JetFieldBase
from omnibias.torch.activations.registry import ActivationSpec
from omnibias.torch.architectures.multiscale import AdaptiveJetMLP, MscaleMLP
from torch import Tensor


class AdaptiveJetMLPVectorField(_JetFieldBase):
    r"""Deep PINN field with a trainable activation frequency per hidden layer.

    The adaptive-activation field: each hidden layer carries a trainable slope
    ``a`` and computes ``sigma(n a z)``, so the network tunes its own frequency
    content during training instead of being stuck with the fixed spectrum of a
    plain ``tanh`` MLP. ``n`` is a fixed amplification factor that scales the
    gradient reaching ``a`` without changing the initial function.

    This is the cheapest of the multi-scale fields -- one extra scalar per layer --
    and unlike a Fourier encoding it needs no guess about *where* the frequencies
    are. It is also the one that most obviously ought to break closed-form
    differentiation, since the frequency moves every step; it does not, because the
    slope enters as the temperature of :func:`omnibias.core.spec.tempered` and the
    kernel reads it at call time.

    Parameters
    ----------
    coordinate_spec, components:
        Input-axis / output-channel metadata, as for every omnibias PINN field.
    hidden, depth, base, jet_order, dtype:
        As for :class:`~omnibias.pinn.torch.fields.JetMLPVectorField`.
    slope_scale:
        The fixed factor ``n > 0``. Jagtap's recommendation is ``n > 1`` (e.g. 10)
        with the effective slope still starting at 1.
    granularity:
        ``"layer"`` for one slope per hidden layer (L-LAAF) or ``"neuron"`` for one
        per hidden unit (N-LAAF).
    scale_init:
        Initial effective slope ``n a``; ``1.0`` starts exactly at ``base``.
    """

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        hidden: int = 64,
        depth: int = 3,
        base: str | ActivationSpec[Tensor] = "tanh",
        slope_scale: float = 1.0,
        granularity: str = "layer",
        scale_init: float = 1.0,
        jet_order: int = 2,
        net: AdaptiveJetMLP | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if net is None:
            net = AdaptiveJetMLP(
                in_dim=coordinate_spec.ndim,
                hidden=hidden,
                out_dim=components.n_components,
                depth=depth,
                base=base,
                slope_scale=slope_scale,
                granularity=granularity,
                scale_init=scale_init,
                dtype=dtype,
            )
        net.to(dtype)
        super().__init__(
            coordinate_spec=coordinate_spec,
            components=components,
            net=net,
            jet_order=jet_order,
        )

    def slopes(self) -> tuple[Tensor, ...]:
        """Current effective slope ``n a`` of each hidden layer.

        The training diagnostic for this field: slopes drifting well above 1 mean
        the solution carries more high-frequency content than the base activation
        supplies on its own.
        """
        net = self.net
        assert isinstance(net, AdaptiveJetMLP)
        return net.slopes()


class MscaleVectorField(_JetFieldBase):
    r"""MscaleDNN band-mixture PINN field ``u(x) = sum_j f_j(alpha_j x)``.

    A mixture of subnetworks, each seeing the input pre-scaled by its own band
    factor ``alpha_j``. A feature oscillating at frequency ``k`` looks like ``k /
    alpha_j`` to band ``j``, so the high bands turn the hard, high-frequency part
    of the target into the easy, low-frequency part that a plain MLP learns
    quickly; summing the bands reassembles the solution.

    Complementary to :class:`~omnibias.pinn.torch.fields.FourierFeatureVectorField`:
    the Fourier encoding widens the *input basis* with a fixed random draw, an
    Mscale mixture widens the *hypothesis class* with trainable bands. Both are
    ``jet_mlp`` fields, so they mix freely inside one residual.

    Cost note: the mixture evaluates one exact jet per band, so an ``M``-band
    field costs ``M`` jets per residual. The bands are narrower in exchange --
    ``hidden`` is the total width, split evenly -- so the parameter count is
    comparable to a single :class:`~omnibias.pinn.torch.fields.JetMLPVectorField`
    of the same width.

    Parameters
    ----------
    coordinate_spec, components:
        Input-axis / output-channel metadata.
    hidden:
        *Total* hidden width, split evenly across the bands.
    depth, base, jet_order, dtype:
        As for :class:`~omnibias.pinn.torch.fields.JetMLPVectorField`.
    scales:
        The band factors ``alpha_j``. Use
        :func:`~omnibias.pinn._core.multiscale.geometric_bands` for the literature
        ladder or :func:`~omnibias.pinn._core.multiscale.suggest_frequency_bands` to
        read them off a measured spectrum.
    adaptive:
        If ``True`` each band is additionally an :class:`AdaptiveJetMLP`, so the
        bands tune their own slopes on top of their fixed scale.
    """

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        hidden: int = 64,
        depth: int = 3,
        base: str | ActivationSpec[Tensor] = "tanh",
        scales: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        adaptive: bool = False,
        jet_order: int = 2,
        net: MscaleMLP | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if net is None:
            net = MscaleMLP(
                in_dim=coordinate_spec.ndim,
                hidden=hidden,
                out_dim=components.n_components,
                depth=depth,
                base=base,
                scales=scales,
                adaptive=adaptive,
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
    def scales(self) -> tuple[float, ...]:
        """The band factors ``alpha_j`` of the mixture."""
        net = self.net
        assert isinstance(net, MscaleMLP)
        return net.scales

    @property
    def band_hidden(self) -> int:
        """Hidden width of each band subnetwork."""
        net = self.net
        assert isinstance(net, MscaleMLP)
        return int(net.band_hidden)


def build_adaptive_jet_mlp_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    hidden: int = 64,
    depth: int = 3,
    base: str | ActivationSpec[Tensor] = "tanh",
    slope_scale: float = 1.0,
    granularity: str = "layer",
    scale_init: float = 1.0,
    jet_order: int = 2,
    seed: int | None = 0,
    dtype: torch.dtype = torch.float64,
) -> AdaptiveJetMLPVectorField:
    """Seeded convenience builder for an :class:`AdaptiveJetMLPVectorField`."""
    if seed is not None:
        torch.manual_seed(seed)
    return AdaptiveJetMLPVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        hidden=hidden,
        depth=depth,
        base=base,
        slope_scale=slope_scale,
        granularity=granularity,
        scale_init=scale_init,
        jet_order=jet_order,
        dtype=dtype,
    )


def build_mscale_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    hidden: int = 64,
    depth: int = 3,
    base: str | ActivationSpec[Tensor] = "tanh",
    scales: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    adaptive: bool = False,
    jet_order: int = 2,
    seed: int | None = 0,
    dtype: torch.dtype = torch.float64,
) -> MscaleVectorField:
    """Seeded convenience builder for a :class:`MscaleVectorField`."""
    if seed is not None:
        torch.manual_seed(seed)
    return MscaleVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        hidden=hidden,
        depth=depth,
        base=base,
        scales=scales,
        adaptive=adaptive,
        jet_order=jet_order,
        dtype=dtype,
    )


__all__ = [
    "AdaptiveJetMLPVectorField",
    "MscaleVectorField",
    "build_adaptive_jet_mlp_vector_field",
    "build_mscale_vector_field",
]
