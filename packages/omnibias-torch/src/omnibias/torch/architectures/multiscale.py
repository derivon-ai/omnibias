# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Multi-scale, frequency-aware PINN architectures on the closed-form tower (torch).

:mod:`omnibias.torch.architectures.pinn` cures spectral bias by *lifting the
input* (random Fourier features, SIREN). This module adds the two constructions
that instead put the frequency knob **inside the network**, and keeps both of
them exactly closed form:

* :class:`AdaptiveActivation` -- the Jagtap et al. (2020) adaptive activation
  ``sigma(n a z)`` with a trainable slope ``a``. It is not a bespoke module with
  a hand-written derivative: it is a genuine :class:`ActivationSpec` produced by
  the backend-neutral :func:`omnibias.core.spec.tempered` combinator, so the
  whole tower ``sigma_a^{(k)}(z) = (n a)^k sigma^{(k)}(n a z)`` comes from the
  base activation's tower for free, at every order, on every backend.
* :class:`MscaleMLP` -- the MscaleDNN band mixture
  ``u(x) = sum_j f_j(alpha_j x)`` (Liu, Cai & Xu 2020). Each subnetwork sees the
  input pre-scaled by its own band factor, which converts a high-frequency target
  into a low-frequency one that the subnet can actually learn. Scaling the input
  is the same thing as scaling the first weight matrix, so a band is just a
  :class:`~omnibias.torch.architectures.pinn.JetMLP` with ``alpha_j W_0`` and the
  mixture's jet is the *sum* of the per-band jets -- still one exact
  :func:`~omnibias.torch.jet_mv.mlp_jet_mv` evaluation per band, no autodiff.

Both are :class:`~omnibias.torch.architectures.pinn._JetMLPCore` subclasses, so
they carry the same exact-jet readout (``value`` / ``jet`` / ``gradient`` /
``hessian`` / ``partials``) and drop straight into the omnibias PINN field
substrate via :mod:`omnibias.pinn.torch.fields.multiscale`.

Why the trainable slope stays exact
-----------------------------------
A learnable frequency is usually where closed-form differentiation is abandoned,
because ``a`` moves every step. Here ``a`` enters only as the temperature of
:func:`~omnibias.core.spec.tempered`, whose kernel reads the parameter *at call
time* -- the same live-parameter pattern as
:class:`omnibias.torch.tempered_blocks.TemperedActivation`. So the spec is
rebuilt per forward pass from the current ``a`` and every derivative order stays
analytic while ``a`` trains by ordinary backprop.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core.spec import tempered
from omnibias.torch.activations.registry import ActivationSpec, get_activation
from omnibias.torch.architectures.pinn import JetMLP, _JetMLPCore
from omnibias.torch.jet_mv import mlp_jet_mv

import torch
import torch.nn as nn
from torch import Tensor

LayerSpec = tuple[Tensor, Tensor | None, "ActivationSpec[Tensor] | None"]


def _as_band_scales(scales: Sequence[float]) -> tuple[float, ...]:
    """Normalise a band-scale sequence to a tuple of positive floats."""
    out = tuple(float(s) for s in scales)
    if not out:
        raise ValueError("scales must contain at least one band")
    if any(s <= 0.0 for s in out):
        raise ValueError(f"all band scales must be > 0, got {out}")
    return out


class AdaptiveActivation(nn.Module):
    r"""Trainable-frequency activation ``sigma(n a z)`` as a live :class:`ActivationSpec`.

    The adaptive activation of Jagtap, Kawaguchi & Karniadakis (2020): a scalar
    (or per-neuron) slope ``a`` is trained jointly with the weights, letting each
    layer discover the frequency content it needs instead of inheriting the
    network's low-frequency bias. ``n`` is a *fixed* amplification factor -- the
    effective slope is ``n a``, so the gradient reaching ``a`` is ``n`` times
    larger, which is the whole point of splitting it out.

    :attr:`spec` returns the :class:`ActivationSpec` for the *current* slope, with
    the exact tower

    .. math::

        \frac{d^k}{dz^k}\,\sigma(n a z) = (n a)^k\, \sigma^{(k)}(n a z)

    supplied by :func:`omnibias.core.spec.tempered`. Read it fresh on every
    forward pass (as :meth:`AdaptiveJetMLP._layer_specs` does) so the kernel sees
    the updated parameter.

    Parameters
    ----------
    base:
        Base activation (name or :class:`ActivationSpec`); must carry a
        closed-form fast path.
    slope_scale:
        The fixed factor ``n > 0``. Larger values amplify the gradient w.r.t.
        ``a`` without changing the initial function.
    width:
        ``None`` (default) for one slope per layer (Jagtap's L-LAAF); an integer
        for one slope per hidden unit (N-LAAF), broadcast over the pre-activation.
    scale_init:
        Initial value of the *effective* slope ``n a``: the stored parameter is
        ``scale_init / n``. The default ``1.0`` with the default ``n = 1`` starts
        the layer exactly at the base activation; for other ``n`` the round trip
        through the division is exact only to a floating-point ulp.
    learnable:
        If ``False`` the slope is a buffer rather than a parameter, which turns
        this into a plain fixed-frequency activation.
    dtype:
        Slope dtype, defaulting to the framework default. Worth setting
        explicitly: unlike the weights, ``a`` starts at a *stated* value, and
        creating it narrow and widening later would only preserve that value to
        the narrow precision.
    """

    a: Tensor

    def __init__(
        self,
        base: str | ActivationSpec[Tensor] = "tanh",
        *,
        slope_scale: float = 1.0,
        width: int | None = None,
        scale_init: float = 1.0,
        learnable: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        if spec.fastpath is None:
            raise ValueError(
                f"AdaptiveActivation requires a base activation with a closed-form "
                f"derivative kernel; activation {spec.name!r} has none."
            )
        if slope_scale <= 0.0:
            raise ValueError(f"slope_scale must be > 0, got {slope_scale}")
        if scale_init <= 0.0:
            raise ValueError(f"scale_init must be > 0, got {scale_init}")
        if width is not None and width < 1:
            raise ValueError(f"width must be >= 1 when given, got {width}")
        self.base = spec
        self.slope_scale = float(slope_scale)
        self.width = width
        shape: tuple[int, ...] = () if width is None else (width,)
        a_init = torch.full(
            shape,
            scale_init / self.slope_scale,
            dtype=torch.get_default_dtype() if dtype is None else dtype,
        )
        if learnable:
            self.a = nn.Parameter(a_init)
        else:
            self.register_buffer("a", a_init)

    @property
    def scale(self) -> Tensor:
        """The effective slope ``n a`` at the current parameter value."""
        return self.slope_scale * self.a

    @property
    def spec(self) -> ActivationSpec[Tensor]:
        """Closed-form :class:`ActivationSpec` of ``sigma(n a z)`` at the current ``a``."""
        return tempered(
            self.base,
            self.scale,
            scale="unit",
            name=f"adaptive_{self.base.name}",
            operator_role=(
                f"Adaptive-frequency {self.base.name}: sigma(n a z) with trainable "
                f"slope a; tower (n a)^k sigma^(k)(n a z)."
            ),
        )

    def fastpath(self, z: Tensor, n: int) -> Tensor:
        """``d^n/dz^n sigma(n a z)`` at the current slope."""
        kernel = self.spec.fastpath
        assert kernel is not None, "base fastpath checked at __init__"
        return kernel(z, n)

    def forward(self, z: Tensor) -> Tensor:
        return self.fastpath(z, 0)

    def extra_repr(self) -> str:
        kind = "layer" if self.width is None else f"neuron[{self.width}]"
        return (
            f"base={self.base.name!r}, slope_scale={self.slope_scale}, "
            f"granularity={kind}, learnable={isinstance(self.a, nn.Parameter)}"
        )


class AdaptiveJetMLP(_JetMLPCore):
    r"""Deep MLP with a trainable-frequency activation per hidden layer.

    :class:`~omnibias.torch.architectures.pinn.JetMLP` with every hidden
    activation replaced by an :class:`AdaptiveActivation`. Each layer's slope
    trains alongside the weights, so the network adapts its own frequency content
    rather than relying on a fixed activation -- and because the slope enters
    through :func:`omnibias.core.spec.tempered`, ``D^alpha u(x)`` stays exactly
    closed form at arbitrary order through
    :func:`omnibias.torch.jet_mv.mlp_jet_mv`.

    Parameters
    ----------
    in_dim, hidden, out_dim, depth, base:
        As for :class:`~omnibias.torch.architectures.pinn.JetMLP`.
    slope_scale:
        Fixed amplification ``n`` of the trainable slope (see
        :class:`AdaptiveActivation`).
    granularity:
        ``"layer"`` for one slope per hidden layer (L-LAAF) or ``"neuron"`` for
        one per hidden unit (N-LAAF).
    scale_init:
        Initial effective slope ``n a``; ``1.0`` starts at the plain base
        activation (see :class:`AdaptiveActivation`).
    dtype:
        Slope dtype (see :class:`AdaptiveActivation`); the weights follow the
        framework default and are cast by the caller as usual.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int = 1,
        depth: int = 3,
        base: str | ActivationSpec[Tensor] = "tanh",
        *,
        slope_scale: float = 1.0,
        granularity: str = "layer",
        scale_init: float = 1.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        if depth < 1:
            raise ValueError(f"depth (number of hidden layers) must be >= 1, got {depth}")
        if granularity not in ("layer", "neuron"):
            raise ValueError(
                f"granularity must be 'layer' or 'neuron', got {granularity!r}"
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth
        self.granularity = granularity
        linears: list[nn.Linear] = []
        prev = in_dim
        for _ in range(depth):
            linears.append(nn.Linear(prev, hidden))
            prev = hidden
        linears.append(nn.Linear(prev, out_dim))  # affine readout (no activation)
        self.linears = nn.ModuleList(linears)
        self.activations = nn.ModuleList(
            AdaptiveActivation(
                base,
                slope_scale=slope_scale,
                width=hidden if granularity == "neuron" else None,
                scale_init=scale_init,
                dtype=dtype,
            )
            for _ in range(depth)
        )

    def _layer_specs(self) -> list[LayerSpec]:
        """``(W, b, spec)`` list with each hidden spec rebuilt at the current slope."""
        n = len(self.linears)
        specs: list[LayerSpec] = []
        for i, lin in enumerate(self.linears):
            assert isinstance(lin, nn.Linear)
            if i == n - 1:
                specs.append((lin.weight, lin.bias, None))
                continue
            act = self.activations[i]
            assert isinstance(act, AdaptiveActivation)
            specs.append((lin.weight, lin.bias, act.spec))
        return specs

    def slopes(self) -> tuple[Tensor, ...]:
        """Current effective slope ``n a`` of each hidden layer."""
        out: list[Tensor] = []
        for act in self.activations:
            assert isinstance(act, AdaptiveActivation)
            out.append(act.scale)
        return tuple(out)


class MscaleMLP(_JetMLPCore):
    r"""MscaleDNN band mixture ``u(x) = sum_j f_j(alpha_j x)`` with exact jets.

    The multi-scale DNN of Liu, Cai & Xu (2020). Each subnetwork ``f_j`` sees the
    input pre-scaled by its band factor ``alpha_j``, so a feature that oscillates
    at frequency ``k`` looks like frequency ``k / alpha_j`` to band ``j``: the
    high bands convert the hard, high-frequency part of the target into the easy,
    low-frequency part that a plain MLP learns quickly. Summing the bands
    reassembles the full solution.

    Because ``f_j(alpha_j x)`` is just ``f_j`` with its first weight matrix scaled
    by ``alpha_j``, each band is an ordinary
    :class:`~omnibias.torch.architectures.pinn.JetMLP` chain, and the jet of the
    sum is the sum of the jets. Every mixed partial is therefore still exact and
    closed form -- ``n_bands`` calls to
    :func:`omnibias.torch.jet_mv.mlp_jet_mv`, no ``torch.autograd.grad``.

    Complementary to :class:`~omnibias.torch.architectures.pinn.FourierFeatureMLP`:
    Fourier features widen the *input basis*, an Mscale mixture widens the
    *hypothesis class*, and the two compose.

    Parameters
    ----------
    in_dim, out_dim:
        Input coordinate count and output component count.
    hidden:
        *Total* hidden width, split evenly across the bands (the MscaleDNN
        convention, so the mixture costs about as much as one MLP of this width).
        Each band gets at least one unit.
    depth:
        Number of hidden (activated) layers per band.
    base:
        Band activation with a closed-form derivative fast path.
    scales:
        The band factors ``alpha_j``; the usual choice is a geometric ladder such
        as ``(1, 2, 4, 8)``. ``omnibias.pinn``'s ``suggest_frequency_bands``
        reads them off a measured power spectrum instead of guessing.
    adaptive:
        If ``True`` each band is an :class:`AdaptiveJetMLP`, so the bands also
        tune their own slopes; otherwise a plain :class:`JetMLP`.
    dtype:
        Slope dtype when ``adaptive`` (see :class:`AdaptiveActivation`).
    """

    subnets: nn.ModuleList

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int = 1,
        depth: int = 3,
        base: str | ActivationSpec[Tensor] = "tanh",
        *,
        scales: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        adaptive: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        if depth < 1:
            raise ValueError(f"depth (number of hidden layers) must be >= 1, got {depth}")
        band_scales = _as_band_scales(scales)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth
        self.scales = band_scales
        self.band_hidden = max(1, hidden // len(band_scales))
        subnets: list[_JetMLPCore] = []
        for _ in band_scales:
            if adaptive:
                subnets.append(
                    AdaptiveJetMLP(
                        in_dim,
                        self.band_hidden,
                        out_dim=out_dim,
                        depth=depth,
                        base=base,
                        dtype=dtype,
                    )
                )
            else:
                subnets.append(
                    JetMLP(in_dim, self.band_hidden, out_dim=out_dim, depth=depth, base=base)
                )
        self.subnets = nn.ModuleList(subnets)

    # -- the band mixture is a *parallel* graph, not a single chain -------------- #

    def _layer_specs(self) -> list[LayerSpec]:
        raise NotImplementedError(
            "MscaleMLP is a sum of band subnetworks, not a single layer chain; "
            "use _band_layer_specs()."
        )

    def _band_layer_specs(self) -> list[list[LayerSpec]]:
        r"""One ``(W, b, spec)`` chain per band, with ``W_0`` scaled to ``alpha_j W_0``.

        ``f_j(alpha_j x)`` and ``(alpha_j W_0) x + b_0`` are the same map, so the
        band scale never needs a separate input-scaling node -- it is absorbed into
        the first layer and the chain stays exactly what ``mlp_jet_mv`` consumes.
        """
        groups: list[list[LayerSpec]] = []
        for scale, sub in zip(self.scales, self.subnets, strict=True):
            assert isinstance(sub, _JetMLPCore)
            layers = sub._layer_specs()
            w0, b0, act0 = layers[0]
            groups.append([(w0 * scale, b0, act0), *layers[1:]])
        return groups

    def _check_fastpath(self, max_order: int) -> None:
        for sub in self.subnets:
            assert isinstance(sub, _JetMLPCore)
            sub._check_fastpath(max_order)

    def _point_jet(self, xi: Tensor, order: int) -> Tensor:
        """Single-point jet of the mixture: the sum of the per-band jets."""
        total: Tensor | None = None
        for layers in self._band_layer_specs():
            j = mlp_jet_mv(xi, layers, order)
            total = j if total is None else total + j
        assert total is not None
        return total

    def value(self, x: Tensor) -> Tensor:
        """Plain mixture value ``sum_j f_j(alpha_j x)``, shape ``(..., out_dim)``."""
        total: Tensor | None = None
        for layers in self._band_layer_specs():
            h = x
            for w, b, spec in layers:
                h = h @ w.t()
                if b is not None:
                    h = h + b
                if spec is not None:
                    h = spec.forward(h)
            total = h if total is None else total + h
        assert total is not None
        return total

    def extra_repr(self) -> str:
        return f"scales={self.scales}, band_hidden={self.band_hidden}, depth={self.depth}"


__all__ = [
    "AdaptiveActivation",
    "AdaptiveJetMLP",
    "MscaleMLP",
]
