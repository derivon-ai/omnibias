# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Deep and Fourier-feature PINN fields on the exact multivariate jet (torch).

Until now the only trainable free-form field on the omnibias field substrate was
:class:`~omnibias.pinn.torch.fields.OneLayerVectorField` -- a *single* hidden
layer, because the closed-form ``sigma``-tower reduction it uses is written for
one layer. Deep networks lived one package away, in
:mod:`omnibias.torch.architectures`, as bare :class:`torch.nn.Module` objects
that return raw tensors and therefore never reach the field operators.

This module closes that gap. A :class:`_JetFieldBase` subclass owns one of the
architectures from :mod:`omnibias.torch.architectures.pinn` and adapts it to the
:class:`~omnibias.pinn._core.state.FieldState` protocol, so a deep network plugs
straight into ``state.u.grad``, the 111 field operators, the conservation cages
and the prebuilt PDE residuals.

Closed form, not autodiff
-------------------------
Derivatives come from :func:`omnibias.torch.jet_mv.mlp_jet_mv`, the exact
multivariate Faa di Bruno jet: one forward pass yields *every* mixed partial
``D^alpha u(x)`` up to total order ``N`` as ``alpha! c_alpha``. There is no
``torch.autograd.grad`` in the differential operator at any depth or order --
unlike :class:`~omnibias.pinn.torch.partition.PartitionedField`, which must fall
back to the autodiff product rule.

One jet per residual
--------------------
A naive adapter would rebuild the jet on every ``derivative()`` call, which is
slower than autodiff. Instead the jet is memoised in ``FieldState.extra`` (the
per-evaluation op scratch cache) keyed by the order it was computed at, and
every partial is a row lookup into it. Because
:func:`~omnibias.core.multi_index.multi_indices` sorts by total degree, the rows
of an order-``N`` jet are a prefix-compatible superset of an order-``M`` jet for
``M <= N``, so a cached higher-order jet also serves lower-order requests.

``jet_order`` is the planning knob: the jet is built at
``max(requested_order, jet_order)``, so setting it to the highest derivative
order appearing in your residual makes the whole residual cost exactly one jet.
The default ``2`` covers gradient / Hessian / Laplacian residuals.

Value path
----------
:meth:`_JetFieldBase.value_component` uses the plain forward pass rather than
reading row 0 of the jet, so a value-only term (a boundary-condition loss) never
pays for a jet. Both are exact evaluations of the same network and agree to
float64 round-off.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from omnibias.core.multi_index import index_position, multi_index_factorial, multi_indices
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.base import FieldBase
from omnibias.torch.activations.registry import ActivationSpec
from omnibias.torch.architectures.pinn import (
    FourierFeatureMLP,
    JetMLP,
    _JetMLPCore,
    make_siren,
)
from torch import Tensor
from torch.func import vmap

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState

#: ``FieldState.extra`` key under which the per-evaluation jet cache lives.
JET_CACHE_KEY = "_jet_mlp_jets"


def _polylaplacian_terms(n_spatial: int, k: int) -> tuple[tuple[tuple[int, ...], int], ...]:
    r"""Multinomial expansion of ``Delta^k = (sum_i d_i^2)^k``.

    Returns ``(beta, k! / beta!)`` pairs over the spatial axes with ``|beta| = k``,
    so that ``Delta^k f = sum_beta (k! / beta!) D^{2 beta} f``.
    """
    k_fact = math.factorial(k)
    return tuple(
        (beta, k_fact // multi_index_factorial(beta))
        for beta in multi_indices(n_spatial, k)
        if sum(beta) == k
    )


class _JetFieldBase(FieldBase):
    """Adapter turning a jet-capable architecture into an omnibias PINN field.

    Subclasses set :attr:`net` to a :class:`omnibias.torch.architectures.pinn._JetMLPCore`
    (``JetMLP``, ``FourierFeatureMLP``, ...) whose ``in_dim`` matches the coordinate
    count and whose ``out_dim`` matches the component count.
    """

    _omnibias_dispatch = "jet_mlp"

    net: _JetMLPCore

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        net: _JetMLPCore,
        jet_order: int = 2,
    ) -> None:
        super().__init__(coordinate_spec=coordinate_spec, components=components)
        if net.in_dim != coordinate_spec.ndim:
            raise ValueError(
                f"net.in_dim {net.in_dim} != coordinate_spec.ndim {coordinate_spec.ndim}"
            )
        if net.out_dim != components.n_components:
            raise ValueError(
                f"net.out_dim {net.out_dim} != components.n_components "
                f"{components.n_components}"
            )
        if jet_order < 1:
            raise ValueError(f"jet_order must be >= 1, got {jet_order}")
        # Fail loudly at construction rather than deep inside the jet kernel: an
        # activation without a closed-form fast path cannot back this field at all.
        net._check_fastpath(jet_order)
        self.net = net
        self.jet_order = int(jet_order)

    # -- FieldBase plumbing: no single pre-activation tower ---------------------- #

    def _pre_activations(self, coords: Tensor) -> Tensor | None:
        """No single ``z``: a deep net has one pre-activation per layer."""
        return None

    # -- jet machinery ---------------------------------------------------------- #

    def _compute_jet(self, coords: Tensor, order: int) -> Tensor:
        """Batched multivariate jet of shape ``(B, M, C)``.

        Routed through the architecture's ``_point_jet`` hook rather than
        ``mlp_jet_mv`` directly, so a wrapper network -- a band mixture, a
        hard-constraint ansatz -- overrides one method and the field inherits its
        exact derivatives unchanged.
        """

        def one(xi: Tensor) -> Tensor:
            return self.net._point_jet(xi, order)

        out: Tensor = vmap(one)(coords)
        return out

    def _jet_at_least(self, state: FieldState, order: int) -> tuple[Tensor, int]:
        """Return ``(jet, jet_order)`` with ``jet_order >= order``, memoised on the state."""
        cache = cast(
            "dict[int, Tensor]", state.extra.setdefault(JET_CACHE_KEY, {})
        )
        for cached_order in sorted(cache):
            if cached_order >= order:
                return cache[cached_order], cached_order
        want = max(int(order), self.jet_order)
        jet = self._compute_jet(state.coords, want)
        cache[want] = jet
        return jet, want

    def _partial(
        self, state: FieldState, name: str, alpha: tuple[int, ...], order: int
    ) -> Tensor:
        """``D^alpha f_name`` of shape ``(B,)``, read off the cached jet."""
        jet, jet_order = self._jet_at_least(state, order)
        pos = index_position(self.coordinate_spec.ndim, jet_order)
        ci = self.components.index(name)
        return jet[:, pos[alpha], ci] * multi_index_factorial(alpha)

    def _spatial_axis_indices(self) -> tuple[int, ...]:
        return tuple(
            self.coordinate_spec.axis_index(a)
            for a in self.coordinate_spec.spatial_axes
        )

    # -- state-method path consumed by the fields ops dispatch ("jet_mlp") ------- #

    def forward_values(self, coords: Tensor) -> Tensor:
        """All component values ``(B, C)`` from the plain forward pass."""
        return self.net.value(coords)

    def value_component(self, state: FieldState, name: str) -> Tensor:
        ci = self.components.index(name)
        return self.forward_values(state.coords)[:, ci]

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1
    ) -> Tensor:
        r"""``d^order f_name / dx_axis^order`` from the exact jet."""
        if order < 1:
            raise ValueError(f"derivative order must be >= 1, got {order}")
        D = self.coordinate_spec.ndim
        alpha = tuple(order if i == axis else 0 for i in range(D))
        return self._partial(state, name, alpha, order)

    def mixed_partial(
        self, state: FieldState, name: str, axes: tuple[int, ...], orders: tuple[int, ...]
    ) -> Tensor:
        D = self.coordinate_spec.ndim
        acc = [0] * D
        for a, o in zip(axes, orders, strict=False):
            acc[a] += int(o)
        total = sum(acc)
        if total == 0:
            return self.value_component(state, name)
        return self._partial(state, name, tuple(acc), total)

    # -- fast paths: one jet already holds the whole gradient / Hessian ---------- #

    def gradient_full(self, state: FieldState, name: str) -> Tensor:
        """``nabla f_name`` over *all* axes, shape ``(B, D)``."""
        D = self.coordinate_spec.ndim
        cols = [
            self._partial(state, name, tuple(1 if j == i else 0 for j in range(D)), 1)
            for i in range(D)
        ]
        return torch.stack(cols, dim=-1)

    def hessian_full(self, state: FieldState, name: str) -> Tensor:
        """Full Hessian over all axes, shape ``(B, D, D)``."""
        D = self.coordinate_spec.ndim
        rows = []
        for i in range(D):
            row = []
            for j in range(D):
                alpha = tuple(
                    (1 if k == i else 0) + (1 if k == j else 0) for k in range(D)
                )
                row.append(self._partial(state, name, alpha, 2))
            rows.append(torch.stack(row, dim=-1))
        return torch.stack(rows, dim=-2)

    def laplacian(self, state: FieldState, name: str) -> Tensor:
        """``Delta f_name`` over the spatial axes, shape ``(B,)``."""
        return self.polylaplacian(state, name, k=1)

    def polylaplacian(self, state: FieldState, name: str, *, k: int) -> Tensor:
        r"""``Delta^k f_name`` via the multinomial expansion of ``(sum_i d_i^2)^k``.

        Every term is a row of the *same* order-``2k`` jet, so the cost is one jet
        regardless of ``k`` -- the multivariate analogue of the one-layer field's
        ``sigma^{(2k)}`` shortcut.
        """
        if k < 1:
            raise ValueError(f"polylaplacian k must be >= 1, got {k}")
        sa = self._spatial_axis_indices()
        if not sa:
            raise ValueError("polylaplacian requires at least one spatial axis")
        D = self.coordinate_spec.ndim
        out: Tensor | None = None
        for beta, coeff in _polylaplacian_terms(len(sa), k):
            acc = [0] * D
            for axis, b in zip(sa, beta, strict=True):
                acc[axis] = 2 * b
            term = self._partial(state, name, tuple(acc), 2 * k)
            out = coeff * term if out is None else out + coeff * term
        assert out is not None
        return out

    def biharmonic(self, state: FieldState, name: str) -> Tensor:
        """``Delta^2 f_name`` of shape ``(B,)``."""
        return self.polylaplacian(state, name, k=2)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(axes={self.coordinate_spec.axes}, "
            f"components={self.components.names}, net={type(self.net).__name__}, "
            f"jet_order={self.jet_order})"
        )


class JetMLPVectorField(_JetFieldBase):
    r"""Deep multi-output PINN field with exact closed-form input derivatives.

    The deep counterpart of :class:`~omnibias.pinn.torch.fields.OneLayerVectorField`:
    arbitrary depth, arbitrary derivative order, and no ``torch.autograd.grad`` in
    the differential operator. With ``depth=1`` the architecture coincides with the
    one-layer field and reproduces its derivatives.

    Parameters
    ----------
    coordinate_spec, components:
        Input-axis / output-channel metadata, as for every omnibias PINN field.
    hidden:
        Hidden width.
    depth:
        Number of hidden (activated) layers ``>= 1``; the readout is affine.
    base:
        Base activation (name or :class:`ActivationSpec`) with a closed-form
        derivative fast path (``"tanh"``, ``"sigmoid"``, ``"softplus"``, ``"sin"``...).
    jet_order:
        Highest derivative order the residual will request. The jet is built once
        per evaluation at this order, so a residual staying within it costs a single
        jet. Default ``2``.
    net:
        Optional pre-built architecture (used by :func:`make_siren_vector_field`);
        when given, ``hidden`` / ``depth`` / ``base`` are ignored.
    dtype:
        Parameter dtype, defaulting to ``torch.float64`` to match the rest of the
        PINN field surface (the cross-backend parity tests run in double precision).
    """

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        hidden: int = 64,
        depth: int = 3,
        base: str | ActivationSpec[Tensor] = "tanh",
        jet_order: int = 2,
        net: JetMLP | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if net is None:
            net = JetMLP(
                in_dim=coordinate_spec.ndim,
                hidden=hidden,
                out_dim=components.n_components,
                depth=depth,
                base=base,
            )
        net.to(dtype)
        super().__init__(
            coordinate_spec=coordinate_spec,
            components=components,
            net=net,
            jet_order=jet_order,
        )


class FourierFeatureVectorField(_JetFieldBase):
    r"""Spectral-bias-mitigating PINN field with a random Fourier-feature front end.

    The input is lifted by ``gamma(x) = [cos(B x), sin(B x)]`` (Tancik et al. 2020)
    before a deep body acts on it, which lets the network represent high-frequency
    targets -- sharp gradients, boundary layers, shock-like structure -- that a plain
    MLP learns only very slowly.

    The omnibias twist is that the encoding is free in the differential operator:
    because ``cos(z) = sin(z + pi/2)`` the whole map is a single ``sin`` layer, whose
    tower ``sin^{(n)}(z) = sin(z + n pi/2)`` is exact at every order. So the PDE
    residual of a Fourier-feature field is still closed form.

    Passing a *sequence* of ``frequency_scale`` values concatenates several bands
    into one encoding, which is the multi-scale regime: low bands carry the bulk
    solution, high bands the thin-layer detail.

    Parameters
    ----------
    coordinate_spec, components:
        Input-axis / output-channel metadata.
    num_features:
        Fourier features ``F`` *per band*; the encoding width is
        ``2 * F * len(frequency_scale)``.
    hidden, depth:
        Body width and number of hidden (activated) body layers; ``depth=0`` is a
        pure random-Fourier-feature model with a linear readout.
    base:
        Body activation with a closed-form derivative fast path.
    frequency_scale:
        Bandwidth(s) of the Gaussian frequency matrix ``B ~ N(0, (2 pi s)^2)``.
    trainable_features:
        If ``True`` the frequencies are learnable parameters rather than fixed
        buffers (the frequencies then adapt to the solution during training).
    jet_order:
        Highest derivative order the residual will request; see
        :class:`JetMLPVectorField`.
    seed:
        Seed for the frequency-matrix draw.
    dtype:
        Parameter dtype (default ``torch.float64``).
    """

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        num_features: int = 64,
        hidden: int = 64,
        depth: int = 3,
        base: str | ActivationSpec[Tensor] = "tanh",
        frequency_scale: float | Sequence[float] = 1.0,
        trainable_features: bool = False,
        jet_order: int = 2,
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        net = FourierFeatureMLP(
            in_dim=coordinate_spec.ndim,
            num_features=num_features,
            hidden=hidden,
            out_dim=components.n_components,
            depth=depth,
            base=base,
            frequency_scale=frequency_scale,
            trainable_features=trainable_features,
            seed=seed,
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
        """The frequency bands of the encoding."""
        return cast(FourierFeatureMLP, self.net).scales

    @property
    def feature_dim(self) -> int:
        """Width of the Fourier encoding ``2 * F * len(scales)``."""
        return int(cast(FourierFeatureMLP, self.net).feature_dim)


def build_jet_mlp_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    hidden: int = 64,
    depth: int = 3,
    base: str | ActivationSpec[Tensor] = "tanh",
    jet_order: int = 2,
    seed: int | None = 0,
    dtype: torch.dtype = torch.float64,
) -> JetMLPVectorField:
    """Seeded convenience builder for a :class:`JetMLPVectorField`."""
    if seed is not None:
        torch.manual_seed(seed)
    return JetMLPVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        hidden=hidden,
        depth=depth,
        base=base,
        jet_order=jet_order,
        dtype=dtype,
    )


def build_fourier_feature_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    num_features: int = 64,
    hidden: int = 64,
    depth: int = 3,
    base: str | ActivationSpec[Tensor] = "tanh",
    frequency_scale: float | Sequence[float] = 1.0,
    trainable_features: bool = False,
    jet_order: int = 2,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> FourierFeatureVectorField:
    """Seeded convenience builder for a :class:`FourierFeatureVectorField`."""
    torch.manual_seed(seed)
    return FourierFeatureVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        num_features=num_features,
        hidden=hidden,
        depth=depth,
        base=base,
        frequency_scale=frequency_scale,
        trainable_features=trainable_features,
        jet_order=jet_order,
        seed=seed,
        dtype=dtype,
    )


def make_siren_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    hidden: int = 64,
    depth: int = 3,
    omega_0: float = 30.0,
    jet_order: int = 2,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> JetMLPVectorField:
    r"""Build a SIREN (Sitzmann et al. 2020) as an omnibias PINN field.

    A ``sin``-activation network with the SIREN initialisation. Its derivative tower
    is exact at *every* order (``sin^{(n)}(z) = sin(z + n pi/2)``), which makes it a
    natural fit for high-order PDE residuals.
    """
    net = make_siren(
        coordinate_spec.ndim,
        hidden,
        out_dim=components.n_components,
        depth=depth,
        omega_0=omega_0,
        seed=seed,
    )
    return JetMLPVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        jet_order=jet_order,
        net=net,
        dtype=dtype,
    )


__all__ = [
    "FourierFeatureVectorField",
    "JET_CACHE_KEY",
    "JetMLPVectorField",
    "build_fourier_feature_vector_field",
    "build_jet_mlp_vector_field",
    "make_siren_vector_field",
]
