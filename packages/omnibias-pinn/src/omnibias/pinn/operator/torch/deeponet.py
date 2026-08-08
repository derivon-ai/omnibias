# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""DeepONet with closed-form query-coordinate derivatives (torch).

A DeepONet is ``G(u)(y) = b_0 + sum_k b_k(u) t_k(y)``. Because the map is linear
in the trunk basis ``t_k``, every mixed partial in the query coordinate is

    d^alpha G(u)(y) = sum_k b_k(u) d^alpha t_k(y)   (|alpha| >= 1)

and the trunk is an omnibias :class:`~omnibias.torch.architectures.pinn.JetMLP`,
so one multivariate trunk jet yields every ``d^alpha t_k`` up to a chosen order.
The branch coefficients are a *per-sample* live readout -- structurally the same
split as the readout-independence invariant on
:class:`~omnibias.pinn.torch.fields.jet_mlp._JetFieldBase`, just with a batch
axis on the readout.

The field subclasses :class:`~omnibias.pinn.torch.fields.jet_mlp._JetFieldBase`
and inherits ``_omnibias_dispatch = "jet_mlp"``, so every existing field op
(``derivative``, ``laplacian``, ``hessian``, ``polylaplacian``, ...) and every
cage works without an ops edit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.state import FieldState
from omnibias.pinn.operator._core.branch import BranchHeadLayout
from omnibias.pinn.operator._core.conditioning import ConditioningSpec
from omnibias.pinn.operator._core.spec import OperatorSpec
from omnibias.pinn.torch.fields.jet_mlp import _JetFieldBase
from omnibias.torch.activations.registry import ActivationSpec, get_activation
from omnibias.torch.architectures.pinn import JetMLP, _JetMLPCore
from torch import Tensor
from torch.func import vmap

if TYPE_CHECKING:  # pragma: no cover
    pass

#: ``FieldState.extra`` key for the per-evaluation trunk-jet cache.
#: Distinct from ``JET_CACHE_KEY`` so a DeepONet field never collides with a
#: plain jet-MLP field that happens to share a state dict.
TRUNK_JET_CACHE_KEY = "_deeponet_trunk_jets"


class PerSampleReadoutError(TypeError):
    """Raised when a shared affine readout is asked of a DeepONet core.

    DeepONet branch coefficients are per-sample; the shared
    :meth:`_JetMLPCore._apply_readout_jet` contract does not apply. Callers must
    go through :meth:`DeepONetField._jet_at_least`, which contracts the trunk
    jet with the live per-sample coefficients.
    """


class _DeepONetCore(_JetMLPCore):
    """Trunk jet adapter that satisfies the ``_JetMLPCore`` protocol.

    ``_layer_specs`` / ``_check_fastpath`` / ``_point_hidden_jet`` forward to the
    trunk so the closed-form tower gate still fires at construction.
    ``_apply_readout_jet`` is intentionally unusable: the readout is per-sample.
    """

    def __init__(self, trunk: JetMLP, n_components: int) -> None:
        super().__init__()
        if trunk.out_dim < 1:
            raise ValueError(f"trunk.out_dim must be >= 1, got {trunk.out_dim}")
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.trunk = trunk
        self.in_dim = trunk.in_dim
        self.out_dim = int(n_components)
        # Register trunk as a submodule so its parameters train with the field.
        self.add_module("_trunk", trunk)

    def _layer_specs(
        self,
    ) -> list[tuple[Tensor, Tensor | None, ActivationSpec[Tensor] | None]]:
        return self.trunk._layer_specs()

    def _point_hidden_jet(self, xi: Tensor, order: int) -> Tensor:
        """Full trunk jet of shape ``(M, p)`` -- the trunk basis and all its partials.

        Unlike a plain jet MLP, the "hidden" quantity here *includes* the trunk's
        own affine readout to width ``p``: that is the basis ``t_k(y)`` whose
        derivatives the DeepONet identity differentiates through.
        """
        return self.trunk._point_jet(xi, order)

    def _apply_readout_jet(self, hidden_jet: Tensor) -> Tensor:
        raise PerSampleReadoutError(
            "DeepONet uses a per-sample branch readout; call "
            "DeepONetField._jet_at_least instead of _apply_readout_jet."
        )

    def value(self, x: Tensor) -> Tensor:
        raise PerSampleReadoutError(
            "DeepONetCore.value has no shared readout; use "
            "DeepONetField.forward_values with live branch coefficients."
        )


class _HeadEncoder(nn.Module):
    """Independently normalised conditioning head -> fixed-width encoding."""

    def __init__(
        self,
        n_input: int,
        encoder_dim: int,
        *,
        hidden: int = 64,
        depth: int = 1,
        base: str | ActivationSpec[Tensor] = "tanh",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if n_input < 1:
            raise ValueError(f"n_input must be >= 1, got {n_input}")
        if depth < 1:
            raise ValueError(f"encoder depth must be >= 1, got {depth}")
        self.n_input = int(n_input)
        self.encoder_dim = int(encoder_dim)
        self.spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        # LayerNorm over a width-1 feature maps every scalar to zero
        # (mean = itself, variance = 0). Skip it for single-parameter heads.
        self.norm: nn.Module
        if n_input == 1:
            self.norm = nn.Identity()
        else:
            self.norm = nn.LayerNorm(n_input, dtype=dtype)
        linears: list[nn.Linear] = []
        prev = n_input
        for i in range(depth):
            out = encoder_dim if i == depth - 1 else hidden
            linears.append(nn.Linear(prev, out, dtype=dtype))
            prev = out
        self.linears = nn.ModuleList(linears)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.n_input:
            raise ValueError(
                f"head input trailing dim must be {self.n_input}, "
                f"got {tuple(x.shape)}"
            )
        h = self.norm(x)
        for i, lin in enumerate(self.linears):
            h = lin(h)
            if i < len(self.linears) - 1:
                h = self.spec.forward(h)
        return h  # type: ignore[no-any-return]


class _BranchNet(nn.Module):
    """Multi-head encoders + fusion network -> ``(coeffs, bias)``.

    Each active head in :class:`~omnibias.pinn.operator.ConditioningSpec` is
    layer-normalised and encoded independently; a fusion MLP maps the
    concatenated encodings to branch coefficients. The function-only path is
    function-encoder -> fusion (no raw concatenation).
    """

    def __init__(
        self,
        conditioning: ConditioningSpec,
        n_components: int,
        trunk_width: int,
        *,
        hidden: int = 64,
        depth: int = 2,
        encoder_dim: int | None = None,
        encoder_depth: int = 1,
        base: str | ActivationSpec[Tensor] = "tanh",
        per_sample_bias: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"branch fusion depth must be >= 1, got {depth}")
        enc_dim = int(encoder_dim if encoder_dim is not None else hidden)
        self.layout = BranchHeadLayout(conditioning, enc_dim)
        self.n_components = int(n_components)
        self.trunk_width = int(trunk_width)
        self.per_sample_bias = bool(per_sample_bias)
        self.spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        # Backward-compatible aliases used by parity tests / callers.
        self.n_input = int(conditioning.total_dim)
        self.n_sensors = int(conditioning.total_dim)
        encoders = nn.ModuleDict()
        _KEY = {
            "function": "function",
            "parameters": "pde_params",
            "boundary": "boundary",
            "geometry": "geometry",
        }
        for name, width in self.layout.head_dims:
            encoders[_KEY[name]] = _HeadEncoder(
                width,
                enc_dim,
                hidden=hidden,
                depth=encoder_depth,
                base=base,
                dtype=dtype,
            )
        self.encoders = encoders
        out = n_components * trunk_width
        if per_sample_bias:
            out += n_components
        fusion_in = self.layout.fusion_dim
        linears: list[nn.Linear] = []
        prev = fusion_in
        for _ in range(depth):
            linears.append(nn.Linear(prev, hidden, dtype=dtype))
            prev = hidden
        linears.append(nn.Linear(prev, out, dtype=dtype))
        self.fusion = nn.ModuleList(linears)

    def forward(
        self,
        *,
        sensors: Tensor,
        parameters: Tensor | None = None,
        boundary: Tensor | None = None,
        geometry: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return ``(coeffs, bias_or_None)`` from the conditioning heads."""
        parts: list[Tensor] = [self.encoders["function"](sensors)]
        if "pde_params" in self.encoders:
            assert parameters is not None
            parts.append(self.encoders["pde_params"](parameters))
        if "boundary" in self.encoders:
            assert boundary is not None
            parts.append(self.encoders["boundary"](boundary))
        if "geometry" in self.encoders:
            assert geometry is not None
            parts.append(self.encoders["geometry"](geometry))
        h = torch.cat(parts, dim=-1)
        n = len(self.fusion)
        for i, lin in enumerate(self.fusion):
            h = lin(h)
            if i < n - 1:
                h = self.spec.forward(h)
        leading = h.shape[:-1]
        c_p = self.n_components * self.trunk_width
        coeffs = h[..., :c_p].reshape(*leading, self.n_components, self.trunk_width)
        if self.per_sample_bias:
            return coeffs, h[..., c_p:]
        return coeffs, None


class DeepONetField(_JetFieldBase):
    """A conditioned DeepONet evaluated as an omnibias PINN field.

    Constructed by :meth:`DeepONetOperator.condition`. Carries live per-sample
    branch coefficients; the trunk jet is cached readout-independently under
    :data:`TRUNK_JET_CACHE_KEY`.

    Parameters
    ----------
    net
        The :class:`_DeepONetCore` wrapping the trunk.
    coeffs
        Branch coefficients of shape ``(F, C, p)`` (or ``(C, p)`` for F=1).
    bias
        Additive bias of shape ``(F, C)`` or ``(C,)``. Added to jet row 0 only.
    jet_order
        Highest derivative order the residual will request.
    """

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        net: _DeepONetCore,
        coeffs: Tensor,
        bias: Tensor,
        jet_order: int = 2,
        n_functions: int | None = None,
        shared_query_size: int | None = None,
    ) -> None:
        super().__init__(
            coordinate_spec=coordinate_spec,
            components=components,
            net=net,
            jet_order=jet_order,
        )
        if coeffs.ndim == 2:
            coeffs = coeffs.unsqueeze(0)
        if coeffs.ndim != 3:
            raise ValueError(
                f"coeffs must have shape (F, C, p) or (C, p); got {tuple(coeffs.shape)}"
            )
        C = components.n_components
        p = net.trunk.out_dim
        if coeffs.shape[-2] != C or coeffs.shape[-1] != p:
            raise ValueError(
                f"coeffs trailing dims must be (C={C}, p={p}); got {tuple(coeffs.shape)}"
            )
        if bias.ndim == 1:
            if bias.shape[0] != C:
                raise ValueError(f"bias shape must be (C,)={C}, got {tuple(bias.shape)}")
        elif bias.ndim == 2:
            if bias.shape[-1] != C:
                raise ValueError(
                    f"bias trailing dim must be C={C}, got {tuple(bias.shape)}"
                )
            if bias.shape[0] != coeffs.shape[0]:
                raise ValueError(
                    f"bias batch {bias.shape[0]} != coeffs batch {coeffs.shape[0]}"
                )
        else:
            raise ValueError(f"bias must be 1-D or 2-D, got shape {tuple(bias.shape)}")
        # Buffers (not parameters): coefficients are produced by the branch and
        # attached at condition-time; gradients still flow through them because
        # they remain part of the autograd graph when created from branch(sensors).
        self.coeffs = coeffs
        self.bias = bias
        self._n_functions = int(n_functions if n_functions is not None else coeffs.shape[0])
        self._shared_query_size = (
            int(shared_query_size) if shared_query_size is not None else None
        )

    # -- overrides: trunk jet + per-sample branch readout ----------------------- #

    @property
    def _core(self) -> _DeepONetCore:
        return cast(_DeepONetCore, self.net)

    def _compute_hidden_jet(self, coords: Tensor, order: int) -> Tensor:
        """Batched trunk jet of shape ``(Q, M, p)`` (readout-independent)."""
        return vmap(lambda xi: self._core._point_hidden_jet(xi, order))(coords)

    def _contract(
        self, trunk_jet: Tensor, coeffs: Tensor, bias: Tensor
    ) -> Tensor:
        """``einsum`` trunk jet with branch coeffs; add bias to row 0.

        ``trunk_jet``: ``(Q, M, p)`` or ``(F*Q, M, p)``
        ``coeffs``: ``(F, C, p)``
        returns ``(B, M, C)`` where ``B`` matches the leading dim of ``trunk_jet``.
        """
        # trunk_jet (B, M, p), coeffs need broadcasting to B.
        B = trunk_jet.shape[0]
        F = coeffs.shape[0]
        if self._shared_query_size is not None:
            Q = self._shared_query_size
            if B != F * Q:
                raise ValueError(
                    f"shared-grid coords leading dim {B} != F*Q = {F}*{Q}"
                )
            # Expand coeffs from (F, C, p) to (F*Q, C, p) by repeating each
            # sample's coeffs across its Q query points.
            coeffs_b = coeffs.repeat_interleave(Q, dim=0)  # (F*Q, C, p)
            if bias.ndim == 1:
                bias_b = bias.unsqueeze(0).expand(B, -1)
            else:
                bias_b = bias.repeat_interleave(Q, dim=0)
        else:
            if F == 1 and B != 1:
                # Single conditioned field evaluated at B query points.
                coeffs_b = coeffs.expand(B, -1, -1)
                bias_b = bias.expand(B, -1)
            elif F == B:
                coeffs_b = coeffs
                bias_b = bias if bias.ndim == 2 else bias.unsqueeze(0).expand(B, -1)
            else:
                raise ValueError(
                    f"cannot align coeffs batch F={F} with coords batch B={B}; "
                    f"use on_grid() for a shared query grid"
                )
        # (B, M, p) x (B, C, p) -> (B, M, C)
        out = torch.einsum("bmp,bcp->bmc", trunk_jet, coeffs_b)
        out = out.clone()
        out[:, 0, :] = out[:, 0, :] + bias_b
        return out

    def _jet_at_least(
        self, state: FieldState[Tensor], order: int
    ) -> tuple[Tensor, int]:
        """Return ``(operator_jet, jet_order)`` with ``jet_order >= order``.

        The *trunk* jet is memoised under :data:`TRUNK_JET_CACHE_KEY` (independent
        of the branch coefficients); the live per-sample readout is applied on
        every call.
        """
        # FieldState slots are unannotated; cast past the attribute-DSL __getattr__.
        coords = cast(Tensor, state.coords)
        extra = cast(dict[str, Any], state.extra)
        cache = cast(
            "dict[int, Tensor]", extra.setdefault(TRUNK_JET_CACHE_KEY, {})
        )
        trunk: Tensor | None = None
        got_order = -1
        for cached_order in sorted(cache):
            if cached_order >= order:
                trunk = cache[cached_order]
                got_order = cached_order
                break
        if trunk is None:
            want = max(int(order), self.jet_order)
            # Shared-grid path: cache the compact (Q, M, p) trunk jet and
            # expand to (F*Q, M, p) only at contraction time.
            if self._shared_query_size is not None:
                Q = self._shared_query_size
                trunk = self._compute_hidden_jet(coords[:Q], want)
            else:
                trunk = self._compute_hidden_jet(coords, want)
            cache[want] = trunk
            got_order = want
        if self._shared_query_size is not None and trunk.shape[0] == self._shared_query_size:
            trunk = trunk.repeat(self._n_functions, 1, 1)
        return self._contract(trunk, self.coeffs, self.bias), got_order

    def forward_values(self, coords: Tensor) -> Tensor:
        """All component values ``(B, C)`` from trunk value × branch coeffs."""
        # Trunk value: (B_t, p). For shared grid B_t = Q; else B_t = B.
        if self._shared_query_size is not None:
            Q = self._shared_query_size
            trunk_val = self._core.trunk.value(coords[:Q])  # (Q, p)
            trunk_val = trunk_val.repeat(self._n_functions, 1)  # (F*Q, p)
        else:
            trunk_val = self._core.trunk.value(coords)  # (B, p)
        # Fake a 1-row jet so _contract adds the bias correctly.
        trunk_jet = trunk_val.unsqueeze(1)  # (B, 1, p)
        out = self._contract(trunk_jet, self.coeffs, self.bias)  # (B, 1, C)
        return out[:, 0, :]

    def value_component(self, state: FieldState[Tensor], name: str) -> Tensor:
        """Component value from the plain forward path (no jet)."""
        ci = self.components.index(name)
        return self.forward_values(cast(Tensor, state.coords))[:, ci]

    def on_grid(self, query_coords: Tensor) -> FieldState[Tensor]:
        """Evaluate on a query grid shared by every conditioned sample.

        ``query_coords`` has shape ``(Q, D)``. The returned state has coords of
        shape ``(F*Q, D)`` (samples slow, queries fast) and caches a single
        trunk jet of cost ``O(Q)`` rather than ``O(F Q)``.
        """
        if query_coords.ndim != 2:
            raise ValueError(
                f"query_coords must be 2-D (Q, D); got shape {tuple(query_coords.shape)}"
            )
        if query_coords.shape[-1] != self.coordinate_spec.ndim:
            raise ValueError(
                f"query_coords trailing dim {query_coords.shape[-1]} != "
                f"coordinate_spec.ndim {self.coordinate_spec.ndim}"
            )
        Q = int(query_coords.shape[0])
        F = self._n_functions
        tiled = query_coords.repeat(F, 1)
        # Rebuild a field marked for shared-grid contraction.
        field = DeepONetField(
            coordinate_spec=self.coordinate_spec,
            components=self.components,
            net=self._core,
            coeffs=self.coeffs,
            bias=self.bias,
            jet_order=self.jet_order,
            n_functions=F,
            shared_query_size=Q,
        )
        # Keep parameter / buffer identity for autograd.
        field.coeffs = self.coeffs
        field.bias = self.bias
        return cast(FieldState[Tensor], field.evaluate(tiled))


class DeepONetOperator(nn.Module):
    """Trainable DeepONet: ``condition(sensors) -> DeepONetField``.

    Parameters
    ----------
    spec
        :class:`~omnibias.pinn.operator._core.spec.OperatorSpec`.
    trunk_hidden, trunk_depth
        Trunk :class:`~omnibias.torch.architectures.pinn.JetMLP` width / depth.
    branch_hidden, branch_depth
        Branch MLP width / depth.
    base
        Activation for trunk and branch (must expose a closed-form fast path
        for the trunk).
    jet_order
        Highest derivative order residuals will request.
    per_sample_bias
        If True the branch emits a per-sample bias of shape ``(F, C)``; else a
        shared learnable ``(C,)`` bias is used.
    dtype
        Parameter dtype (default ``float64``).
    """

    def __init__(
        self,
        spec: OperatorSpec,
        *,
        trunk_hidden: int = 64,
        trunk_depth: int = 3,
        branch_hidden: int = 64,
        branch_depth: int = 2,
        base: str | ActivationSpec[Tensor] = "tanh",
        jet_order: int = 2,
        per_sample_bias: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.jet_order = int(jet_order)
        self.per_sample_bias = bool(per_sample_bias)
        trunk = JetMLP(
            in_dim=spec.ndim,
            hidden=trunk_hidden,
            out_dim=spec.trunk_width,
            depth=trunk_depth,
            base=base,
        )
        trunk.to(dtype)
        self.core = _DeepONetCore(trunk, n_components=spec.n_components)
        self.core._check_fastpath(jet_order)
        conditioning = spec.conditioning
        if conditioning is None:
            conditioning = ConditioningSpec.function_only(spec.n_sensors)
        self.branch = _BranchNet(
            conditioning=conditioning,
            n_components=spec.n_components,
            trunk_width=spec.trunk_width,
            hidden=branch_hidden,
            depth=branch_depth,
            base=base,
            per_sample_bias=per_sample_bias,
            dtype=dtype,
        )
        if not per_sample_bias:
            self.shared_bias = nn.Parameter(torch.zeros(spec.n_components, dtype=dtype))
        else:
            self.register_parameter("shared_bias", None)

    @property
    def coordinate_spec(self) -> CoordinateSpec:
        return self.spec.coordinate_spec

    @property
    def components(self) -> ComponentSpec:
        return self.spec.components

    def _validate_heads(
        self,
        sensors: Tensor,
        *,
        parameters: Tensor | None = None,
        boundary: Tensor | None = None,
        geometry: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        """Validate and broadcast conditioning heads to batch ``F``."""
        cond = self.spec.conditioning
        assert cond is not None
        if sensors.ndim == 1:
            sensors = sensors.unsqueeze(0)
        if sensors.ndim != 2:
            raise ValueError(
                f"sensors must be 1-D or 2-D; got shape {tuple(sensors.shape)}"
            )
        if sensors.shape[-1] != cond.n_function_sensors:
            raise ValueError(
                f"sensors trailing dim must be n_function_sensors="
                f"{cond.n_function_sensors}, got {tuple(sensors.shape)}"
            )
        F = int(sensors.shape[0])

        def _check(name: str, t: Tensor | None, width: int) -> Tensor | None:
            if width == 0:
                if t is not None:
                    raise ValueError(
                        f"{name} provided but conditioning.{name} width is 0"
                    )
                return None
            if t is None:
                raise ValueError(
                    f"{name} required: conditioning width is {width}"
                )
            if t.ndim == 1:
                t = t.unsqueeze(0)
            if t.ndim != 2 or t.shape[-1] != width:
                raise ValueError(
                    f"{name} must have shape (F, {width}) or ({width},); "
                    f"got {tuple(t.shape)}"
                )
            if t.shape[0] == 1 and F > 1:
                t = t.expand(F, -1)
            elif t.shape[0] != F:
                raise ValueError(
                    f"{name} batch {t.shape[0]} != sensors batch {F}"
                )
            return t

        return (
            sensors,
            _check("parameters", parameters, cond.n_parameters),
            _check("boundary", boundary, cond.n_boundary_sensors),
            _check("geometry", geometry, cond.n_geometry_probes),
        )

    def condition(
        self,
        sensors: Tensor,
        *,
        parameters: Tensor | None = None,
        boundary: Tensor | None = None,
        geometry: Tensor | None = None,
    ) -> DeepONetField:
        """Attach branch coefficients for the multi-head conditioning input.

        ``sensors`` has shape ``(F, m)`` or ``(m,)``. Optional ``parameters``,
        ``boundary``, and ``geometry`` heads are required exactly when the
        operator's :class:`~omnibias.pinn.operator.ConditioningSpec` declares
        a non-zero width for that head.
        """
        sensors_v, params_v, boundary_v, geometry_v = self._validate_heads(
            sensors,
            parameters=parameters,
            boundary=boundary,
            geometry=geometry,
        )
        coeffs, bias = self.branch(
            sensors=sensors_v,
            parameters=params_v,
            boundary=boundary_v,
            geometry=geometry_v,
        )
        if bias is None:
            assert self.shared_bias is not None
            bias = self.shared_bias
        return DeepONetField(
            coordinate_spec=self.spec.coordinate_spec,
            components=self.spec.components,
            net=self.core,
            coeffs=coeffs,
            bias=bias,
            jet_order=self.jet_order,
            n_functions=int(sensors_v.shape[0]),
        )


def build_deeponet(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    n_sensors: int,
    trunk_width: int = 32,
    trunk_hidden: int = 64,
    trunk_depth: int = 3,
    branch_hidden: int = 64,
    branch_depth: int = 2,
    base: str | ActivationSpec[Tensor] = "tanh",
    jet_order: int = 2,
    per_sample_bias: bool = True,
    dtype: torch.dtype = torch.float64,
    conditioning: Any = None,
) -> DeepONetOperator:
    """Build a :class:`DeepONetOperator` from explicit metadata."""
    spec = OperatorSpec(
        coordinate_spec=coordinate_spec,
        components=components,
        n_sensors=n_sensors,
        trunk_width=trunk_width,
        conditioning=conditioning,
    )
    return DeepONetOperator(
        spec,
        trunk_hidden=trunk_hidden,
        trunk_depth=trunk_depth,
        branch_hidden=branch_hidden,
        branch_depth=branch_depth,
        base=base,
        jet_order=jet_order,
        per_sample_bias=per_sample_bias,
        dtype=dtype,
    )


# DeepONetField inherits _omnibias_dispatch / _omnibias_readout_independent
# from _JetFieldBase ("jet_mlp", True). That is honest: the trunk-jet cache is
# independent of the branch coefficients, and every op re-reads them live.
DeepONetField._omnibias_dispatch = "jet_mlp"
DeepONetField._omnibias_readout_independent = True

__all__ = [
    "DeepONetField",
    "DeepONetOperator",
    "PerSampleReadoutError",
    "TRUNK_JET_CACHE_KEY",
    "build_deeponet",
]
