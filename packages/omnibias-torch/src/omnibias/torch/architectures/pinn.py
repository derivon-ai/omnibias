# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Physics-informed neural networks built from OMBU primitives.

These architectures sidestep the standard PINN pain point -- repeated
``torch.autograd.grad`` calls to compute spatial / temporal derivatives
of the network output -- by structuring the network as a single hidden
layer of width ``H`` and computing each derivative analytically via the
activation's closed-form derivative tower:

    u(x_1, ..., x_d) = c_bias + sum_{i=1..H} c_i * sigma(W_i . x + b_i)

    du/dx_a       = sum_i c_i * W_{i,a} * sigma'(z_i)
    d^2u/dx_a^2   = sum_i c_i * W_{i,a}^2 * sigma''(z_i)
    d^2u/dx_a dx_b = sum_i c_i * W_{i,a} * W_{i,b} * sigma''(z_i)

with ``z_i = W_i . x + b_i`` and ``sigma^(n)`` evaluated through the
fast-path kernel of the chosen :class:`ActivationSpec`. There is no
``torch.autograd.grad`` in the inner loop and no quotient-of-differences
in the bias-collapse regime.

The :class:`PINNOMBU` base class collects the chain-rule helpers; the
concrete :class:`PINNHeat` implements the 1D heat equation
``u_t = alpha * u_xx``.

For *deep* networks and *arbitrary* derivative order, :class:`JetMLP`
replaces the hand-coded order-2 chain rule with the exact multivariate jet
kernel :func:`omnibias.torch.jet_mv.mlp_jet_mv`: one forward pass yields the
value, gradient, Hessian and every mixed partial up to total order ``N`` with
no ``torch.autograd.grad`` stacking. :class:`DeepPINNHeat` is the deep
counterpart of :class:`PINNHeat` built on it.

*Spectral bias* -- the tendency of plain MLPs to learn low frequencies first --
is mitigated by two omnibias-native, closed-form-tower constructs that share the
same exact jet readout (:class:`_JetMLPCore`):

* :class:`FourierFeatureMLP` lifts the input through a random Fourier-feature
  encoding ``gamma(x) = [cos(B x), sin(B x)]`` (Tancik et al. 2020). Because
  ``cos(z) = sin(z + pi/2)``, the whole encoding is a *single* omnibias ``sin``
  layer, so ``D^alpha u(x)`` stays exactly closed form to arbitrary order -- no
  autodiff for the PDE residual. A tuple of ``frequency_scale`` values gives a
  multi-scale encoding.
* :func:`make_siren` builds a SIREN (Sitzmann et al. 2020): a :class:`JetMLP`
  with ``sin`` activations and the SIREN initialisation. Its derivative tower is
  exact for every order (``sin^{(n)}(z) = sin(z + n pi/2)``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from omnibias.core.multi_index import multi_index_factorial, multi_indices
from omnibias.torch.activations.registry import ActivationSpec, get_activation
from omnibias.torch.jet_mv import jet_gradient, jet_hessian, mlp_jet_mv

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import vmap


class PINNOMBU(nn.Module):
    """Template for closed-form-derivative PINNs.

    Subclasses must define:

    - ``self.W``: ``nn.Linear(in_dim, hidden)`` producing the pre-activation
      ``z = W . input + b``.
    - ``self.c``: ``nn.Linear(hidden, out_dim)`` producing the readout.
    - ``self.spec``: :class:`ActivationSpec` whose ``fastpath`` kernel is
      not None (i.e. derivative orders 1 and 2 must be available).

    The helper methods evaluate the network's value and its partial
    derivatives at any given pre-activation ``z`` via the chain rule.
    """

    W: nn.Linear
    c: nn.Linear
    spec: ActivationSpec[Tensor]

    def _check_fastpath(self, max_order: int) -> None:
        if self.spec.fastpath is None:
            raise ValueError(
                f"PINNOMBU requires a base activation with a closed-form "
                f"derivative kernel; activation {self.spec.name!r} has none."
            )
        try:
            self.spec.fastpath(torch.zeros(1), max_order)
        except NotImplementedError as e:
            raise ValueError(
                f"Activation {self.spec.name!r} fast-path does not support order {max_order}: {e}"
            ) from None

    def _fp(self, z: Tensor, order: int) -> Tensor:
        """Internal fast-path call with ``None`` guarded out by ``_check_fastpath``."""
        fp = self.spec.fastpath
        assert fp is not None, "fastpath checked at __init__"
        out: Tensor = fp(z, order)
        return out

    def base_forward(self, inp: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the value ``u`` and the pre-activation ``z``.

        Returns
        -------
        u : Tensor of shape ``(..., out_dim)``
        z : Tensor of shape ``(..., hidden)``
        """
        z = self.W(inp)
        u_h = self.spec.forward(z)
        u = self.c(u_h)
        return u, z

    def first_derivative(self, z: Tensor, axis: int) -> Tensor:
        """``du / dx_{axis}`` evaluated via the chain rule.

        Returns a tensor of shape ``(..., out_dim)``.
        """
        sigma_p = self._fp(z, 1)  # (..., H)
        W_axis = self.W.weight[:, axis]  # (H,)
        c_w = self.c.weight  # (out_dim, H)
        weighted = sigma_p * W_axis  # (..., H)
        out: Tensor = weighted @ c_w.T  # (..., out_dim)
        return out

    def second_derivative(self, z: Tensor, axis: int) -> Tensor:
        """``d^2 u / dx_{axis}^2`` evaluated via the chain rule."""
        sigma_pp = self._fp(z, 2)
        W_axis = self.W.weight[:, axis]
        c_w = self.c.weight
        weighted = sigma_pp * (W_axis * W_axis)
        out: Tensor = weighted @ c_w.T
        return out

    def mixed_second_derivative(self, z: Tensor, axis_a: int, axis_b: int) -> Tensor:
        """``d^2 u / dx_a dx_b``."""
        sigma_pp = self._fp(z, 2)
        Wa = self.W.weight[:, axis_a]
        Wb = self.W.weight[:, axis_b]
        c_w = self.c.weight
        weighted = sigma_pp * (Wa * Wb)
        out: Tensor = weighted @ c_w.T
        return out


class PINNHeat(PINNOMBU):
    """1D heat equation ``u_t = alpha * u_xx`` solver.

    Single hidden layer of width ``hidden``; spatial and temporal
    derivatives are computed via :class:`PINNOMBU` chain-rule helpers,
    no autograd-through-derivative.

    Parameters
    ----------
    hidden : int, default 64
        Number of hidden units (``H``).
    base : str or :class:`ActivationSpec`, default ``"softplus"``
        Base activation. Must have a fast-path supporting orders 1 and 2.
    alpha : float, default 0.1
        Diffusion coefficient.
    """

    def __init__(
        self,
        hidden: int = 64,
        base: str | ActivationSpec[Tensor] = "softplus",
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.alpha = alpha
        self.W = nn.Linear(2, hidden, bias=True)  # input (x, t) -> hidden
        self.c = nn.Linear(hidden, 1, bias=True)
        self.spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        self._check_fastpath(max_order=2)

    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Compute ``(u, residual)`` at the given collocation points.

        Parameters
        ----------
        x, t : Tensor of shape ``(B,)``

        Returns
        -------
        u : Tensor of shape ``(B,)``
            Network output ``u(x, t)``.
        residual : Tensor of shape ``(B,)``
            PDE residual ``u_t - alpha * u_xx`` (target zero).
        """
        if x.shape != t.shape:
            raise ValueError(f"x and t must have matching shape, got {x.shape} vs {t.shape}.")
        inp = torch.stack([x, t], dim=-1)  # (B, 2); axis 0 is x, axis 1 is t
        u, z = self.base_forward(inp)  # u: (B, 1), z: (B, H)
        u_t = self.first_derivative(z, axis=1)  # (B, 1)
        u_xx = self.second_derivative(z, axis=0)  # (B, 1)
        residual = u_t - self.alpha * u_xx
        return u.squeeze(-1), residual.squeeze(-1)


class _JetMLPCore(nn.Module):
    r"""Shared closed-form-derivative readout for jet-based MLPs.

    Subclasses own the parameters and implement :meth:`_layer_specs`, which returns
    the ``(W, b, spec)`` layer list (``spec=None`` marks the affine readout) consumed
    by the multivariate-jet kernel :func:`omnibias.torch.jet_mv.mlp_jet_mv`. Every
    derivative method below reads a single jet per collocation point and batches it
    with :func:`torch.func.vmap`; there is no ``torch.autograd.grad`` anywhere in the
    differential operator. Concrete networks: :class:`JetMLP` (uniform activation)
    and :class:`FourierFeatureMLP` (sin-encoded random Fourier features).
    """

    in_dim: int
    out_dim: int

    def _layer_specs(
        self,
    ) -> list[tuple[Tensor, Tensor | None, ActivationSpec[Tensor] | None]]:
        raise NotImplementedError  # pragma: no cover - abstract hook

    def _check_fastpath(self, max_order: int) -> None:
        """Reject any layer activation lacking a closed-form derivative kernel of ``max_order``."""
        seen: set[str] = set()
        for _w, _b, spec in self._layer_specs():
            if spec is None or spec.name in seen:
                continue
            seen.add(spec.name)
            if spec.fastpath is None:
                raise ValueError(
                    f"{type(self).__name__} requires activations with a closed-form "
                    f"derivative kernel; activation {spec.name!r} has none."
                )
            try:
                spec.fastpath(torch.zeros(1), max_order)
            except NotImplementedError as e:
                raise ValueError(
                    f"Activation {spec.name!r} fast-path does not support order "
                    f"{max_order}: {e}"
                ) from None

    def _point_jet(self, xi: Tensor, order: int) -> Tensor:
        """Single-point multivariate jet, shape ``(M, out_dim)``.

        Default: the bare network jet. Subclasses that wrap the network (for
        example the hard-constraint ansatz ``u = g + b * net``) override this so
        every readout below stays exact and closed form.
        """
        return mlp_jet_mv(xi, self._layer_specs(), order)

    def value(self, x: Tensor) -> Tensor:
        """Plain network value ``u(x)``, shape ``(..., out_dim)`` (no jet needed)."""
        h = x
        for w, b, spec in self._layer_specs():
            h = h @ w.t()
            if b is not None:
                h = h + b
            if spec is not None:
                h = spec.forward(h)
        return h

    def forward(self, x: Tensor) -> Tensor:
        return self.value(x)

    def jet(self, x: Tensor, order: int) -> Tensor:
        """Batched multivariate jet, shape ``(B, M, out_dim)`` (``M`` multi-indices)."""
        self._check_fastpath(order)

        def f(xi: Tensor) -> Tensor:
            return self._point_jet(xi, order)

        out: Tensor = vmap(f)(x)
        return out

    def gradient(self, x: Tensor) -> Tensor:
        """Exact input gradient ``d u / d x_i``, shape ``(B, in_dim, out_dim)``."""
        self._check_fastpath(1)
        dim = self.in_dim

        def f(xi: Tensor) -> Tensor:
            return jet_gradient(self._point_jet(xi, 1), dim, 1)

        out: Tensor = vmap(f)(x)
        return out

    def hessian(self, x: Tensor) -> Tensor:
        """Exact input Hessian, shape ``(B, in_dim, in_dim, out_dim)``."""
        self._check_fastpath(2)
        dim = self.in_dim

        def f(xi: Tensor) -> Tensor:
            return jet_hessian(self._point_jet(xi, 2), dim, 2)

        out: Tensor = vmap(f)(x)
        return out

    def value_grad_hessian(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """One jet -> ``(value, gradient, Hessian)`` for 2nd-order PDE residuals.

        Shapes ``(B, out_dim)``, ``(B, in_dim, out_dim)``, ``(B, in_dim, in_dim,
        out_dim)``. Cheaper than separate calls: a single order-2 jet per point.
        """
        self._check_fastpath(2)
        dim = self.in_dim

        def f(xi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            j = self._point_jet(xi, 2)
            return j[0], jet_gradient(j, dim, 2), jet_hessian(j, dim, 2)

        res = vmap(f)(x)
        value_b: Tensor = res[0]
        grad_b: Tensor = res[1]
        hess_b: Tensor = res[2]
        return value_b, grad_b, hess_b

    def partials(self, x: Tensor, order: int) -> dict[tuple[int, ...], Tensor]:
        """All raw partials ``{alpha: D^alpha u(x)}`` to total ``order``.

        Values have shape ``(B, out_dim)``. Demonstrates the *arbitrary-order*
        closed-form capability (``D^alpha u = alpha! c_alpha``).
        """
        jet_b = self.jet(x, order)  # (B, M, out_dim)
        idx = multi_indices(self.in_dim, order)
        return {
            alpha: jet_b[:, i] * multi_index_factorial(alpha)
            for i, alpha in enumerate(idx)
        }


class JetMLP(_JetMLPCore):
    r"""Deep MLP whose exact input derivatives come from omnibias multivariate jets.

    A standard fully-connected network

    .. math::

        u(x) = (A_L \circ \sigma \circ A_{L-1} \circ \cdots \circ \sigma \circ A_1)(x),

    but *every* mixed input partial ``D^alpha u(x)`` up to total order ``N`` is
    obtained **exactly** in a single forward pass through
    :func:`omnibias.torch.jet_mv.mlp_jet_mv` -- no ``torch.autograd.grad`` stacking,
    no finite differences, and a cost independent of the derivative order beyond the
    jet truncation. This generalises :class:`PINNOMBU` (single hidden layer, order 2)
    to arbitrary depth and order while keeping the closed-form-tower contract: the
    base activation must expose a fast-path derivative kernel.

    The parameters are ordinary :class:`torch.nn.Linear` leaves, so training gradients
    still flow through the backend autograd / optimiser; omnibias supplies the
    *spatial* differential operator, not the optimiser. Derivatives are read out per
    collocation point and batched with :func:`torch.func.vmap`.

    Parameters
    ----------
    in_dim:
        Number of input coordinates ``D`` (e.g. 2 for ``(x, t)``).
    hidden:
        Hidden width.
    out_dim:
        Number of output components (default 1).
    depth:
        Number of hidden (activated) layers ``>= 1``; the readout is a pure affine
        map. ``depth=1`` recovers a single-hidden-layer network like
        :class:`PINNOMBU`.
    base:
        Base activation (name or :class:`ActivationSpec`) with a closed-form
        derivative fast path (e.g. ``"tanh"``, ``"sigmoid"``, ``"softplus"``).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int = 1,
        depth: int = 3,
        base: str | ActivationSpec[Tensor] = "tanh",
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
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth
        linears: list[nn.Linear] = []
        prev = in_dim
        for _ in range(depth):
            linears.append(nn.Linear(prev, hidden))
            prev = hidden
        linears.append(nn.Linear(prev, out_dim))  # affine readout (no activation)
        self.linears = nn.ModuleList(linears)
        self.spec = base if isinstance(base, ActivationSpec) else get_activation(base)

    def _layer_specs(
        self,
    ) -> list[tuple[Tensor, Tensor | None, ActivationSpec[Tensor] | None]]:
        """Assemble the ``(W, b, spec)`` list ``mlp_jet_mv`` consumes (readout = affine)."""
        n = len(self.linears)
        specs: list[tuple[Tensor, Tensor | None, ActivationSpec[Tensor] | None]] = []
        for i, lin in enumerate(self.linears):
            assert isinstance(lin, nn.Linear)
            act = None if i == n - 1 else self.spec
            specs.append((lin.weight, lin.bias, act))
        return specs


def _as_scales(frequency_scale: float | Sequence[float]) -> tuple[float, ...]:
    """Normalise the ``frequency_scale`` argument to a tuple of positive floats."""
    scales: tuple[float, ...]
    if isinstance(frequency_scale, int | float):
        scales = (float(frequency_scale),)
    else:
        scales = tuple(float(s) for s in frequency_scale)
    if not scales:
        raise ValueError("frequency_scale must contain at least one scale")
    if any(s <= 0.0 for s in scales):
        raise ValueError(f"all frequency scales must be > 0, got {scales}")
    return scales


class FourierFeatureMLP(_JetMLPCore):
    r"""Spectral-bias-mitigating MLP: a sin-encoded random Fourier-feature front end.

    The input is lifted by the random Fourier-feature map (Tancik et al. 2020)

    .. math::

        \gamma(x) = \big[\cos(B x),\ \sin(B x)\big], \qquad B \in \mathbb{R}^{F \times D},

    and a standard :class:`JetMLP` body acts on ``gamma(x)``. Lifting the input into a
    high-frequency basis lets the downstream net represent high-frequency targets with
    *low-frequency* weights, which is the standard cure for the low-frequency
    (spectral) bias of plain MLPs.

    The omnibias twist is that the encoding costs nothing in the differential operator:
    because ``cos(z) = sin(z + pi/2)``,

    .. math::

        \gamma(x) = \sigma_{\sin}\!\big([B;\,B]\,x + [\tfrac{\pi}{2}\mathbf 1;\,\mathbf 0]\big),

    i.e. the whole encoding is a *single* omnibias ``sin`` layer with the exact
    derivative tower ``sin^{(n)}(z) = sin(z + n pi/2)``. Stacked in front of the body,
    the composite ``u(x)`` therefore still yields *every* mixed partial ``D^alpha u(x)``
    in closed form through :func:`omnibias.torch.jet_mv.mlp_jet_mv` -- exact, arbitrary
    order, no ``torch.autograd.grad`` in the PDE residual.

    Parameters
    ----------
    in_dim:
        Number of input coordinates ``D``.
    num_features:
        Number of Fourier features ``F`` *per frequency band*; the encoding width is
        ``2 * F * len(scales)``.
    hidden:
        Hidden width of the body MLP.
    out_dim:
        Number of output components (default 1).
    depth:
        Number of hidden (activated) body layers ``>= 0``; ``depth=0`` is a pure
        random-Fourier-feature model with a linear readout.
    base:
        Body activation (name or :class:`ActivationSpec`) with a closed-form
        derivative fast path (default ``"tanh"``).
    frequency_scale:
        Bandwidth(s) of the Gaussian frequency matrix. A scalar selects one band with
        ``B ~ N(0, (2 pi * scale)^2)``; a sequence concatenates several bands for a
        *multi-scale* encoding (helps when the target mixes low and high frequencies).
    trainable_features:
        If ``True`` the frequency matrix / phases are learnable parameters; otherwise
        they are fixed buffers (the classic random-features regime).
    seed:
        Seed for the frequency-matrix draw.
    """

    spec: ActivationSpec[Tensor]
    sin_spec: ActivationSpec[Tensor]
    W_ff: Tensor
    b_ff: Tensor
    linears: nn.ModuleList

    def __init__(
        self,
        in_dim: int,
        num_features: int = 64,
        hidden: int = 64,
        out_dim: int = 1,
        depth: int = 3,
        base: str | ActivationSpec[Tensor] = "tanh",
        *,
        frequency_scale: float | Sequence[float] = 1.0,
        trainable_features: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        if num_features < 1:
            raise ValueError(f"num_features must be >= 1, got {num_features}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        if depth < 0:
            raise ValueError(f"depth (hidden layers after encoding) must be >= 0, got {depth}")
        scales = _as_scales(frequency_scale)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth
        self.num_features = num_features
        self.scales = scales
        self.spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        self.sin_spec = get_activation("sin")

        gen = torch.Generator().manual_seed(seed)
        bands = [
            torch.randn(num_features, in_dim, generator=gen) * (2.0 * math.pi * s)
            for s in scales
        ]
        b_mat = torch.cat(bands, dim=0)  # (F_total, in_dim)
        f_total = b_mat.shape[0]
        self.feature_dim = 2 * f_total
        # sin([B x + pi/2 ; B x]) = [cos(B x) ; sin(B x)]: the encoding is one sin layer.
        w_ff = torch.cat([b_mat, b_mat], dim=0)  # (2 F_total, in_dim)
        b_ff = torch.cat(
            [torch.full((f_total,), 0.5 * math.pi), torch.zeros(f_total)]
        )
        if trainable_features:
            self.W_ff = nn.Parameter(w_ff)
            self.b_ff = nn.Parameter(b_ff)
        else:
            self.register_buffer("W_ff", w_ff)
            self.register_buffer("b_ff", b_ff)

        linears: list[nn.Linear] = []
        prev = self.feature_dim
        for _ in range(depth):
            linears.append(nn.Linear(prev, hidden))
            prev = hidden
        linears.append(nn.Linear(prev, out_dim))  # affine readout (no activation)
        self.linears = nn.ModuleList(linears)

    def _layer_specs(
        self,
    ) -> list[tuple[Tensor, Tensor | None, ActivationSpec[Tensor] | None]]:
        specs: list[tuple[Tensor, Tensor | None, ActivationSpec[Tensor] | None]] = [
            (self.W_ff, self.b_ff, self.sin_spec)
        ]
        n = len(self.linears)
        for i, lin in enumerate(self.linears):
            assert isinstance(lin, nn.Linear)
            act = None if i == n - 1 else self.spec
            specs.append((lin.weight, lin.bias, act))
        return specs


def make_siren(
    in_dim: int,
    hidden: int,
    out_dim: int = 1,
    depth: int = 3,
    *,
    omega_0: float = 30.0,
    seed: int = 0,
) -> JetMLP:
    r"""Build a SIREN (Sitzmann et al. 2020) as an omnibias :class:`JetMLP`.

    A SIREN is an MLP with ``sin`` activations and a specific initialisation that keeps
    the pre-activation distribution stable across depth. Because the base activation is
    ``sin``, every input derivative is exact for *all* orders via the closed-form tower
    ``sin^{(n)}(z) = sin(z + n pi/2)`` -- so a SIREN trained here has bit-stable,
    arbitrary-order derivatives with no autodiff in the differential operator.

    The first layer's frequency is scaled by ``omega_0`` (folded into its weights),
    weights are drawn ``U(-1/fan_in, 1/fan_in)`` for the first layer and
    ``U(-sqrt(6/fan_in)/omega_0, +sqrt(6/fan_in)/omega_0)`` thereafter, and all biases
    start at zero -- the standard SIREN scheme.

    Parameters
    ----------
    in_dim, hidden, out_dim, depth:
        As for :class:`JetMLP` (``depth >= 1``).
    omega_0:
        First-layer frequency scale (default ``30``, the SIREN default).
    seed:
        Seed for the weight draw.
    """
    if omega_0 <= 0.0:
        raise ValueError(f"omega_0 must be > 0, got {omega_0}")
    net = JetMLP(in_dim, hidden, out_dim=out_dim, depth=depth, base="sin")
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for i, lin in enumerate(net.linears):
            assert isinstance(lin, nn.Linear)
            fan_in = int(lin.weight.shape[1])
            if i == 0:
                bound = 1.0 / fan_in
                lin.weight.uniform_(-bound, bound, generator=gen)
                lin.weight.mul_(omega_0)  # fold omega_0 into the first-layer frequency
            else:
                bound = math.sqrt(6.0 / fan_in) / omega_0
                lin.weight.uniform_(-bound, bound, generator=gen)
            if lin.bias is not None:
                lin.bias.zero_()
    return net


class DeepPINNHeat(nn.Module):
    """Deep 1-D heat-equation PINN ``u_t = alpha * u_xx`` on :class:`JetMLP`.

    The deep, arbitrary-depth counterpart of :class:`PINNHeat`: the spatial and
    temporal derivatives come from a single closed-form order-2 multivariate jet per
    collocation point (no autograd-through-derivative), so the PDE residual is exact.

    Parameters
    ----------
    hidden : int, default 32
        Hidden width.
    depth : int, default 3
        Number of hidden layers.
    base : str or :class:`ActivationSpec`, default ``"tanh"``
        Base activation with a fast path supporting orders 1 and 2.
    alpha : float, default 0.1
        Diffusion coefficient.
    """

    def __init__(
        self,
        hidden: int = 32,
        depth: int = 3,
        base: str | ActivationSpec[Tensor] = "tanh",
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.net = JetMLP(in_dim=2, hidden=hidden, out_dim=1, depth=depth, base=base)
        self.net._check_fastpath(max_order=2)

    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(u, residual)`` with ``residual = u_t - alpha * u_xx`` (target 0)."""
        if x.shape != t.shape:
            raise ValueError(f"x and t must have matching shape, got {x.shape} vs {t.shape}.")
        inp = torch.stack([x, t], dim=-1)  # (B, 2): axis 0 is x, axis 1 is t
        value, grad, hess = self.net.value_grad_hessian(inp)
        u = value.squeeze(-1)  # (B,)
        u_t = grad[:, 1, 0]  # d/dt
        u_xx = hess[:, 0, 0, 0]  # d^2/dx^2
        residual = u_t - self.alpha * u_xx
        return u, residual


__all__ = [
    "DeepPINNHeat",
    "FourierFeatureMLP",
    "JetMLP",
    "PINNHeat",
    "PINNOMBU",
    "make_siren",
]
