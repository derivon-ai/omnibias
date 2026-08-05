# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Measuring loss-term statistics, and pointwise self-adaptive weights (torch).

Two halves of the adaptive-weighting story, split by what is backend-specific:

* The **measurement**. :func:`grad_stats` and :func:`ntk_trace_stats` reduce the
  per-term parameter gradients to host-side floats, which is all
  :mod:`omnibias.pinn._core.weighting` needs -- the EMA / cadence state machine
  itself is shared pure Python, so torch and jax weights agree by construction.
* The **pointwise weights**. :func:`self_adaptive_loss` is a different animal:
  its weights are one per collocation point, they are *trained* rather than
  estimated, and they are trained by gradient **ascent** (McClenny &
  Braga-Neto's soft-attention mechanism). Nothing about that reduces to a float,
  so it is a real tensor primitive.

References
----------
Wang, Teng & Perdikaris, *Understanding and mitigating gradient pathologies in
physics-informed neural networks*, arXiv:2001.04536 (2021).

McClenny & Braga-Neto, *Self-Adaptive Physics-Informed Neural Networks using a
Soft Attention Mechanism*, arXiv:2009.04544 (2020).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from omnibias.pinn._core.weighting import GradStats
from omnibias.torch.activations.registry import ActivationSpec, get_activation
from torch import Tensor


def grad_stats(
    losses: Mapping[str, Tensor],
    params: Iterable[Tensor],
    *,
    retain_graph: bool = True,
) -> dict[str, GradStats]:
    r"""Per-term ``max`` and ``mean`` of ``|dL_k / dtheta|``.

    The measurement half of :class:`~omnibias.pinn._core.weighting.GradNormWeighter`:
    one backward pass per term, reduced to two floats each.

    Both statistics run over the *same* parameter set, and a parameter a term
    does not touch counts as an exact zero rather than being dropped. That
    matters: ``mean_theta`` is a statement about the whole network, so silently
    shrinking the denominator for a term with narrow support would inflate its
    weight.

    Parameters
    ----------
    losses:
        Named scalar loss terms, all built from the same graph.
    params:
        The parameters to differentiate against; entries with
        ``requires_grad=False`` are skipped.
    retain_graph:
        Keep the graph alive after the last term. The default is ``True``
        because the caller almost always still has to backward the combined
        loss; pass ``False`` when this is the last use of the graph.

    Returns
    -------
    ``{name: GradStats(max_abs, mean_abs)}``.
    """
    if not losses:
        raise ValueError("grad_stats: empty losses dict")
    param_list = [p for p in params if p.requires_grad]
    if not param_list:
        raise ValueError("grad_stats: no parameters with requires_grad=True")

    out: dict[str, GradStats] = {}
    n = len(losses)
    for i, (name, loss) in enumerate(losses.items()):
        if loss.ndim != 0:
            raise ValueError(
                f"grad_stats: loss {name!r} must be a scalar, got shape "
                f"{tuple(loss.shape)}"
            )
        grads = torch.autograd.grad(
            loss,
            param_list,
            retain_graph=retain_graph or i < n - 1,
            allow_unused=True,
            create_graph=False,
        )
        total = 0.0
        count = 0
        peak = 0.0
        for g, p in zip(grads, param_list, strict=True):
            count += p.numel()
            if g is None:
                continue
            a = g.detach().abs()
            total += float(a.sum())
            peak = max(peak, float(a.max()))
        out[name] = GradStats(max_abs=peak, mean_abs=total / count)
    return out


def ntk_trace_stats(
    losses: Mapping[str, Tensor],
    params: Iterable[Tensor],
    *,
    retain_graph: bool = True,
) -> dict[str, float]:
    """Per-term NTK trace proxy ``sum_theta (dL_k / dtheta)^2``, as floats.

    The measurement half of :class:`~omnibias.pinn._core.weighting.NTKWeighter`;
    the same proxy as :func:`~omnibias.pinn.torch.losses.estimate_ntk_trace`,
    computed for every term in one sweep.
    """
    if not losses:
        raise ValueError("ntk_trace_stats: empty losses dict")
    param_list = [p for p in params if p.requires_grad]
    if not param_list:
        raise ValueError("ntk_trace_stats: no parameters with requires_grad=True")

    out: dict[str, float] = {}
    n = len(losses)
    for i, (name, loss) in enumerate(losses.items()):
        grads = torch.autograd.grad(
            loss,
            param_list,
            retain_graph=retain_graph or i < n - 1,
            allow_unused=True,
            create_graph=False,
        )
        total = 0.0
        for g in grads:
            if g is not None:
                total += float((g.detach() ** 2).sum())
        out[name] = total
    return out


def reverse_gradient(x: Tensor) -> Tensor:
    """Identity forward, negated gradient backward.

    ``2 x_detached - x`` is exactly ``x`` in IEEE-754 (Sterbenz: ``2x`` is exact
    and ``2x - x`` is exact), and differentiates to ``-1``. Turning an ascent
    into a descent this way means one ordinary optimiser and one ``backward()``
    drive both halves of a minimax objective -- and because Adam's update is
    ``m / sqrt(v)``, flipping the gradient's sign flips ``m`` and leaves ``v``,
    so this is exactly ``maximize=True`` on those parameters, not an
    approximation of it.
    """
    return 2.0 * x.detach() - x


_MASK_SHORTCUTS = {"identity", "square"}


def _mask_value(lam: Tensor, mask: str | ActivationSpec[Tensor]) -> Tensor:
    """Apply a soft-attention mask ``m`` to the raw weights."""
    if isinstance(mask, str):
        if mask == "identity":
            return lam
        if mask == "square":
            return lam * lam
        try:
            spec = get_activation(mask)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"unknown mask {mask!r}: expected an ActivationSpec, one of "
                f"{sorted(_MASK_SHORTCUTS)}, or an omnibias activation name"
            ) from exc
    else:
        spec = mask
    return spec.forward(lam)


def self_adaptive_loss(
    residual: Tensor,
    lambdas: Tensor,
    *,
    mask: str | ActivationSpec[Tensor] = "sigmoid",
    ascent: bool = True,
) -> Tensor:
    r"""Soft-attention self-adaptive residual loss (McClenny & Braga-Neto 2020).

    .. math::

        L(\theta, \lambda) = \frac{1}{N} \sum_k m(\lambda_k)\, r_k(\theta)^2

    minimised over the network parameters and **maximised** over the pointwise
    weights ``lambda``. The maximisation is what makes it work: a point the
    network fits badly grows its own weight, so the optimiser is pushed toward
    the stiff parts of the domain -- shock fronts, boundary layers, the first
    instants after an initial condition -- instead of averaging them away. It is
    the pointwise counterpart of the per-term weighters in
    :mod:`omnibias.pinn._core.weighting`, and composes with them: weight the
    terms, and let each term weight its own points.

    With ``ascent=True`` (the default) the mask's gradient is reversed, so a
    single ``loss.backward()`` and a single optimiser holding both ``theta`` and
    ``lambdas`` performs descent on one and ascent on the other. Pass
    ``ascent=False`` to get the plain weighted loss -- for a separate ascent
    optimiser, or for a diagnostic.

    Parameters
    ----------
    residual:
        Pointwise residual ``r_k``, any shape.
    lambdas:
        Raw (pre-mask) weights, broadcastable to ``residual``. Usually a
        trainable parameter of the same length as the collocation batch.
    mask:
        The monotone non-negative map ``m``. An omnibias activation name or
        :class:`ActivationSpec` (``"sigmoid"``, ``"softplus"``, ...) -- so the
        mask comes from the shared dictionary and is bit-identical across
        backends -- or ``"identity"`` / ``"square"`` for the polynomial masks of
        the original paper. A bounded mask like ``"sigmoid"`` caps how far one
        point can dominate; ``"identity"`` does not.
    ascent:
        Reverse the gradient reaching ``lambdas``.
    """
    try:
        broadcast = torch.broadcast_shapes(residual.shape, lambdas.shape)
    except RuntimeError:
        broadcast = None
    if broadcast != residual.shape:
        raise ValueError(
            f"lambdas of shape {tuple(lambdas.shape)} do not broadcast to "
            f"residual of shape {tuple(residual.shape)}"
        )
    w = _mask_value(lambdas, mask)
    if ascent:
        w = reverse_gradient(w)
    return (w * residual**2).mean()


class SelfAdaptiveWeights(torch.nn.Module):
    """Trainable pointwise weights for :func:`self_adaptive_loss`.

    Holds one raw weight per collocation point and applies the mask, so the
    weights join ``field.parameters()`` in the ordinary way and a single Adam
    over both does the minimax.

    Parameters
    ----------
    n_points:
        Number of collocation points.
    mask:
        As for :func:`self_adaptive_loss`.
    init:
        Initial raw weight. The default ``0.0`` starts a ``"sigmoid"`` mask at
        ``m = 0.5``, i.e. uniform attention, which is the neutral start.
    ascent:
        Reverse the gradient reaching the weights.
    dtype:
        Defaults to the framework default dtype.
    """

    def __init__(
        self,
        n_points: int,
        *,
        mask: str | ActivationSpec[Tensor] = "sigmoid",
        init: float = 0.0,
        ascent: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1, got {n_points}")
        dt = torch.get_default_dtype() if dtype is None else dtype
        self.mask = mask
        self.ascent = bool(ascent)
        self.raw = torch.nn.Parameter(torch.full((n_points,), float(init), dtype=dt))

    def attention(self) -> Tensor:
        """The current masked weights ``m(lambda)``, detached, for diagnostics."""
        with torch.no_grad():
            return _mask_value(self.raw, self.mask).detach()

    def forward(self, residual: Tensor) -> Tensor:
        """The self-adaptive loss of a residual with matching leading length."""
        if residual.shape[0] != self.raw.shape[0]:
            raise ValueError(
                f"residual has {residual.shape[0]} points but this module holds "
                f"{self.raw.shape[0]} weights"
            )
        lam = self.raw.reshape((-1,) + (1,) * (residual.dim() - 1))
        return self_adaptive_loss(
            residual, lam, mask=self.mask, ascent=self.ascent
        )


__all__ = [
    "SelfAdaptiveWeights",
    "grad_stats",
    "ntk_trace_stats",
    "reverse_gradient",
    "self_adaptive_loss",
]
