# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form deep-network Laplacian fast lane (torch).

Bit-identical twin of :mod:`omnibias.jax.laplacian`; see that module's
``deep_field_*`` section for the full derivation and honesty notes. This
module additionally carries every ``neural_field_*`` one-layer kernel (torch
previously had no ``laplacian`` module at all -- this closes that asymmetry)
plus the deep-network family:

* :func:`deep_field_laplacian` / :func:`deep_field_value_grad_laplacian`
  (Tier A) -- the forward-Laplacian recursion, exact and ceiling-free at any
  input dimension ``D``.
* :func:`deep_field_polylaplacian` (Tiers B/C) -- exact via a support-grouped
  local multivariate jet while a budget is respected, else an unbiased
  sphere-average estimator via the directional jet.

The one-layer kernels below (``neural_field_*``) are ports of the JAX
originals, useful on their own and as the depth-1 reference the deep kernels
are pinned against in the test suite.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from omnibias.core.contraction import (
    DEFAULT_SUPPORT_BUDGET,
    Mode,
    PolylaplacianReport,
    polylaplacian_normalizer,
    polylaplacian_support_terms,
    select_mode,
)
from omnibias.core.multi_index import index_position, multi_index_factorial
from omnibias.core.verified.sampled import ConcentrationReport, hoeffding_enclosure
from omnibias.torch.activations.registry import ActivationSpec, get_activation
from omnibias.torch.jet import jet_to_tower, mlp_jet
from omnibias.torch.jet_mv import mlp_jet_mv

import torch
from torch import Tensor
from torch.func import vmap

if TYPE_CHECKING:  # pragma: no cover
    LayerSpec = tuple[Tensor, Tensor | None, ActivationSpec[Tensor] | str | None]

# Activations whose a.e. second derivative is identically zero (piecewise linear
# or step-like). The deep-field Laplacian uses sigma''; a silent zero is a footgun.
_PIECEWISE_AE_SECOND_ORDER_ZERO = frozenset(
    {
        "relu",
        "leaky_relu",
        "relu6",
        "hardtanh",
        "hardsigmoid",
        "hardshrink",
        "threshold",
        "abs",
        "sign",
        "step",
        "softshrink",
    }
)


def _check_piecewise_second_order(
    name: str, *, allow_almost_everywhere: bool
) -> None:
    if allow_almost_everywhere:
        return
    if name in _PIECEWISE_AE_SECOND_ORDER_ZERO:
        raise ValueError(
            f"activation {name!r} has an a.e.-zero second derivative; the "
            "deep-field Laplacian would return a confident zero. Pass "
            "allow_almost_everywhere=True to opt in explicitly."
        )


# ---------------------------------------------------------------------------
# Plain field evaluation
# ---------------------------------------------------------------------------


def neural_field_value(
    x: Tensor,  # (B, D) or (D,)
    W: Tensor,  # (H, D)
    beta: Tensor,  # (H,)
    c: Tensor,  # (H,)
    b: Tensor | float,  # scalar
    activation: str | ActivationSpec[Tensor],
) -> Tensor:
    """``f(x) = b + sum_h c_h sigma(W_h . x + beta_h)``.

    Batched in ``x`` if it has shape ``(B, D)``; returns ``(B,)``. For a
    single point ``(D,)`` it returns a scalar.
    """
    spec = get_activation(activation)
    z = torch.matmul(x, W.t()) + beta  # (B, H) or (H,)
    return b + spec.forward(z) @ c  # (B,) or scalar


# ---------------------------------------------------------------------------
# Closed-form Laplacian (single hidden layer)
# ---------------------------------------------------------------------------


def neural_field_laplacian(
    x: Tensor,
    W: Tensor,
    beta: Tensor,
    c: Tensor,
    activation: str | ActivationSpec[Tensor],
) -> Tensor:
    """``nabla_x^2 f(x)`` in closed form.

    ``b`` does not enter the Laplacian, so it is not a parameter here.
    """
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(
            f"activation {spec.name!r} has no fast-path kernel, cannot use closed-form Laplacian"
        )
    z = torch.matmul(x, W.t()) + beta  # (B, H) or (H,)
    sigma_pp = spec.fastpath(z, 2)  # (B, H) or (H,)
    row_norm_sq = (W * W).sum(dim=-1)  # (H,)
    return (sigma_pp * (c * row_norm_sq)) @ torch.ones_like(c)


def neural_field_value_and_laplacian(
    x: Tensor,
    W: Tensor,
    beta: Tensor,
    c: Tensor,
    b: Tensor | float,
    activation: str | ActivationSpec[Tensor],
) -> tuple[Tensor, Tensor]:
    """Returns ``(f(x), nabla_x^2 f(x))`` in one forward pass.

    Reuses the pre-activation ``z`` between the value and the Laplacian, so
    this is strictly cheaper than calling :func:`neural_field_value` and
    :func:`neural_field_laplacian` separately.
    """
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(f"activation {spec.name!r} has no fast-path kernel")
    z = torch.matmul(x, W.t()) + beta
    sigma_z = spec.forward(z)
    sigma_pp = spec.fastpath(z, 2)
    f = b + sigma_z @ c
    row_norm_sq = (W * W).sum(dim=-1)
    lap = sigma_pp @ (c * row_norm_sq)
    return f, lap


def neural_field_value_grad_laplacian(
    x: Tensor,
    W: Tensor,
    beta: Tensor,
    c: Tensor,
    b: Tensor | float,
    activation: str | ActivationSpec[Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns ``(f(x), grad_x f(x), nabla_x^2 f(x))``.

    Closed form for all three. The gradient is needed by every VMC
    local-energy estimator (it shows up in the ``|grad log psi|^2`` term).
    """
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(f"activation {spec.name!r} has no fast-path kernel")
    z = torch.matmul(x, W.t()) + beta  # (..., H)
    sigma_z = spec.forward(z)  # (..., H)
    sigma_p = spec.fastpath(z, 1)  # (..., H)
    sigma_pp = spec.fastpath(z, 2)  # (..., H)

    f = b + sigma_z @ c  # (..., )
    grad = (sigma_p * c) @ W  # (..., D)
    row_norm_sq = (W * W).sum(dim=-1)  # (H,)
    lap = sigma_pp @ (c * row_norm_sq)  # (..., )
    return f, grad, lap


# ---------------------------------------------------------------------------
# Closed-form full Hessian (single hidden layer)
# ---------------------------------------------------------------------------


def neural_field_hessian(
    x: Tensor,
    W: Tensor,
    beta: Tensor,
    c: Tensor,
    activation: str | ActivationSpec[Tensor],
) -> Tensor:
    """``H_x f`` -- the full ``D x D`` Hessian of the one-layer field.

    ``b`` does not enter the Hessian, so it is not a parameter here. The
    Hessian is symmetric by construction (a sum of symmetric rank-1 outer
    products ``sigma''(z_h) * W_h W_h^T``).
    """
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(
            f"activation {spec.name!r} has no fast-path kernel, cannot use closed-form Hessian"
        )
    z = torch.matmul(x, W.t()) + beta  # (..., H)
    sigma_pp = spec.fastpath(z, 2)  # (..., H)
    weights = sigma_pp * c  # (..., H)
    return torch.einsum("...h,hi,hj->...ij", weights, W, W)


def neural_field_value_grad_hessian(
    x: Tensor,
    W: Tensor,
    beta: Tensor,
    c: Tensor,
    b: Tensor | float,
    activation: str | ActivationSpec[Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns ``(f(x), grad_x f(x), H_x f(x))`` in one fused pass.

    See :func:`omnibias.jax.laplacian.neural_field_value_grad_hessian` for
    the full exposition; this is its bit-identical torch twin.
    """
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(
            f"activation {spec.name!r} has no fast-path kernel, cannot use closed-form Hessian"
        )
    z = torch.matmul(x, W.t()) + beta  # (..., H)
    sigma_z = spec.forward(z)  # (..., H)
    sigma_p = spec.fastpath(z, 1)  # (..., H)
    sigma_pp = spec.fastpath(z, 2)  # (..., H)

    f = b + sigma_z @ c  # (..., )
    grad = (sigma_p * c) @ W  # (..., D)
    weights = sigma_pp * c  # (..., H)
    hessian = torch.einsum("...h,hi,hj->...ij", weights, W, W)
    return f, grad, hessian


# ---------------------------------------------------------------------------
# Closed-form polylaplacian (single hidden layer)
# ---------------------------------------------------------------------------


def neural_field_polylaplacian(
    x: Tensor,
    W: Tensor,
    beta: Tensor,
    c: Tensor,
    activation: str | ActivationSpec[Tensor],
    k: int,
) -> Tensor:
    """``Delta^k f(x) = (nabla^2)^k f(x)`` in closed form.

    Computes the k-th iterated Laplacian of the one-layer field
    ``f(x) = b + sum_h c_h sigma(W_h . x + beta_h)``. ``b`` cancels for
    ``k >= 1``, so it is not a parameter here. Memory cost: ``O(B * H)``,
    independent of ``k``.
    """
    if k < 1:
        raise ValueError(f"polylaplacian order k must be >= 1, got {k}")
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(
            f"activation {spec.name!r} has no fast-path kernel, "
            "cannot use closed-form polylaplacian"
        )
    z = torch.matmul(x, W.t()) + beta  # (..., H)
    sigma_2k = spec.fastpath(z, 2 * k)  # (..., H)
    row_norm_sq = (W * W).sum(dim=-1)  # (H,)
    row_norm_2k = row_norm_sq**k  # (H,)
    return sigma_2k @ (c * row_norm_2k)


def neural_field_value_and_polylaplacian(
    x: Tensor,
    W: Tensor,
    beta: Tensor,
    c: Tensor,
    b: Tensor | float,
    activation: str | ActivationSpec[Tensor],
    k: int,
) -> tuple[Tensor, Tensor]:
    """Returns ``(f(x), Delta^k f(x))`` in one fused pass."""
    if k < 1:
        raise ValueError(f"polylaplacian order k must be >= 1, got {k}")
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(f"activation {spec.name!r} has no fast-path kernel")
    z = torch.matmul(x, W.t()) + beta
    sigma_z = spec.forward(z)
    sigma_2k = spec.fastpath(z, 2 * k)
    f = b + sigma_z @ c
    row_norm_sq = (W * W).sum(dim=-1)
    row_norm_2k = row_norm_sq**k
    poly_lap = sigma_2k @ (c * row_norm_2k)
    return f, poly_lap


def neural_field_local_p4_over_psi(
    x: Tensor,
    W: Tensor,
    beta: Tensor,
    c: Tensor,
    b: Tensor | float,
    activation: str | ActivationSpec[Tensor],
) -> Tensor:
    """Relativistic mass-velocity local-energy operator ``p^4 psi / psi``.

    ``L_rel(x) = p^4 psi(x) / psi(x) = Delta^2 f / f`` for a wavefunction
    ``psi = f`` represented directly by the one-layer field.
    """
    f, p4_psi = neural_field_value_and_polylaplacian(x, W, beta, c, b, activation, k=2)
    return p4_psi / f


# ---------------------------------------------------------------------------
# Deep-network fast lane: no C(D+2k,D) ceiling at any depth or dimension
#
# See omnibias.jax.laplacian for the full derivation (Tier A forward-Laplacian
# recursion, Tier B support-grouped local jet cubature, Tier C spherical
# estimator) and the honest-scope note. This module is the bit-identical
# torch twin.
# ---------------------------------------------------------------------------


def deep_field_value_grad_laplacian(
    x: Tensor,
    layers: Sequence[LayerSpec],
    *,
    allow_almost_everywhere: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""``(f(x), grad_x f(x), nabla_x^2 f(x))`` for a deep MLP, no ceiling.

    Forward-Laplacian recursion (Tier A): carries the Jacobian ``J`` of the
    running activation w.r.t. the *original* input and the scalar Laplacian
    ``L`` of each unit through every layer, using the closed-form
    ``sigma'``/``sigma''`` fastpath (no autodiff through the activation).
    Memory is ``O(B * H * D)`` -- the same order as the forward pass -- with
    no ``comb(D + 2k, D)`` combinatorial term at any ``D``.

    Parameters
    ----------
    x : Tensor of shape ``(D,)`` or ``(B, D)``.
    layers : sequence of ``(W, b, spec)``
        As consumed by :func:`omnibias.torch.jet.mlp_jet` /
        :func:`omnibias.torch.jet_mv.mlp_jet_mv`. ``spec=None`` is a pure
        affine layer (used for the readout); every other layer computes
        ``sigma(W a + b)``.

    Returns
    -------
    value : Tensor of shape ``(C,)`` or ``(B, C)``
    grad : Tensor of shape ``(D, C)`` or ``(B, D, C)``
    laplacian : Tensor of shape ``(C,)`` or ``(B, C)``

    At ``depth=1`` (one hidden layer plus an affine readout) this reproduces
    :func:`neural_field_value_grad_laplacian` to floating-point round-off.
    """
    if not layers:
        raise ValueError("layers must be non-empty")
    x = torch.as_tensor(x)
    batched = x.ndim > 1
    a = x if batched else x.unsqueeze(0)
    B, dim = a.shape
    L = torch.zeros_like(a)  # (B, D); Laplacian of the identity map is 0.
    J: Tensor | None = None  # sentinel for the identity Jacobian (never materialised)
    for W, b, spec in layers:
        W = torch.as_tensor(W)
        u = torch.matmul(a, W.t())
        if b is not None:
            b_t = torch.as_tensor(b)
            u = u + (b_t if b_t.ndim == 1 else b_t)
        first_layer = J is None
        if first_layer:
            # First layer: (identity Jacobian) @ W.T collapses to W.T broadcast
            # over the batch, so the O(B*D^2) identity is never formed.
            J_u = W.t().unsqueeze(0).expand(B, dim, W.shape[0])
        else:
            J_u = torch.tensordot(J, W, dims=([-1], [-1]))  # (B, D, H_out)
        L_u = torch.matmul(L, W.t())  # (B, H_out)
        if spec is None:
            a, J, L = u, J_u, L_u
            continue
        resolved = get_activation(spec)
        if resolved.fastpath is None:
            raise ValueError(
                f"activation {resolved.name!r} has no fast-path kernel, cannot "
                "use the deep-field Laplacian"
            )
        _check_piecewise_second_order(
            resolved.name, allow_almost_everywhere=allow_almost_everywhere
        )
        try:
            sigma_p = resolved.fastpath(u, 1)
            sigma_pp = resolved.fastpath(u, 2)
        except NotImplementedError as exc:
            raise ValueError(
                f"activation {resolved.name!r} fastpath does not support order "
                "2, required for the deep-field Laplacian"
            ) from exc
        a = resolved.forward(u)
        J = sigma_p.unsqueeze(1) * J_u
        if first_layer:
            row_norm_sq = torch.sum(W * W, dim=-1)  # (H,) == ||W_h||^2
            lap_contrib = sigma_pp * row_norm_sq
        else:
            lap_contrib = sigma_pp * torch.sum(J_u * J_u, dim=1)
        L = lap_contrib + sigma_p * L_u
    assert J is not None  # every loop iteration above assigns J; layers is non-empty
    value = a if batched else a[0]
    grad = J if batched else J[0]
    lap = L if batched else L[0]
    return value, grad, lap


def deep_field_laplacian(
    x: Tensor, layers: Sequence[LayerSpec], *, allow_almost_everywhere: bool = False
) -> Tensor:
    r"""``nabla_x^2 f(x)`` for a deep MLP, no ceiling at any ``D`` (Tier A).

    Thin wrapper over :func:`deep_field_value_grad_laplacian` discarding the
    value and gradient. See that function for the recursion and the
    depth-1 collapse onto :func:`neural_field_laplacian`.
    """
    _, _, lap = deep_field_value_grad_laplacian(
        x, layers, allow_almost_everywhere=allow_almost_everywhere
    )
    return lap


def restrict_first_layer(
    layers: Sequence[LayerSpec], x0: Tensor, support: tuple[int, ...]
) -> list[LayerSpec]:
    """Restrict ``layers`` to the input axes in ``support``, folding the rest into the bias.

    If ``x = x0 + E_support @ y`` (``y`` living on the ``support`` axes only,
    everything else pinned at ``x0``), then the first affine layer
    ``z = W x + b`` becomes ``z = W[:, support] @ y + (W @ x0 + b)`` -- exact,
    and the local weight matrix has only ``len(support)`` columns. Used by
    Tier B's per-support local jet and, in
    :mod:`omnibias.pinn.torch.fields.jet_mlp`, to isolate the spatial axes of a
    field that also carries a non-spatial (e.g. time) coordinate.
    """
    W0, b0, spec0 = layers[0]
    W0 = torch.as_tensor(W0)
    cols = torch.as_tensor(support, dtype=torch.long, device=W0.device)
    W_local = W0.index_select(1, cols)
    shift = torch.matmul(W0, torch.as_tensor(x0))
    b_local = shift if b0 is None else shift + torch.as_tensor(b0)
    return [(W_local, b_local, spec0), *layers[1:]]


def _polylaplacian_support(x: Tensor, layers: Sequence[LayerSpec], k: int) -> Tensor:
    """Tier B: exact ``Delta^k`` via the support-grouped local multivariate jet."""
    dim = x.shape[-1]
    terms = polylaplacian_support_terms(dim, k)

    def one_point(xi: Tensor) -> Tensor:
        total: Tensor | None = None
        for support, local_terms in terms:
            local_layers = restrict_first_layer(layers, xi, support)
            s = len(support)
            local_jet = mlp_jet_mv(
                torch.zeros(s, dtype=xi.dtype, device=xi.device), local_layers, 2 * k
            )
            pos = index_position(s, 2 * k)
            for a_local, coeff in local_terms:
                alpha = tuple(2 * v for v in a_local)
                term = local_jet[pos[alpha]] * (multi_index_factorial(alpha) * coeff)
                total = term if total is None else total + term
        assert total is not None
        return total

    return vmap(one_point)(x)


def _polylaplacian_estimator(
    x: Tensor,
    layers: Sequence[LayerSpec],
    k: int,
    *,
    n_directions: int,
    seed: int,
    return_samples: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Tier C: unbiased sphere-average estimator via the directional jet."""
    dim = x.shape[-1]
    gen = torch.Generator(device=x.device).manual_seed(seed)
    g = torch.randn(n_directions, dim, dtype=x.dtype, device=x.device, generator=gen)
    v = g / torch.linalg.norm(g, dim=-1, keepdim=True)
    normalizer = polylaplacian_normalizer(dim, k)

    def one_point(xi: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        def directional_2k(vi: Tensor) -> Tensor:
            jet = mlp_jet(xi, vi, layers, 2 * k)
            return jet_to_tower(jet)[2 * k]

        samples = vmap(directional_2k)(v)  # (n_directions, C)
        scaled = normalizer * samples
        mean = torch.mean(scaled, dim=0)
        if return_samples:
            return mean, scaled
        return mean

    if return_samples:
        means, scaled = vmap(one_point, out_dims=(0, 0))(x)
        return means, scaled
    return vmap(one_point)(x)


def _resolve_polylaplacian_mode(
    dim: int, k: int, mode: str, budget: int | None
) -> tuple[Mode, int]:
    resolved_budget = budget if budget is not None else DEFAULT_SUPPORT_BUDGET
    if k == 1:
        return "forward", resolved_budget
    if mode == "auto":
        return select_mode(dim, k, budget=resolved_budget), resolved_budget
    if mode not in ("forward", "support", "estimator"):
        raise ValueError(f"unknown mode {mode!r}")
    return mode, resolved_budget  # type: ignore[return-value]


def _tier_c_concentration(scaled_samples: Tensor) -> ConcentrationReport:
    """Build a Hoeffding report from the first batch point, first output."""
    flat = [float(x) for x in scaled_samples[0, :, 0]]
    lo = min(flat)
    hi = max(flat)
    if lo == hi:
        hi = lo + 1.0
    return hoeffding_enclosure(flat, value_range=(lo, hi))


def deep_field_polylaplacian(
    x: Tensor,
    layers: Sequence[LayerSpec],
    k: int,
    *,
    mode: str = "auto",
    n_directions: int = 256,
    budget: int | None = None,
    seed: int = 0,
) -> Tensor:
    r"""``Delta^k f(x)`` for a deep MLP at any input dimension ``D``.

    See :func:`deep_field_polylaplacian_with_report` and the JAX twin for the
    full parameter documentation and honesty notes.
    """
    out, _ = deep_field_polylaplacian_with_report(
        x,
        layers,
        k,
        mode=mode,
        n_directions=n_directions,
        budget=budget,
        seed=seed,
    )
    return out


def deep_field_polylaplacian_with_report(
    x: Tensor,
    layers: Sequence[LayerSpec],
    k: int,
    *,
    mode: str = "auto",
    n_directions: int = 256,
    budget: int | None = None,
    seed: int = 0,
) -> tuple[Tensor, PolylaplacianReport]:
    r"""``Delta^k f(x)`` plus metadata about which tier ran.

    Returns the same point estimate as :func:`deep_field_polylaplacian` and a
    :class:`~omnibias.core.contraction.PolylaplacianReport` carrying the
    resolved mode, ``D``, ``k``, the budget, and for Tier C the direction
    count, sample standard error, and a :class:`~omnibias.core.verified.sampled.ConcentrationReport`.
    """
    if k < 1:
        raise ValueError(f"polylaplacian order k must be >= 1, got {k}")
    x = torch.as_tensor(x)
    batched = x.ndim > 1
    x2 = x if batched else x.unsqueeze(0)
    dim = x2.shape[-1]
    resolved_mode, resolved_budget = _resolve_polylaplacian_mode(dim, k, mode, budget)
    concentration = None
    if k == 1:
        out = deep_field_laplacian(x2, layers)
    elif resolved_mode == "forward":
        raise ValueError(
            "mode='forward' only applies to k=1 (the Laplacian); use "
            "'support' or 'estimator' for k >= 2"
        )
    elif resolved_mode == "support":
        out = _polylaplacian_support(x2, layers, k)
    elif resolved_mode == "estimator":
        out, scaled = _polylaplacian_estimator(
            x2,
            layers,
            k,
            n_directions=n_directions,
            seed=seed,
            return_samples=True,
        )
        concentration = _tier_c_concentration(scaled)
    else:
        raise ValueError(f"unknown mode {resolved_mode!r}")
    report = PolylaplacianReport(
        mode=resolved_mode,
        dim=dim,
        k=k,
        budget=resolved_budget,
        n_directions=n_directions if resolved_mode == "estimator" else None,
        concentration=concentration,
    )
    return (out if batched else out[0]), report


__all__ = [
    "deep_field_laplacian",
    "deep_field_polylaplacian",
    "deep_field_polylaplacian_with_report",
    "deep_field_value_grad_laplacian",
    "neural_field_hessian",
    "neural_field_laplacian",
    "neural_field_local_p4_over_psi",
    "neural_field_polylaplacian",
    "neural_field_value",
    "neural_field_value_and_laplacian",
    "neural_field_value_and_polylaplacian",
    "neural_field_value_grad_hessian",
    "neural_field_value_grad_laplacian",
    "restrict_first_layer",
]
