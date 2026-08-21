# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Depth-causal residual sweep (theory 08-05), PyTorch twin.

After each hidden layer a linear decode is a field. The PDE residual of
that field is formed from a closed-form ``layer_jet`` / ``jet_to_tower``
(bias collapse, ``delta -> 0``), optionally wrapped by a hard Dirichlet
factor, and a damped Gauss-Newton step updates only that layer. Later
layers see the corrected activations. This is not time marching
(:func:`~omnibias.pinn.train.torch.march.march_solve`) and not the 08-03
proxy residual.

Do not wrap the driver in ``torch.compile``. ``n_directions >= n_params``
raises :class:`DepthResidualForbidden` unless ``allow_full=True``. Local
GN is greedy, not a global min, and not CCF stretch. Hilbert is out of
scope.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import cast

import torch
from omnibias.core.local_jet import mlp_param_count
from omnibias.pinn.train._core.depth_residual import (
    DepthResidualConfig,
    DepthResidualForbidden,
    DepthResidualReport,
    HardBCKind,
    layer_is_unlocked,
    reject_depth_residual_flood,
)
from omnibias.torch.jet import affine_jet, jet_to_tower, layer_jet
from omnibias.torch.optim import gauss_newton_direction
from omnibias.torch.train_local import make_input_jet
from torch import Tensor
from torch.func import jacrev

Layer = tuple[Tensor, Tensor | None, str | None]
Decode = tuple[Tensor, Tensor | None]
PdeResidualFn = Callable[[Tensor], Tensor]


def _as_layer_list(layers: Sequence[Layer]) -> list[Layer]:
    if not layers:
        raise ValueError("layers must be non-empty")
    out: list[Layer] = []
    for W, b, spec in layers:
        out.append((torch.as_tensor(W), None if b is None else torch.as_tensor(b), spec))
    return out


def _as_decode_one(decode: Decode) -> Decode:
    W, b = decode
    return (torch.as_tensor(W), None if b is None else torch.as_tensor(b))


def _as_decode_list(decode: Decode | Sequence[Decode], n_layers: int) -> list[Decode]:
    if n_layers < 1:
        raise ValueError("layers must be non-empty")
    if isinstance(decode, tuple) and len(decode) == 2 and torch.is_tensor(decode[0]):
        one = _as_decode_one((decode[0], decode[1]))
        return [
            (_clone(one[0]), None if one[1] is None else _clone(one[1]))
            for _ in range(n_layers)
        ]
    seq = list(cast(Sequence[Decode], decode))
    if len(seq) != n_layers:
        raise ValueError(
            f"decode length {len(seq)} must equal the number of layers {n_layers}"
        )
    return [_as_decode_one(item) for item in seq]


def _clone(t: Tensor) -> Tensor:
    return t.detach().clone()


def _param_sizes(layers: Sequence[Layer], decodes: Sequence[Decode]) -> list[tuple[int, int | None]]:
    sizes: list[tuple[int, int | None]] = []
    for W, b, _spec in layers:
        sizes.append((int(W.numel()), None if b is None else int(b.numel())))
    for W, b in decodes:
        sizes.append((int(W.numel()), None if b is None else int(b.numel())))
    return sizes


def _pack(W: Tensor, b: Tensor | None) -> Tensor:
    flat_w = W.reshape(-1)
    if b is None:
        return flat_w
    return torch.cat([flat_w, b.reshape(-1)], dim=0)


def _unpack(theta: Tensor, W: Tensor, b: Tensor | None) -> tuple[Tensor, Tensor | None]:
    n_w = int(W.numel())
    new_w = theta[:n_w].reshape(W.shape)
    if b is None:
        return new_w, None
    return new_w, theta[n_w:].reshape(b.shape)


def _push_layer(z_jet: Tensor, layer: Layer) -> Tensor:
    W, b, spec = layer
    if spec is None:
        return affine_jet(z_jet, W, b)
    return layer_jet(z_jet, W, b, spec)


def _prepare_x(x: Tensor, d_in: int) -> Tensor:
    x_t = torch.as_tensor(x)
    if x_t.ndim == 0:
        x_t = x_t.reshape(1)
    if x_t.ndim == 1 and int(x_t.shape[0]) != d_in:
        if d_in == 1:
            return x_t.unsqueeze(-1)
        raise ValueError(
            f"x shape {tuple(x_t.shape)} is not a single sample of dim {d_in} "
            f"or a batch of scalars"
        )
    if x_t.ndim >= 2 and int(x_t.shape[-1]) != d_in:
        raise ValueError(
            f"x trailing dim {int(x_t.shape[-1])} must equal d_in={d_in}"
        )
    return x_t


def _default_directions(
    d_in: int, k: int, *, dtype: torch.dtype, device: torch.device
) -> Tensor:
    eye = torch.eye(d_in, dtype=dtype, device=device)
    return torch.stack([eye[i % d_in] for i in range(k)], dim=0)


def _coord1d(x: Tensor) -> Tensor:
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim == 1:
        return x
    return x[..., 0]


def apply_hard_bc_tower(field_tower: Tensor, x: Tensor, kind: HardBCKind) -> Tensor:
    """Apply a named 1-D Dirichlet factor to a batched derivative tower."""
    tower = torch.as_tensor(field_tower)
    if tower.ndim == 1:
        tower = tower.unsqueeze(-1)
    order = int(tower.shape[0]) - 1
    xs = _coord1d(x).reshape(-1).to(dtype=tower.dtype, device=tower.device)
    n = int(xs.shape[0])
    if int(tower.shape[1]) != n:
        raise ValueError(
            f"field tower collocation {int(tower.shape[1])} must match x length {n}"
        )
    mask = torch.zeros(order + 1, n, dtype=tower.dtype, device=tower.device)
    if kind == "none":
        mask[0] = 1.0
    elif kind == "unit_interval":
        mask[0] = xs * (1.0 - xs)
        if order >= 1:
            mask[1] = 1.0 - 2.0 * xs
        if order >= 2:
            mask[2] = -2.0
    else:
        mask[0] = 1.0 - xs * xs
        if order >= 1:
            mask[1] = -2.0 * xs
        if order >= 2:
            mask[2] = -2.0
    rows: list[Tensor] = []
    for m in range(order + 1):
        acc = torch.zeros(n, dtype=tower.dtype, device=tower.device)
        for k in range(m + 1):
            acc = acc + float(math.comb(m, k)) * mask[k] * tower[m - k]
        rows.append(acc)
    return torch.stack(rows, dim=0)


def _decoded_tower(
    incoming_jet: Tensor,
    layer: Layer,
    decode: Decode,
    x: Tensor,
    hard_bc: HardBCKind,
) -> Tensor:
    h_jet = _push_layer(incoming_jet, layer)
    n_jet = affine_jet(h_jet, decode[0], decode[1])
    # incoming_jet is (order+1, k, n, d_in); n_jet is (order+1, k, n, 1).
    spatial = n_jet[:, 0]
    if spatial.ndim >= 2 and int(spatial.shape[-1]) == 1:
        spatial = spatial.squeeze(-1)
    tower = jet_to_tower(spatial)
    return apply_hard_bc_tower(tower, x, hard_bc)


def poisson_1d_residual(force: Tensor) -> PdeResidualFn:
    """``r = -u_xx - f`` from a derivative tower (closed-form, not FD)."""
    f = torch.as_tensor(force).reshape(-1)

    def residual(tower: Tensor, force: Tensor = f) -> Tensor:
        t = torch.as_tensor(tower)
        if int(t.shape[0]) < 3:
            raise ValueError("Poisson residual needs jet_order >= 2")
        uxx = t[2].reshape(-1)
        ff = force.to(dtype=uxx.dtype, device=uxx.device).reshape(-1)
        if int(uxx.shape[0]) != int(ff.shape[0]):
            raise ValueError(
                f"u_xx has {int(uxx.shape[0])} entries, force has {int(ff.shape[0])}"
            )
        return -uxx - ff

    return residual


def field_tower(
    layers: Sequence[Layer],
    decode: Decode | Sequence[Decode],
    x: Tensor,
    *,
    config: DepthResidualConfig | None = None,
    layer_index: int | None = None,
    directions: Tensor | None = None,
) -> Tensor:
    """Derivative tower of the hard-BC field through ``layer_index`` (default last)."""
    cfg = config if config is not None else DepthResidualConfig()
    packed = _as_layer_list(layers)
    decodes = _as_decode_list(decode, len(packed))
    ell = len(packed) - 1 if layer_index is None else int(layer_index)
    if ell < 0 or ell >= len(packed):
        raise ValueError(f"layer_index {ell} is out of range for {len(packed)} layers")
    d_in = int(packed[0][0].shape[-1])
    x_t = _prepare_x(x, d_in)
    if directions is None:
        dirs = _default_directions(
            d_in, cfg.n_directions, dtype=x_t.dtype, device=x_t.device
        )
    else:
        dirs = torch.as_tensor(directions, dtype=x_t.dtype, device=x_t.device)
    jet = make_input_jet(x_t, dirs, cfg.jet_order)
    for layer in packed[:ell]:
        jet = _push_layer(jet, layer)
    return _decoded_tower(jet, packed[ell], decodes[ell], x_t, cfg.hard_bc)


def _block_pack(
    layer: Layer,
    decode: Decode,
    update_decode: bool,
    *,
    freeze_layer: bool = False,
) -> Tensor:
    if freeze_layer:
        return _pack(decode[0], decode[1])
    theta = _pack(layer[0], layer[1])
    if not update_decode:
        return theta
    return torch.cat([theta, _pack(decode[0], decode[1])], dim=0)


def _block_unpack(
    theta: Tensor,
    layer: Layer,
    decode: Decode,
    update_decode: bool,
    *,
    freeze_layer: bool = False,
) -> tuple[Layer, Decode]:
    if freeze_layer:
        new_dw, new_db = _unpack(theta, decode[0], decode[1])
        return layer, (new_dw, new_db)
    W0, b0, spec = layer
    n_layer = int(W0.numel()) + (0 if b0 is None else int(b0.numel()))
    new_w, new_b = _unpack(theta[:n_layer], W0, b0)
    new_layer: Layer = (new_w, new_b, spec)
    if not update_decode:
        return new_layer, decode
    new_dw, new_db = _unpack(theta[n_layer:], decode[0], decode[1])
    return new_layer, (new_dw, new_db)


def _residual_of_block(
    theta: Tensor,
    layer: Layer,
    decode: Decode,
    incoming_jet: Tensor,
    x: Tensor,
    pde_residual: PdeResidualFn,
    config: DepthResidualConfig,
    *,
    freeze_layer: bool = False,
) -> Tensor:
    trial_layer, trial_decode = _block_unpack(
        theta, layer, decode, config.update_decode, freeze_layer=freeze_layer
    )
    tower = _decoded_tower(incoming_jet, trial_layer, trial_decode, x, config.hard_bc)
    return pde_residual(tower).reshape(-1)


def _gn_update(
    layer: Layer,
    decode: Decode,
    incoming_jet: Tensor,
    x: Tensor,
    pde_residual: PdeResidualFn,
    config: DepthResidualConfig,
    *,
    freeze_layer: bool = False,
) -> tuple[Layer, Decode, float]:
    theta0 = _block_pack(layer, decode, config.update_decode, freeze_layer=freeze_layer)

    def residual_of(theta: Tensor) -> Tensor:
        return _residual_of_block(
            theta,
            layer,
            decode,
            incoming_jet,
            x,
            pde_residual,
            config,
            freeze_layer=freeze_layer,
        )

    probe = residual_of(theta0)
    if probe.numel() == 0:
        return layer, decode, 0.0
    jac = cast(Tensor, jacrev(residual_of)(theta0))
    if jac.ndim == 1:
        jac = jac.unsqueeze(0)
    delta = gauss_newton_direction(jac, probe, config.damping)
    new_layer, new_decode = _block_unpack(
        theta0 + delta, layer, decode, config.update_decode, freeze_layer=freeze_layer
    )
    new_res = residual_of(theta0 + delta)
    norm = float(torch.linalg.vector_norm(new_res)) if new_res.numel() else 0.0
    return new_layer, new_decode, norm


def _residual_norm(
    layer: Layer,
    decode: Decode,
    incoming_jet: Tensor,
    x: Tensor,
    pde_residual: PdeResidualFn,
    config: DepthResidualConfig,
    *,
    freeze_layer: bool = False,
) -> float:
    theta0 = _block_pack(layer, decode, config.update_decode, freeze_layer=freeze_layer)
    res = _residual_of_block(
        theta0,
        layer,
        decode,
        incoming_jet,
        x,
        pde_residual,
        config,
        freeze_layer=freeze_layer,
    )
    if res.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(res))


def depth_residual_sweep(
    layers: Sequence[Layer],
    decode: Decode | Sequence[Decode],
    pde_residual: PdeResidualFn,
    collocation: Tensor,
    *,
    config: DepthResidualConfig | None = None,
    directions: Tensor | None = None,
    allow_full: bool | None = None,
) -> tuple[list[Layer], list[Decode], DepthResidualReport]:
    """One causal-in-depth PDE-residual sweep.

    ``pde_residual`` consumes the derivative tower of the hard-BC field
    (``tower[k] = d^k u / dx^k``) and must use ``mlp_jet`` / this tower,
    not an unlabelled finite-difference stencil.
    """
    cfg = config if config is not None else DepthResidualConfig()
    packed = _as_layer_list(layers)
    decodes = _as_decode_list(decode, len(packed))
    n_params = mlp_param_count(_param_sizes(packed, decodes))
    full = cfg.allow_full if allow_full is None else bool(allow_full)
    reject_depth_residual_flood(cfg.n_directions, n_params, full)
    d_in = int(packed[0][0].shape[-1])
    x_t = _prepare_x(collocation, d_in)
    if directions is None:
        dirs = _default_directions(
            d_in, cfg.n_directions, dtype=x_t.dtype, device=x_t.device
        )
    else:
        dirs = torch.as_tensor(directions, dtype=x_t.dtype, device=x_t.device)
        if int(dirs.shape[0]) != cfg.n_directions:
            raise ValueError(
                f"directions rows {int(dirs.shape[0])} must equal "
                f"n_directions={cfg.n_directions}"
            )
    incoming = make_input_jet(x_t, dirs, cfg.jet_order)
    updated = list(packed)
    dec_updated = list(decodes)
    norms_before: list[float] = []
    norms_after: list[float] = []
    unlocked: list[bool] = []
    n_gn = 0
    reference_norm: float | None = None
    prev_after: float | None = None
    jet = incoming
    last = len(updated) - 1
    matched_last_steps = cfg.steps_per_layer * len(updated)
    for ell in range(len(updated)):
        freeze = bool(
            cfg.last_layer_only and cfg.last_layer_scope == "decode" and ell == last
        )
        before = _residual_norm(
            updated[ell],
            dec_updated[ell],
            jet,
            x_t,
            pde_residual,
            cfg,
            freeze_layer=freeze,
        )
        if ell == 0:
            reference_norm = before
        may = layer_is_unlocked(ell, prev_after, reference_norm, cfg.unlock_ratio)
        if cfg.last_layer_only:
            may = may and ell == last
        unlocked.append(may)
        norms_before.append(before)
        steps = 0
        if may:
            steps = matched_last_steps if cfg.last_layer_only else cfg.steps_per_layer
        for _ in range(steps):
            updated[ell], dec_updated[ell], _norm = _gn_update(
                updated[ell],
                dec_updated[ell],
                jet,
                x_t,
                pde_residual,
                cfg,
                freeze_layer=freeze,
            )
            n_gn += 1
        after = _residual_norm(
            updated[ell],
            dec_updated[ell],
            jet,
            x_t,
            pde_residual,
            cfg,
            freeze_layer=freeze,
        )
        norms_after.append(after)
        prev_after = after
        jet = _push_layer(jet, updated[ell])
    report = DepthResidualReport(
        residual_norms=tuple(norms_after),
        residual_norms_before=tuple(norms_before),
        unlocked=tuple(unlocked),
        n_directions=cfg.n_directions,
        n_params=n_params,
        n_gn_steps=n_gn,
        last_layer_only=cfg.last_layer_only,
    )
    return updated, dec_updated, report


__all__ = [
    "Decode",
    "DepthResidualConfig",
    "DepthResidualForbidden",
    "DepthResidualReport",
    "Layer",
    "PdeResidualFn",
    "apply_hard_bc_tower",
    "depth_residual_sweep",
    "field_tower",
    "poisson_1d_residual",
]
