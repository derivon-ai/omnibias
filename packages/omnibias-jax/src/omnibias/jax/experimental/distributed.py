# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Data-parallel residual Gram, Gauss-Newton step and NTK diagonal over a device mesh.

The points of every block are sharded over one mesh axis; the weights are replicated. Each device runs the forward-Laplacian
streams of its own points (the whole cost of the forward pass, divided by the number of devices) and then needs only
*compact factors* of the other devices' points to form its rows of ``K = J J^T``: for every hidden layer the input streams
``P`` of shape ``(B, T, in)`` and the output-stream cotangents ``Q`` of shape ``(B, R, T, out)``
(:func:`~omnibias.jax.experimental.deep_residual_gram`), and for the first layer its explicit rows. They are all-gathered
(``O(N T (in + out))`` per layer, against the ``N^2`` of ``K``); the stream identity then fills the local row slab
``K[local rows, all rows]`` at ``1/n_dev`` of the arithmetic. The damped solve ``(K + damping I) alpha = r`` runs on the
gathered, replicated ``K`` (``N^2`` memory per device, ``N^3 / 3`` arithmetic that is **not** divided by the number of
devices); ``J^T alpha`` is formed from local rows and summed with a ``psum``. The result equals the single-device program up
to rounding.

Limits: the residual must act pointwise and may depend on the point only through its ``x`` argument (a per-point array closed
over by the residual would have the global shape inside a shard); the first layer's explicit rows are gathered, ``N`` rows of
``H (D + 1)`` numbers, and the projected program for ``D > H`` has no distributed version; every block's point count must
be a multiple of the mesh size (:func:`~omnibias.jax.experimental.pad_block` pads with exact zero rows).

Each function builds its jitted sharded program once per structure (mesh, axis, layer shapes and activations, residual
functions, coordinates, diffusion values, strategy) and keeps the last sixteen; the weights, the damping and the points are
arguments of that program, so a training loop that calls it from eager code compiles it once and a weight update does not
rebuild it (the program used to close over the weights as constants and compile again on every call). A residual that reads
state outside its arguments sees that state as of the call that built the program, as under ``jax.jit``; pass a fresh function
to rebuild. Inside a training step, calling it under your own ``jax.jit`` remains the way to fuse it with the rest of the step.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from omnibias.jax.experimental import residual_gram as rg
from omnibias.jax.experimental.conditioning import ConditioningPolicy, damped_solve

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec

__all__ = [
    "distributed_residual_gauss_newton",
    "distributed_residual_gram",
    "distributed_residual_ntk_diagonal",
    "shard_points",
]


def shard_points(x: Array, mesh: Mesh, axis: str = "data") -> Array:
    """Place ``x`` of shape ``(B, ...)`` on ``mesh`` with ``B`` split over ``axis``."""
    n = mesh.shape[axis]
    if x.shape[0] % n:
        raise ValueError(f"{x.shape[0]} points do not split over {n} devices")
    return jax.device_put(x, NamedSharding(mesh, PartitionSpec(axis)))


_PROGRAMS: OrderedDict[tuple[Any, ...], Callable[..., Any]] = OrderedDict()
_PROGRAM_LIMIT = 16


def _value_key(v: Any) -> Any:
    """A hashable stand-in for a static argument: arrays by shape, dtype and a digest of their bytes."""
    if v is None or isinstance(v, str | int | float | bool):
        return v
    if isinstance(v, jax.core.Tracer):
        raise TypeError("a traced static argument cannot key a program")
    if isinstance(v, np.ndarray | Array | list | tuple):
        a = np.asarray(v)
        return ("array", a.shape, str(a.dtype), hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    return v


def _spec_key(spec: Any) -> Any:
    return spec if spec is None or isinstance(spec, str) else (type(spec).__qualname__, id(spec))


def _structure(layers: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(
        (tuple(jnp.shape(W)), str(jnp.asarray(W).dtype), b is None, _spec_key(spec)) for W, b, spec in layers
    )


def _arrays(layers: Sequence[Any]) -> list[tuple[Array, Array | None]]:
    return [(jnp.asarray(W), None if b is None else jnp.asarray(b)) for W, b, _ in layers]


def _with_arrays(arrays: Sequence[Any], layers: Sequence[Any]) -> list[tuple[Array, Array | None, Any]]:
    return [(W, b, spec) for (W, b), (_, _, spec) in zip(arrays, layers, strict=True)]


def _program(key: tuple[Any, ...] | None, build: Callable[[], Callable[..., Any]]) -> Callable[..., Any]:
    """The jitted program of ``key``, built once; no key (a traced or unhashable static argument) builds it for one call."""
    if key is not None:
        try:
            hash(key)
        except TypeError:
            key = None
    if key is None:
        return build()
    fn = _PROGRAMS.get(key)
    if fn is None:
        fn = _PROGRAMS[key] = build()
        while len(_PROGRAMS) > _PROGRAM_LIMIT:
            _PROGRAMS.popitem(last=False)
    else:
        _PROGRAMS.move_to_end(key)
    return fn


def _key(kind: str, mesh: Mesh, axis: str, layers: Sequence[Any], blocks: Sequence[Any], *static: Any) -> tuple[Any, ...] | None:
    """The structure that fixes a program: everything but the weights, the damping and the points."""
    try:
        return (
            kind,
            mesh,
            axis,
            _structure(layers),
            tuple((blk[1], *blk[2:]) for blk in blocks),
            *[_value_key(v) for v in static],
        )
    except TypeError:
        return None


def _check(blocks: Sequence[Any], mesh: Mesh, axis: str) -> None:
    n = mesh.shape[axis]
    for blk in blocks:
        if int(jnp.shape(blk[0])[0]) % n:
            raise ValueError(
                f"a block of {int(jnp.shape(blk[0])[0])} points does not split over {n} devices; "
                "pad it with pad_block"
            )


def _gather(a: Array, axis: str) -> Array:
    return jax.lax.all_gather(a, axis, axis=0, tiled=True)


def _local_gram(
    blocks: Sequence[Any],
    layers: Sequence[Any],
    axis: str,
    coordinates: Any,
    diffusion: Any,
    strategy: str,
    allow_ae: bool,
):
    """Inside ``shard_map``: local streams, the local row slab of ``K`` and everything the step needs."""
    norm, w, dtype = rg._prepare(blocks, layers, coordinates, diffusion, strategy)
    data = [rg._streams(layers, x, res, order, w, allow_ae) for x, res, order in norm]
    nb = len(data)
    sizes = [r.size for r, _, _ in data]  # local rows per block
    bias0 = layers[0][1] is not None
    J0 = [
        rg._first_jacobian(x, Q[0], order, bias0)
        for (x, _, order), (_, _, Q) in zip(norm, data, strict=True)
    ]
    J0_all = [_gather(j, axis) for j in J0]
    ndev = jax.lax.axis_size(axis)
    glob = [s * ndev for s in sizes]
    slab = [[jnp.zeros((sizes[i], glob[j]), dtype) for j in range(nb)] for i in range(nb)]
    for i in range(nb):
        for j in range(nb):
            slab[i][j] = slab[i][j] + J0[i] @ J0_all[j].T
    for li in range(1, len(layers)):
        W, b, _ = layers[li]
        bias = b is not None
        out, inp = jnp.shape(W)
        P_all = [_gather(data[j][1][li - 1], axis) for j in range(nb)]
        Q_all = [_gather(data[j][2][li], axis) for j in range(nb)]
        explicit_all: dict[int, Array] = {}
        for i in range(nb):
            Pi, Qi = data[i][1][li - 1], data[i][2][li]
            for j in range(nb):
                Pj, Qj = P_all[j], Q_all[j]
                pairs = Qi.shape[2] * Qj.shape[2]
                use = strategy == "streams" or (
                    strategy == "auto" and rg._STREAMS_COST * pairs * (inp + out) < inp * out
                )
                if use:
                    G = rg._streams_gram(Pi, Qi, Pj, Qj, False)
                    if bias:
                        G = G + Qi[:, :, 0].reshape(-1, out) @ Qj[:, :, 0].reshape(-1, out).T
                else:
                    if j not in explicit_all:
                        explicit_all[j] = rg._explicit_jacobian(Pj, Qj, bias)
                    G = rg._explicit_jacobian(Pi, Qi, bias) @ explicit_all[j].T
                slab[i][j] = slab[i][j] + G
    return (
        norm,
        data,
        sizes,
        glob,
        ndev,
        jnp.concatenate([jnp.concatenate(slab[i], axis=1) for i in range(nb)], axis=0),
        dtype,
    )


def _global_rows(local: Array, sizes: Sequence[int], ndev: int) -> Array:
    """Rows of per-device slabs ``(ndev, sum(sizes), ...)`` regrouped to block-major global order."""
    out, off = [], 0
    for s in sizes:
        out.append(local[:, off : off + s].reshape((ndev * s,) + local.shape[2:]))
        off += s
    return jnp.concatenate(out, axis=0)


def _specs(blocks: Sequence[Any], axis: str):
    return tuple(PartitionSpec(axis) for _ in blocks)


def distributed_residual_gram(
    blocks: Sequence[Any],
    layers: Sequence[Any],
    mesh: Mesh,
    *,
    axis: str = "data",
    coordinates: Any = None,
    diffusion: Any = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
) -> tuple[Array, Array]:
    """``(r, K)`` of :func:`~omnibias.jax.experimental.deep_residual_gram` with the points of each block sharded over ``axis``.

    Rows are ordered as in the single-device program (by block, then point, then residual component); ``r`` and ``K`` are
    replicated. ``blocks`` are ``(x, residual)`` or ``(x, residual, order)``.
    """
    _check(blocks, mesh, axis)
    residuals = [(blk[1], *blk[2:]) for blk in blocks]

    def build() -> Callable[..., Any]:
        def body(arrays: Any, *xs: Array):
            local_blocks = [(x, *rest) for x, rest in zip(xs, residuals, strict=True)]
            _, data, sizes, _, ndev, slab, _ = _local_gram(
                local_blocks,
                _with_arrays(arrays, layers),
                axis,
                coordinates,
                diffusion,
                strategy,
                allow_almost_everywhere,
            )
            r_local = jnp.concatenate([r.reshape(-1) for r, _, _ in data])
            r = _global_rows(_gather_stack(r_local, axis), sizes, ndev)
            K = _global_rows(_gather_stack(slab, axis), sizes, ndev)
            return r, K

        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(PartitionSpec(), *_specs(blocks, axis)),
                out_specs=(PartitionSpec(), PartitionSpec()),
                check_vma=False,
            )
        )

    key = _key("gram", mesh, axis, layers, blocks, coordinates, diffusion, strategy, allow_almost_everywhere)
    return _program(key, build)(_arrays(layers), *[jnp.asarray(blk[0]) for blk in blocks])


def _gather_stack(a: Array, axis: str) -> Array:
    return jax.lax.all_gather(a, axis, axis=0, tiled=False)


def distributed_residual_gauss_newton(
    blocks: Sequence[Any],
    layers: Sequence[Any],
    damping: float | Array,
    mesh: Mesh,
    *,
    axis: str = "data",
    coordinates: Any = None,
    diffusion: Any = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
    conditioning: ConditioningPolicy | None = None,
) -> tuple[Array, Array, list[tuple[Array, Array | None]]]:
    """``(r, K, step)`` of :func:`~omnibias.jax.experimental.deep_residual_gauss_newton`, points sharded over ``axis``.

    The step is ``J^T (K + damping I)^{-1} r`` as a list of ``(dW, db)``, replicated; the solve runs on the replicated ``K``
    (see the module notes for what is and is not divided by the number of devices).
    """
    _check(blocks, mesh, axis)
    residuals = [(blk[1], *blk[2:]) for blk in blocks]

    def build() -> Callable[..., Any]:
        def body(arrays: Any, damping: Array, *xs: Array):
            lays = _with_arrays(arrays, layers)
            local_blocks = [(x, *rest) for x, rest in zip(xs, residuals, strict=True)]
            norm, data, sizes, glob, ndev, slab, _ = _local_gram(
                local_blocks, lays, axis, coordinates, diffusion, strategy, allow_almost_everywhere
            )
            r_local = jnp.concatenate([r.reshape(-1) for r, _, _ in data])
            r = _global_rows(_gather_stack(r_local, axis), sizes, ndev)
            K = _global_rows(_gather_stack(slab, axis), sizes, ndev)
            alpha, _ = damped_solve(K, r, damping, conditioning)
            me = jax.lax.axis_index(axis)
            offsets = np.cumsum([0] + glob)
            a = []
            for k, (rb, _, _) in enumerate(data):
                block_alpha = alpha[offsets[k] : offsets[k + 1]]
                mine = jax.lax.dynamic_slice_in_dim(block_alpha, me * sizes[k], sizes[k])
                a.append(mine.reshape(rb.shape))
            step: list[tuple[Array, Array | None]] = []
            for li, (_, b, _) in enumerate(lays):
                dW: Array | float = 0.0
                db: Array | float = 0.0
                for ab, (x, _, order), (_, P, Q) in zip(a, norm, data, strict=True):
                    Qa = jnp.einsum("br,brto->bto", ab, Q[li])
                    if li == 0:
                        D = x.shape[1]
                        dW = dW + Qa[:, 0].T @ x
                        if order >= 1:
                            dW = dW + jnp.sum(Qa[:, 1 : D + 1], axis=0).T
                    else:
                        dW = dW + jnp.einsum("bto,bti->oi", Qa, P[li - 1])
                    db = db + jnp.sum(Qa[:, 0], axis=0)
                step.append(
                    (
                        jax.lax.psum(jnp.asarray(dW), axis),
                        None if b is None else jax.lax.psum(jnp.asarray(db), axis),
                    )
                )
            return r, K, step

        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(PartitionSpec(), PartitionSpec(), *_specs(blocks, axis)),
                out_specs=(PartitionSpec(), PartitionSpec(), PartitionSpec()),
                check_vma=False,
            )
        )

    key = _key(
        "gauss_newton", mesh, axis, layers, blocks, coordinates, diffusion, strategy, allow_almost_everywhere, conditioning
    )
    first = jnp.asarray(blocks[0][0])
    return _program(key, build)(
        _arrays(layers), jnp.asarray(damping, first.dtype), *[jnp.asarray(blk[0]) for blk in blocks]
    )


def distributed_residual_ntk_diagonal(
    blocks: Sequence[Any],
    layers: Sequence[Any],
    mesh: Mesh,
    *,
    axis: str = "data",
    coordinates: Any = None,
    diffusion: Any = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
) -> tuple[Array, Array]:
    """``(r, diag K)`` of :func:`~omnibias.jax.experimental.deep_residual_ntk_diagonal`; no communication is needed.

    Row ``b`` of the diagonal depends on point ``b`` only, so each device computes its own rows.
    """
    _check(blocks, mesh, axis)
    residuals = [(blk[1], *blk[2:]) for blk in blocks]
    spec = _specs(blocks, axis)

    def build() -> Callable[..., Any]:
        def body(arrays: Any, *xs: Array):
            rs, ds = [], []
            for x, rest in zip(xs, residuals, strict=True):
                r, d = rg.deep_residual_ntk_diagonal(
                    [(x, *rest)],
                    _with_arrays(arrays, layers),
                    coordinates=coordinates,
                    diffusion=diffusion,
                    strategy=strategy,
                    allow_almost_everywhere=allow_almost_everywhere,
                )
                rs.append(r[None])
                ds.append(d[None])
            return tuple(rs), tuple(ds)

        return jax.jit(
            jax.shard_map(body, mesh=mesh, in_specs=(PartitionSpec(), *spec), out_specs=(spec, spec), check_vma=False)
        )

    key = _key("ntk_diagonal", mesh, axis, layers, blocks, coordinates, diffusion, strategy, allow_almost_everywhere)
    rs, ds = _program(key, build)(_arrays(layers), *[jnp.asarray(blk[0]) for blk in blocks])
    # each output is (n_dev, rows per device of one block): device-major is already the global point order
    return jnp.concatenate([r.reshape(-1) for r in rs]), jnp.concatenate(
        [d.reshape(-1) for d in ds]
    )
