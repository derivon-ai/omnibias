# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Closed-form-jet warm-start seeds for the certified minimiser (jax).

Bit-identical twin of :mod:`omnibias.verify.torch.warm_start`: a few steps of
projected gradient descent using the exact closed-form jet gradient
:func:`omnibias.jax.jet_mv.jet_gradient` locate a good point inside the box, which
is then fed to :func:`~omnibias.verify.certified_network_minimize` as ``seeds=``.

Search with the fast differentiable jet; *prove* with the verified jet.  A seed can
only lower the incumbent ``f_upper``; it never affects the sound enclosure.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from omnibias.verify._core.global_opt import GlobalMinResult
from omnibias.verify._core.net_global_opt import certified_network_minimize


def _coerce_layers(net_or_layers: Any) -> list[Any]:
    """Accept a ``JetMLP``-like net (via ``_layer_specs``) or a raw layer list."""
    if hasattr(net_or_layers, "_layer_specs"):
        return list(net_or_layers._layer_specs())
    return list(net_or_layers)


def descent_seeds(
    net_or_layers: Any,
    box: Sequence[tuple[float, float]],
    *,
    component: int = 0,
    order: int = 1,
    starts: int = 8,
    steps: int = 100,
    lr: float = 0.1,
    seed: int = 0,
) -> list[tuple[float, ...]]:
    r"""Closed-form gradient-descent warm-start points inside ``box`` (jax).

    Runs ``starts`` projected gradient-descent trajectories (the box centre plus
    ``starts - 1`` random starts) on ``net(.)[component]`` using the exact
    closed-form jet gradient :func:`omnibias.jax.jet_mv.jet_gradient`, clamping to
    the box after every step.  Returns the trajectory endpoints as plain float
    tuples, ready to pass as ``seeds=`` to
    :func:`~omnibias.verify.certified_network_minimize`.  A pure accelerator: a bad
    seed simply fails to lower the incumbent and is discarded.
    """
    import jax.numpy as jnp
    import numpy as np
    from omnibias.jax.jet_mv import jet_gradient, mlp_jet_mv

    raw = _coerce_layers(net_or_layers)
    dim = len(box)
    if dim < 1:
        raise ValueError("box must have at least one axis")
    if order < 1:
        raise ValueError(f"order must be >= 1 for the gradient jet, got {order}")
    if not raw:
        raise ValueError("layer list is empty")

    first_w = raw[0][0]
    dtype = getattr(first_w, "dtype", None) or jnp.asarray(0.0).dtype

    def _as_a(v: Any) -> Any:
        return v if hasattr(v, "dtype") else jnp.asarray(v, dtype=dtype)

    layers = [(_as_a(w), None if b is None else _as_a(b), spec) for w, b, spec in raw]
    lo = jnp.asarray([float(b[0]) for b in box], dtype=dtype)
    hi = jnp.asarray([float(b[1]) for b in box], dtype=dtype)
    rng = np.random.default_rng(seed)
    points: list[Any] = [0.5 * (lo + hi)]
    for _ in range(max(starts - 1, 0)):
        u = jnp.asarray(rng.random(dim), dtype=dtype)
        points.append(lo + u * (hi - lo))

    seeds: list[tuple[float, ...]] = []
    for start in points:
        x = start
        for _ in range(steps):
            jet = mlp_jet_mv(x, layers, order)
            grad = jet_gradient(jet, dim, order)[:, component]
            x = jnp.clip(x - lr * grad, lo, hi)
        seeds.append(tuple(float(v) for v in x))
    return seeds


def warm_started_network_minimize(
    net_or_layers: Any,
    box: Sequence[tuple[float, float]],
    *,
    component: int = 0,
    order: int = 1,
    tol: float = 1e-6,
    max_boxes: int = 100_000,
    use_gradient: bool = True,
    second_order: bool = False,
    use_newton: bool = True,
    min_width: float = 1e-12,
    starts: int = 8,
    steps: int = 100,
    lr: float = 0.1,
    seed: int = 0,
) -> GlobalMinResult:
    r"""Certified network minimisation warm-started by the closed-form jet gradient.

    Convenience wrapper: build :func:`descent_seeds` with the differentiable jax
    jet, then hand them to :func:`~omnibias.verify.certified_network_minimize`.  The
    certified enclosure is identical to the un-seeded call (seeds only prune the
    search); ``f_lower <= min net <= f_upper`` holds unconditionally.
    """
    seeds = descent_seeds(
        net_or_layers,
        box,
        component=component,
        order=1,
        starts=starts,
        steps=steps,
        lr=lr,
        seed=seed,
    )
    return certified_network_minimize(
        net_or_layers,
        box,
        order=order,
        component=component,
        tol=tol,
        max_boxes=max_boxes,
        use_gradient=use_gradient,
        second_order=second_order,
        use_newton=use_newton,
        min_width=min_width,
        seeds=seeds,
    )


__all__ = ["descent_seeds", "warm_started_network_minimize"]
