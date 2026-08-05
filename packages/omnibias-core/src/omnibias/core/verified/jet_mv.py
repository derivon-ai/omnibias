# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified (interval) *multivariate* jets -- the certified twin of ``jet_mv``.

This is the rigorous counterpart of the float multi-index kernels in
:mod:`omnibias.torch.jet_mv` / :mod:`omnibias.jax.jet_mv`.  Where the float
kernel returns, for a *point* ``x0``, every mixed Taylor coefficient
``c_alpha = D^alpha u(x0) / alpha!`` in one forward pass, this module returns a
guaranteed :class:`~omnibias.core.verified.interval.Interval` **enclosure** of
those coefficients that is valid for *every* point of an input **box**.

Why this is sound
-----------------
The float ``mlp_jet_mv`` computes ``{D^alpha u(x0)/alpha!}`` exactly via a fixed
sequence of operations (identity jet -> affine -> activation tower -> truncated
Cauchy product).  Replaying the *same* sequence in interval arithmetic, with the
constant row of the input jet seeded as the whole box, makes every intermediate
an inclusion-isotonic interval extension of the corresponding real computation:

* the unit-multi-index rows of :func:`identity_jet` are the *exact* basis
  vectors (``d x_i / d delta_j = delta_ij`` independent of ``x0``);
* :func:`~omnibias.core.verified.sigma.sigma_tower_interval` is inclusion
  isotonic in its argument, so the activation tower expanded about the (interval)
  pre-activation encloses the tower at every concrete ``x0`` in the box;
* interval ``+`` / ``*`` are inclusion isotonic.

By induction, for every ``x0`` in the box and every ``|alpha| <= order`` the real
coefficient ``D^alpha u(x0)/alpha!`` lies in ``jet[alpha]``.  Truncating the
Cauchy product at total order ``order`` is exact for the kept coefficients (the
pre-activation perturbation has a zero constant row, so each extra product raises
the minimum order), so dropping higher terms never loosens a kept coefficient.

The enclosures inherit interval *dependency* overestimation, which grows with box
width and network depth; :func:`certified_partials_subdivided` (and the
``splits`` argument of :func:`certified_residual_bound`) tighten it soundly by
splitting the box and taking the hull -- the union of the per-sub-box ranges is
itself a rigorous enclosure of the whole-box range of each derivative.

Supported activations are exactly those of the verified tower:
``"tanh"``, ``"sigmoid"``, ``"gaussian"``, the trigonometric pair ``"sin"`` /
``"cos"`` (which admits closed-form Fourier-mode / plane-wave fields), and the
smooth closed-form neural activations ``"silu"``, ``"gelu"`` (exact) and
``"softplus"``.

Conventions (identical to the float kernels)
--------------------------------------------
* A *jet* is ``list[list[Interval]]`` of shape ``(M, C)`` with
  ``M = num_multi_indices(dim, order)`` rows (canonical
  :func:`~omnibias.core.multi_index.multi_indices` order) and ``C`` output
  components.  Row ``i`` holds ``D^alpha u(x0)/alpha!`` for the ``i``-th
  multi-index.
* A *layer* is ``(W, b, name)`` with ``W`` shape ``(C_out, C_in)``, ``b`` shape
  ``(C_out,)`` or ``None``, and ``name`` one of the supported activations (or
  ``None`` for a pure affine readout).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction

from omnibias.core.multi_index import (
    MultiIndex,
    index_position,
    multi_index_factorial,
    multi_indices,
    multiply_table,
    num_multi_indices,
)
from omnibias.core.verified.interval import Interval, IntervalLike, hull
from omnibias.core.verified.jet import affine_jet
from omnibias.core.verified.sigma import sigma_tower_interval

#: A multivariate interval jet: ``(M, C)`` interval coefficients (row-major over
#: the canonical multi-index ordering).
MVJet = list[list[Interval]]

#: ``(weight, bias, activation-name-or-None)`` describing one MLP layer.
Layer = tuple[
    Sequence[Sequence[IntervalLike]], Sequence[IntervalLike] | None, str | None
]

#: A per-axis box: one ``Interval`` (or ``(lo, hi)`` / point) per input dimension.
BoxLike = Sequence["Interval | tuple[float, float] | float"]


def _zero() -> Interval:
    return Interval.point(0.0)


def _as_interval_box(box: BoxLike) -> list[Interval]:
    """Promote a per-axis box specification to a list of :class:`Interval`."""
    out: list[Interval] = []
    for entry in box:
        if isinstance(entry, Interval):
            out.append(entry)
        elif isinstance(entry, tuple):
            lo, hi = entry
            out.append(Interval(float(lo), float(hi)))
        else:
            out.append(Interval.point(float(entry)))
    return out


def _compose_scalar_mv(
    u_col: Sequence[Interval], tower: Sequence[Interval], dim: int, order: int
) -> list[Interval]:
    r"""Compose an activation onto one scalar multivariate jet column.

    ``u_col`` are the ``M`` interval coefficients of the (scalar) pre-activation
    jet; ``tower[k]`` encloses ``sigma^(k)(u_col[0])``.  Mirrors the float
    :func:`omnibias.torch.jet_mv.compose_jet_mv` for a single component, with the
    shifted-power identity and the multivariate truncated Cauchy product.
    """
    m = num_multi_indices(dim, order)
    table = multiply_table(dim, order)
    zero = _zero()
    # w = pre-activation minus its constant term (row 0 -> 0).
    w: list[Interval] = [zero] + [u_col[g] for g in range(1, m)]
    p: list[Interval] = [Interval.point(1.0)] + [zero for _ in range(m - 1)]
    result: list[Interval] = [tower[0]] + [zero for _ in range(m - 1)]
    fact = 1
    for k in range(1, order + 1):
        fact *= k
        acc: list[Interval] = [zero for _ in range(m)]
        for g, a, b in table:
            acc[g] = acc[g] + p[a] * w[b]
        p = acc
        dk = tower[k] * Interval.from_rational(Fraction(1, fact))
        for g in range(m):
            result[g] = result[g] + dk * p[g]
    return result


def compose_jet_mv(
    u_jet: Sequence[Sequence[Interval]],
    sigma_tower: Sequence[Sequence[Interval]],
    dim: int,
    order: int,
) -> MVJet:
    """Compose an elementwise activation onto a multivariate jet: ``b = sigma(u)``.

    ``u_jet`` has shape ``(M, C)``; ``sigma_tower`` has shape ``(order + 1, C)``
    with ``sigma_tower[k][c]`` enclosing ``sigma^(k)(u_jet[0][c])``.  Returns the
    jet of ``sigma(u)`` with shape ``(M, C)``.
    """
    m = num_multi_indices(dim, order)
    if len(u_jet) != m:
        raise ValueError(
            f"u_jet has {len(u_jet)} rows but dim={dim}, order={order} requires {m}"
        )
    if len(sigma_tower) != order + 1:
        raise ValueError(
            f"sigma_tower order {len(sigma_tower) - 1} must equal jet order {order}"
        )
    width = len(u_jet[0])
    out: MVJet = [[_zero() for _ in range(width)] for _ in range(m)]
    for c in range(width):
        u_col = [u_jet[g][c] for g in range(m)]
        tower = [sigma_tower[k][c] for k in range(order + 1)]
        composed = _compose_scalar_mv(u_col, tower, dim, order)
        for g in range(m):
            out[g][c] = composed[g]
    return out


def jet_multiply(
    a: Sequence[Sequence[Interval]],
    b: Sequence[Sequence[Interval]],
    dim: int,
    order: int,
) -> MVJet:
    r"""Truncated product of two multivariate interval jets: ``jet(a) * jet(b)``.

    The multivariate Cauchy product
    ``(a b)_gamma = sum_{alpha + beta = gamma} a_alpha b_beta`` truncated at total
    order ``order`` -- the jet-level Leibniz rule.  Columns broadcast: a scalar
    mask of shape ``(M, 1)`` multiplies a vector field ``(M, C)`` component-wise,
    so a hard-constraint ansatz ``u = g + b * net`` stays closed form to arbitrary
    order in the verified arithmetic too.
    """
    m = num_multi_indices(dim, order)
    if len(a) != m or len(b) != m:
        raise ValueError(
            f"both jets need {m} rows for dim={dim}, order={order}; "
            f"got {len(a)} and {len(b)}"
        )
    ca = len(a[0])
    cb = len(b[0])
    if ca != cb and ca != 1 and cb != 1:
        raise ValueError(f"incompatible column counts {ca} and {cb} (need equal or 1)")
    width = max(ca, cb)
    table = multiply_table(dim, order)
    out: MVJet = [[_zero() for _ in range(width)] for _ in range(m)]
    for g, al, be in table:
        for col in range(width):
            av = a[al][col if ca > 1 else 0]
            bv = b[be][col if cb > 1 else 0]
            out[g][col] = out[g][col] + av * bv
    return out


def layer_jet_mv(
    z_jet: Sequence[Sequence[Interval]],
    weight: Sequence[Sequence[IntervalLike]],
    bias: Sequence[IntervalLike] | None,
    name: str,
    dim: int,
    order: int,
) -> MVJet:
    """Push a multivariate jet through one ``sigma(W z + b)`` layer (interval)."""
    u_jet = affine_jet(z_jet, weight, bias)
    m = num_multi_indices(dim, order)
    width = len(u_jet[0]) if u_jet else 0
    out: MVJet = [[_zero() for _ in range(width)] for _ in range(m)]
    for c in range(width):
        u_col = [u_jet[g][c] for g in range(m)]
        tower = list(sigma_tower_interval(name, u_col[0], order))
        composed = _compose_scalar_mv(u_col, tower, dim, order)
        for g in range(m):
            out[g][c] = composed[g]
    return out


def identity_jet(box: BoxLike, order: int) -> MVJet:
    """Multivariate jet of the identity map ``x(delta) = x0 + delta`` over a box.

    Returns shape ``(M, D)``: row 0 holds the per-axis box intervals and each
    unit-multi-index row ``e_i`` is the exact basis vector ``e_i``; all other rows
    are zero.  Seeding row 0 with the whole box is what makes the propagated jet a
    valid enclosure for every point of the box (see the module docstring).
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    x0 = _as_interval_box(box)
    dim = len(x0)
    if dim < 1:
        raise ValueError("box must have at least one axis")
    pos = index_position(dim, order)
    m = num_multi_indices(dim, order)
    rows: MVJet = [[_zero() for _ in range(dim)] for _ in range(m)]
    zero_idx = (0,) * dim
    for i in range(dim):
        rows[pos[zero_idx]][i] = x0[i]
    if order >= 1:
        for i in range(dim):
            e = tuple(1 if j == i else 0 for j in range(dim))
            rows[pos[e]][i] = Interval.point(1.0)
    return rows


def mlp_jet_mv(box: BoxLike, layers: Sequence[Layer], order: int) -> MVJet:
    """Certified multivariate Taylor jet of a deep MLP over an input ``box``.

    ``layers`` is a sequence of ``(W, b, name)``; ``name=None`` is a pure affine
    readout.  Returns a jet of shape ``(M, C)`` whose row ``i`` *encloses*
    ``D^alpha u(x0) / alpha!`` for ``alpha = multi_indices(D, order)[i]`` and every
    ``x0`` in the box.  Use :func:`jet_partials`, :func:`jet_gradient`,
    :func:`jet_hessian` or :func:`jet_laplacian` to read out raw derivatives.
    """
    x0 = _as_interval_box(box)
    dim = len(x0)
    jet: MVJet = identity_jet(box, order)
    for weight, bias, name in layers:
        if name is None:
            jet = affine_jet(jet, weight, bias)
        else:
            jet = layer_jet_mv(jet, weight, bias, name, dim, order)
    return jet


# --------------------------------------------------------------------------- #
# Readouts (raw derivative enclosures).
# --------------------------------------------------------------------------- #
def jet_partials(
    jet: Sequence[Sequence[Interval]], dim: int, order: int
) -> dict[MultiIndex, list[Interval]]:
    """Raw partial-derivative enclosures ``{alpha: D^alpha u}`` from a jet.

    Each coefficient row is scaled by ``alpha!`` (``D^alpha u = alpha! c_alpha``).
    """
    idx = multi_indices(dim, order)
    if len(jet) != len(idx):
        raise ValueError(
            f"jet has {len(jet)} rows but dim={dim}, order={order} "
            f"requires {len(idx)}"
        )
    out: dict[MultiIndex, list[Interval]] = {}
    for i, alpha in enumerate(idx):
        f = multi_index_factorial(alpha)
        if f == 1:  # avoid a needless [1, 1] multiply that would inflate by 1 ulp
            out[alpha] = list(jet[i])
        else:
            fact = Interval.from_rational(f)
            out[alpha] = [c * fact for c in jet[i]]
    return out


def jet_gradient(
    jet: Sequence[Sequence[Interval]], dim: int, order: int
) -> list[list[Interval]]:
    """Gradient enclosure ``d u / d x_i``, shape ``(D, C)``. Needs ``order >= 1``."""
    if order < 1:
        raise ValueError(f"gradient needs order >= 1, got {order}")
    pos = index_position(dim, order)
    rows: list[list[Interval]] = []
    for i in range(dim):
        e = tuple(1 if j == i else 0 for j in range(dim))
        rows.append(list(jet[pos[e]]))  # alpha! = 1 for a unit multi-index
    return rows


def jet_hessian(
    jet: Sequence[Sequence[Interval]], dim: int, order: int
) -> list[list[list[Interval]]]:
    """Hessian enclosure ``d^2 u / d x_i d x_j``, shape ``(D, D, C)``. Needs ``order >= 2``."""
    if order < 2:
        raise ValueError(f"hessian needs order >= 2, got {order}")
    pos = index_position(dim, order)
    out: list[list[list[Interval]]] = []
    for i in range(dim):
        row: list[list[Interval]] = []
        for j in range(dim):
            alpha = tuple(
                (1 if k == i else 0) + (1 if k == j else 0) for k in range(dim)
            )
            f = multi_index_factorial(alpha)
            if f == 1:  # off-diagonal mixed partial: alpha! = 1, keep it tight
                row.append(list(jet[pos[alpha]]))
            else:
                fact = Interval.from_rational(f)
                row.append([c * fact for c in jet[pos[alpha]]])
        out.append(row)
    return out


def jet_laplacian(
    jet: Sequence[Sequence[Interval]], dim: int, order: int
) -> list[Interval]:
    r"""Laplacian enclosure ``sum_i d^2 u / d x_i^2``, shape ``(C,)``. Needs ``order >= 2``.

    Reads the pure second-order rows directly: ``D^{2 e_i} u = 2! * c_{2 e_i}``.
    """
    if order < 2:
        raise ValueError(f"laplacian needs order >= 2, got {order}")
    pos = index_position(dim, order)
    width = len(jet[0])
    two = Interval.point(2.0)
    acc = [_zero() for _ in range(width)]
    for i in range(dim):
        alpha = tuple(2 if j == i else 0 for j in range(dim))
        row = jet[pos[alpha]]
        for c in range(width):
            acc[c] = acc[c] + two * row[c]
    return acc


# --------------------------------------------------------------------------- #
# Box-level convenience: certified derivative enclosures and PDE residuals.
# --------------------------------------------------------------------------- #
def certified_partials(
    box: BoxLike, layers: Sequence[Layer], order: int
) -> dict[MultiIndex, list[Interval]]:
    """All certified partial-derivative enclosures ``{alpha: D^alpha u}`` over the box."""
    x0 = _as_interval_box(box)
    dim = len(x0)
    return jet_partials(mlp_jet_mv(box, layers, order), dim, order)


def _subboxes(box: list[Interval], splits: Sequence[int]) -> list[list[Interval]]:
    """Cartesian grid of sub-boxes: ``splits[i]`` equal pieces along axis ``i``."""
    axes: list[list[Interval]] = []
    for iv, s in zip(box, splits, strict=True):
        if s < 1:
            raise ValueError("splits per axis must be >= 1")
        lo, hi = iv.lo, iv.hi
        step = (hi - lo) / s
        pieces = [
            Interval(lo + k * step, hi if k == s - 1 else lo + (k + 1) * step)
            for k in range(s)
        ]
        axes.append(pieces)
    result: list[list[Interval]] = [[]]
    for pieces in axes:
        result = [prefix + [piece] for prefix in result for piece in pieces]
    return result


def _normalize_splits(splits: int | Sequence[int], dim: int) -> list[int]:
    if isinstance(splits, int):
        return [splits] * dim
    out = list(splits)
    if len(out) != dim:
        raise ValueError(f"splits must have one entry per axis ({dim}), got {len(out)}")
    return out


def certified_partials_subdivided(
    box: BoxLike,
    layers: Sequence[Layer],
    order: int,
    splits: int | Sequence[int] = 1,
) -> dict[MultiIndex, list[Interval]]:
    """Tightened certified partials: hull of per-sub-box enclosures over a grid.

    Splitting the box into ``splits`` equal pieces per axis and taking the hull of
    each derivative enclosure is sound (the union of the per-sub-box ranges
    encloses the whole-box range) and reduces interval dependency overestimation.
    """
    x0 = _as_interval_box(box)
    dim = len(x0)
    per_axis = _normalize_splits(splits, dim)
    boxes = _subboxes(x0, per_axis)
    accum: dict[MultiIndex, list[Interval]] | None = None
    for sub in boxes:
        part = certified_partials(sub, layers, order)
        if accum is None:
            accum = {a: list(v) for a, v in part.items()}
        else:
            for a, v in part.items():
                accum[a] = [hull([accum[a][c], v[c]]) for c in range(len(v))]
    assert accum is not None  # at least one sub-box always exists
    return accum


def certified_residual_bound(
    box: BoxLike,
    layers: Sequence[Layer],
    order: int,
    residual: Callable[[dict[MultiIndex, list[Interval]]], Interval],
    splits: int | Sequence[int] = 1,
) -> Interval:
    r"""Rigorous enclosure of a PDE residual over the box (sup-norm via ``.mag``).

    ``residual`` maps a partial-derivative dictionary (as produced by
    :func:`certified_partials`) to a scalar :class:`Interval` -- e.g. for the
    Laplace equation ``lambda P: P[(2, 0)][0] + P[(0, 2)][0]``.  The returned
    interval encloses ``{ residual(x) : x in box }``; a certified sup-norm bound on
    the residual is ``certified_residual_bound(...).mag``.  Increasing ``splits``
    tightens the enclosure (hull over the sub-box grid) and is the standard way to
    drive the certified residual below a target before invoking the
    Newton-Kantorovich / radii-polynomial closure.
    """
    x0 = _as_interval_box(box)
    dim = len(x0)
    per_axis = _normalize_splits(splits, dim)
    boxes = _subboxes(x0, per_axis)
    pieces: list[Interval] = []
    for sub in boxes:
        pieces.append(residual(certified_partials(sub, layers, order)))
    return hull(list(pieces))


__all__ = [
    "BoxLike",
    "Layer",
    "MVJet",
    "certified_partials",
    "certified_partials_subdivided",
    "certified_residual_bound",
    "compose_jet_mv",
    "identity_jet",
    "jet_gradient",
    "jet_hessian",
    "jet_laplacian",
    "jet_multiply",
    "jet_partials",
    "layer_jet_mv",
    "mlp_jet_mv",
]
