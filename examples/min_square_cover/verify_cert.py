# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigorous continuous coverage certificate via ``omnibias-verify`` (the H5 register).

The integer :func:`~examples.min_square_cover.certify.verify_cover` proves the discrete cover
hits every 1-*pixel*; this module proves a strictly stronger, *continuous* statement about the
same placed squares: their **soft-OR coverage field** stays above a threshold over the entire
``+/- delta`` cell around every 1-pixel, i.e. the cover tolerates a ``delta``-pixel placement /
sampling error. Soundness comes from :mod:`omnibias.core.verified` interval arithmetic:

* each square's soft occupancy is a separable product of ``soft_interval`` factors -- a
  difference of two rigorous :func:`~omnibias.core.verified.transcend.sigmoid_iv` enclosures;
* the union is the multilinear soft-OR ``C = 1 - prod_k (1 - occ_k)``;
* :func:`omnibias.verify.certified_minimize` (interval branch-and-bound) rigorously encloses
  ``min_{x in cell} C(x)`` -- the returned ``f_lower`` is an unconditional lower bound.

Because dropping a square only lowers ``C`` (the product moves toward 1), evaluating the union
over the squares *near* a pixel is a sound under-estimate of the true coverage: certifying the
subset certifies the whole cover.

The whole module imports ``omnibias-verify`` lazily; :func:`certify_cover_robustness` returns
``None`` if it (or ``torch``) is unavailable, exactly like the LP register.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class RobustCoverageCertificate:
    """A rigorous lower bound on the soft-OR coverage of a discrete cover over 1-pixel cells."""

    certified_min_coverage: float  # rigorous inf over all 1-pixel +/-delta cells of soft-OR C
    threshold: float  # coverage level the cover is certified to exceed
    delta: float  # cell half-width (robustness radius, in pixels) the certificate holds over
    beta: float  # sharpness of the soft occupancy the statement is about
    robust: bool  # certified_min_coverage >= threshold
    worst_pixel: tuple[int, int]  # the 1-pixel achieving the minimum certified coverage
    n_pixels: int  # number of 1-pixels certified
    all_converged: bool  # every per-pixel branch-and-bound closed its gap to tol


def _coverage_interval_fn(
    centers: list[tuple[float, float]], side: float, beta: float
):  # -> ObjectiveFn
    """Build the interval extension ``box -> Interval`` of the soft-OR coverage of ``centers``."""
    from omnibias.core.verified.interval import Interval
    from omnibias.core.verified.transcend import sigmoid_iv

    half = 0.5 * side
    one = Interval.point(1.0)

    def soft_interval_iv(x: Interval, c: float) -> Interval:
        # soft_box axis factor: sigmoid(beta (x - c + s/2)) - sigmoid(beta (x - c - s/2)).
        return sigmoid_iv(beta * (x - c + half)) - sigmoid_iv(beta * (x - c - half))

    def coverage(box: tuple[Interval, ...]) -> Interval:
        x, y = box[0], box[1]
        product = one
        for cx, cy in centers:
            occ = soft_interval_iv(x, cx) * soft_interval_iv(y, cy)
            product = product * (one - occ)
        return one - product

    return coverage


def certify_cover_robustness(
    image: Tensor,
    squares: list[tuple[int, int]],
    side: int,
    *,
    beta: float = 4.0,
    threshold: float = 0.5,
    delta: float = 0.0,
    tol: float = 1e-4,
    max_boxes: int = 4000,
    neighbourhood: float | None = None,
) -> RobustCoverageCertificate | None:
    r"""Certify that the soft-OR coverage of ``squares`` exceeds ``threshold`` over every cell.

    For each 1-pixel ``(i, j)`` the soft-OR coverage of the placed squares (centers
    ``(r + side/2, c + side/2)``, sharpness ``beta``) is rigorously minimised over the box
    ``[i-delta, i+delta] x [j-delta, j+delta]`` with :func:`omnibias.verify.certified_minimize`.
    The certificate's :attr:`~RobustCoverageCertificate.certified_min_coverage` is the smallest
    such rigorous lower bound over all 1-pixels; when it is ``>= threshold`` the cover is
    certified robust to a ``delta``-pixel placement error. Only squares within ``neighbourhood``
    Chebyshev distance of a pixel enter that pixel's union (a sound under-estimate -- dropping
    squares only lowers coverage); ``neighbourhood`` defaults to ``side`` (beyond it the soft
    occupancy is negligible). Returns ``None`` if ``omnibias-verify`` is unavailable.
    """
    try:
        from omnibias.verify import certified_minimize
    except ImportError:
        return None

    from omnibias.core.verified.interval import Interval

    # A square at top-left (r, c) covers integer pixels r..r+side-1, so its soft box is centered
    # on that pixel centroid r+(side-1)/2 (box spans the cells [r-0.5, r+side-0.5]); the sigmoid
    # transitions then land between pixels, not on the boundary pixel itself.
    offset = (side - 1) / 2.0
    reach = float(side if neighbourhood is None else neighbourhood)
    centers = [(r + offset, c + offset) for r, c in squares]
    ones = image.to(bool).nonzero(as_tuple=False).tolist()

    certified_min = float("inf")
    worst: tuple[int, int] = (-1, -1)
    all_converged = True
    for i, j in ones:
        near = [
            (cx, cy) for cx, cy in centers
            if max(abs(i - cx), abs(j - cy)) <= reach + delta
        ]
        if not near:
            certified_min = 0.0  # an uncovered pixel: soft-OR coverage is exactly 0 there
            worst = (int(i), int(j))
            continue
        cov = _coverage_interval_fn(near, float(side), beta)
        box = [
            Interval(float(i) - delta, float(i) + delta),
            Interval(float(j) - delta, float(j) + delta),
        ]
        res = certified_minimize(cov, box, tol=tol, max_boxes=max_boxes)
        all_converged = all_converged and res.converged
        if res.f_lower < certified_min:
            certified_min = res.f_lower
            worst = (int(i), int(j))

    return RobustCoverageCertificate(
        certified_min_coverage=certified_min,
        threshold=threshold,
        delta=delta,
        beta=beta,
        robust=certified_min >= threshold,
        worst_pixel=worst,
        n_pixels=len(ones),
        all_converged=all_converged,
    )


__all__ = ["RobustCoverageCertificate", "certify_cover_robustness"]
