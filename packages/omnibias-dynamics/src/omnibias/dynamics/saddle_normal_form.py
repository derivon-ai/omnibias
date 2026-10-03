# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Exact-Q Poincare-Dulac normalization at a rational-eigenvalue hyperbolic saddle.

Given a checked :class:`~omnibias.dynamics.graphic.HyperbolicSaddle` with
**exact rational** eigenvalues ``lambda1 < 0 < lambda2``,
:func:`diagonalize_saddle` finds the exact rational eigenvector basis and
re-expresses the field as ``xi' = lambda1*xi + p(xi,eta)``,
``eta' = eta*lambda2 + q(xi,eta)`` with ``p, q`` starting at total degree
``>= 2``. :func:`resonant_normal_form` then removes every *non-resonant*
monomial degree by degree, exactly: at homogeneous degree ``k`` a monomial
``xi^i eta^j`` in the ``xi``-equation has eigenvalue
``i*lambda1 + j*lambda2 - lambda1`` under the linearized flow generator, and
symmetrically ``- lambda2`` in the ``eta``-equation. That eigenvalue is a
single rational number; when it is nonzero the monomial is cleared *exactly*
by one division (a near-identity coordinate change), and when it is exactly
zero the monomial is **resonant** and cannot be removed by any polynomial
change of variables -- it is exactly where a Dulac ``x^r log x`` term comes
from. :func:`dulac_corner_expansion` does not read the resonant monomials off
by hand, though: it derives the corner map's leading correction via the
classical conserved-quantity (``H = xi**lambda2 * eta**mu1``) perturbation
argument -- the linear flow's invariant ``H`` is perturbed to first order by
exactly the surviving resonant terms of ``p_star``/``q_star``, and the
resonance condition forces every contribution onto a *single* ``x``-exponent
(see :func:`_k_series`), giving ``y = x^r - K(x)*log(x)*x^r`` truncated at
plain first order (never exponentiated -- an exactly solvable one-term check
documented on :func:`dulac_corner_expansion` shows resumming into
``exp(-K(x)*log(x))`` would silently claim a second-order accuracy this
argument does not have). A log term appears in the expansion
*automatically*, from that first-order integration, never declared and
never assumed a priori to line up with the normal form's resonant
monomials. :func:`derive_return_map` then composes that corner expansion
with a certified regular-arc return Jacobian from
:func:`omnibias.dynamics.return_maps.certify_stopped_event`, landing on
:class:`~omnibias.dynamics.dulac.RationalInterval` coefficients (the arc
Jacobian is an interval enclosure, not exact) via the new
:class:`DerivedDulacExpansion` wrapper -- kept separate from
:class:`~omnibias.dynamics.dulac.DulacExpansion` itself so every existing
digest and sealed test stays byte-identical.

The full nonlinear pushforward (not merely the leading-order correction) is
carried out at each degree via :meth:`~omnibias.holonomic._core.poly_n.PolyN.compose`,
so the recursion is exact through the declared truncation order, and
:func:`certify_resonant_normal_form` seals a **truncated conjugacy identity**
(one ``polynomial_identity_q`` payload) that :func:`verify_resonant_normal_form`
replays by plain polynomial arithmetic, not by re-running the recursion.

**Scope, read before quoting a resonance:** this module requires *exact
rational* eigenvalues (the graphic's ``stable``/``unstable`` intervals must
be exact points); an irrational ratio raises :class:`ValueError` rather than
drifting into a numeric approximation -- the existing ``power_compensator``
interval route in :mod:`omnibias.dynamics.dulac` remains the tool for that
case. This module proves a **truncated formal** conjugacy on a declared
polynomial vector field. It does not prove that the singular Dulac map of the
*original analytic* field near the saddle equals this truncated normal form
beyond the stated order, and it does not by itself prove finite cyclicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.dynamics.compactify import PlanarPolynomialField
from omnibias.dynamics.dulac import DulacExpansion, DulacTerm, RationalInterval
from omnibias.dynamics.graphic import HyperbolicSaddle
from omnibias.dynamics.return_maps import StoppedEventResult
from omnibias.holonomic._core.poly_n import PolyN

__all__ = [
    "DerivedDulacExpansion",
    "DiagonalSaddleField",
    "ResonantNormalForm",
    "ResonantNormalFormCertificate",
    "ResonantNormalFormFormalVerification",
    "certify_resonant_normal_form",
    "derive_return_map",
    "diagonalize_saddle",
    "dulac_corner_expansion",
    "resonant_normal_form",
    "verify_resonant_normal_form",
    "verify_resonant_normal_form_formally",
]


def _q(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _sparse_to_polyn(poly: SparsePolynomial) -> PolyN:
    return PolyN(poly.nvars, dict(poly.terms))


def _truncate(poly: PolyN, order: int) -> PolyN:
    return PolyN(poly.nvars, {mon: c for mon, c in poly.terms.items() if sum(mon) <= order})


def _polyn_series_reciprocal(eps: PolyN, order: int) -> PolyN:
    """Exact ``1/(1+eps)`` truncated to ``order``, for ``eps`` with zero constant term."""
    if eps.terms.get((0, 0), Fraction(0)) != 0:
        raise ValueError("series_reciprocal requires a zero constant term")
    total = PolyN.const(eps.nvars, 1)
    if eps.is_zero():
        return total
    min_degree = min(sum(mon) for mon in eps.terms)
    power = PolyN.const(eps.nvars, 1)
    steps = 0
    while steps * min_degree <= order and not power.is_zero():
        steps += 1
        power = _truncate(power * (-eps), order)
        total = total + power
    return total


@dataclass(frozen=True)
class DiagonalSaddleField:
    """A field re-expressed at a hyperbolic saddle in exact eigen-coordinates.

    ``xi' = lambda1*xi + p(xi,eta)``, ``eta' = lambda2*eta + q(xi,eta)``,
    ``lambda1 < 0 < lambda2`` exact rationals, ``p, q`` starting at total
    (xi, eta) degree ``>= 2``. ``basis`` is the exact change-of-basis matrix
    ``((m00, m01), (m10, m11))`` (columns are eigenvectors: ``(x,y) = origin
    + M @ (xi, eta)``) and ``origin`` is the saddle point in the original
    coordinates.
    """

    lambda1: Fraction
    lambda2: Fraction
    p: PolyN
    q: PolyN
    basis: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    origin: tuple[Fraction, Fraction]

    def __post_init__(self) -> None:
        if not (self.lambda1 < 0 < self.lambda2):
            raise ValueError("diagonal saddle eigenvalues must satisfy lambda1 < 0 < lambda2")
        for name, poly in (("p", self.p), ("q", self.q)):
            if any(sum(mon) < 2 for mon in poly.terms):
                raise ValueError(f"{name} must start at total degree >= 2")


def _eigenvector(a: Fraction, b: Fraction, c: Fraction, d: Fraction) -> tuple[Fraction, Fraction]:
    """A nonzero kernel vector of ``[[a, b], [c, d]]`` (a 2x2 exactly singular matrix)."""
    if b != 0:
        return (Fraction(1), -a / b)
    if a != 0:
        # b == 0 and a != 0: row 1 forces v0 == 0; singularity (det == a*d == 0)
        # then forces d == 0, so row 2 is automatically satisfied by v1 free.
        return (Fraction(0), Fraction(1))
    if c != 0:
        return (-d / c, Fraction(1))
    # a == b == c == 0: the matrix is at most rank-1 via d alone; (1, 0) is a
    # kernel vector whenever d != 0, and trivially so when the matrix is zero.
    return (Fraction(1), Fraction(0))


def diagonalize_saddle(
    field: PlanarPolynomialField,
    saddle: HyperbolicSaddle,
) -> DiagonalSaddleField:
    """Exact eigen-coordinates at a checked hyperbolic saddle with rational eigenvalues.

    Raises :class:`ValueError` if ``saddle`` does not have exact (point)
    eigenvalue intervals, or if ``saddle`` does not verify against ``field``.
    """
    if not (saddle.stable.exact and saddle.unstable.exact):
        raise ValueError("diagonalize_saddle requires exact rational eigenvalues")
    if not saddle.verifies(field):
        raise ValueError("the declared saddle does not verify against the field")
    lambda1, lambda2 = saddle.stable.lo, saddle.unstable.lo
    x0, y0 = saddle.point
    p_x = field.p.derivative(0).evaluate((x0, y0))
    p_y = field.p.derivative(1).evaluate((x0, y0))
    q_x = field.q.derivative(0).evaluate((x0, y0))
    q_y = field.q.derivative(1).evaluate((x0, y0))
    m00, m10 = _eigenvector(p_x - lambda1, p_y, q_x, q_y - lambda1)
    m01, m11 = _eigenvector(p_x - lambda2, p_y, q_x, q_y - lambda2)
    det = m00 * m11 - m01 * m10
    if det == 0:
        raise ArithmeticError("the two eigenvectors are exactly parallel")
    inv00, inv01 = m11 / det, -m01 / det
    inv10, inv11 = -m10 / det, m00 / det

    xi, eta = PolyN.var(2, 0), PolyN.var(2, 1)
    x_expr = PolyN.const(2, x0) + xi * m00 + eta * m01
    y_expr = PolyN.const(2, y0) + xi * m10 + eta * m11
    p_full = _sparse_to_polyn(field.p).compose([x_expr, y_expr])
    q_full = _sparse_to_polyn(field.q).compose([x_expr, y_expr])
    xi_dot = p_full * inv00 + q_full * inv01
    eta_dot = p_full * inv10 + q_full * inv11
    p_nl = xi_dot - xi * lambda1
    q_nl = eta_dot - eta * lambda2
    for name, poly in (("xi", p_nl), ("eta", q_nl)):
        if any(sum(mon) <= 1 for mon in poly.terms):
            raise ArithmeticError(f"diagonalization failed to cancel the {name} linear part exactly")
    return DiagonalSaddleField(
        lambda1,
        lambda2,
        p_nl,
        q_nl,
        ((m00, m01), (m10, m11)),
        (x0, y0),
    )


@dataclass(frozen=True)
class ResonantNormalForm:
    """A truncated Poincare-Dulac normal form at a hyperbolic saddle.

    ``p_star, q_star`` hold only the **resonant** monomials through
    ``order`` (every non-resonant monomial has been cleared exactly).
    ``transform_x, transform_y`` express the *original* diagonal coordinates
    ``(xi, eta)`` as exact polynomials of the new normal-form coordinates
    ``(u, v)`` (variables ``0, 1``), truncated to ``order``.
    ``resonant_monomials`` lists ``(component, (i, j))`` with ``component``
    ``0`` for the ``xi``-equation (``p_star``) and ``1`` for the
    ``eta``-equation (``q_star``).
    """

    diagonal: DiagonalSaddleField
    order: int
    p_star: PolyN
    q_star: PolyN
    transform_x: PolyN
    transform_y: PolyN
    resonant_monomials: tuple[tuple[int, tuple[int, int]], ...]


def resonant_normal_form(diagonal: DiagonalSaddleField, *, order: int) -> ResonantNormalForm:
    """Exact degree-by-degree Poincare-Dulac normalization through ``order``."""
    if type(order) is not int or order < 2:
        raise ValueError("order must be an integer >= 2")
    lambda1, lambda2 = diagonal.lambda1, diagonal.lambda2
    u, v = PolyN.var(2, 0), PolyN.var(2, 1)
    p_cur = _truncate(diagonal.p, order)
    q_cur = _truncate(diagonal.q, order)
    resonant: list[tuple[int, tuple[int, int]]] = []
    corrections: list[tuple[int, PolyN, PolyN]] = []

    for k in range(2, order + 1):
        p_k = p_cur.homogeneous_part(k)
        q_k = q_cur.homogeneous_part(k)
        hx_terms: dict[tuple[int, int], Fraction] = {}
        hy_terms: dict[tuple[int, int], Fraction] = {}
        for i in range(k + 1):
            j = k - i
            base = i * lambda1 + j * lambda2
            denom1 = base - lambda1
            c1 = p_k.terms.get((i, j), Fraction(0))
            if c1 != 0:
                if denom1 != 0:
                    hx_terms[(i, j)] = c1 / denom1
                else:
                    resonant.append((0, (i, j)))
            denom2 = base - lambda2
            c2 = q_k.terms.get((i, j), Fraction(0))
            if c2 != 0:
                if denom2 != 0:
                    hy_terms[(i, j)] = c2 / denom2
                else:
                    resonant.append((1, (i, j)))
        hx_k = PolyN(2, hx_terms)
        hy_k = PolyN(2, hy_terms)
        corrections.append((k, hx_k, hy_k))
        if hx_k.is_zero() and hy_k.is_zero():
            continue
        big_x = u + hx_k
        big_y = v + hy_k
        p_sub = _truncate(p_cur.compose([big_x, big_y]), order)
        q_sub = _truncate(q_cur.compose([big_x, big_y]), order)
        rhs_xi = _truncate(big_x * lambda1 + p_sub, order)
        rhs_eta = _truncate(big_y * lambda2 + q_sub, order)
        a11 = PolyN.const(2, 1) + hx_k.partial(0)
        a12 = hx_k.partial(1)
        a21 = hy_k.partial(0)
        a22 = PolyN.const(2, 1) + hy_k.partial(1)
        det = _truncate(a11 * a22 - a12 * a21, order)
        inv_det = _polyn_series_reciprocal(det - PolyN.const(2, 1), order)
        du_dt = _truncate((a22 * rhs_xi - a12 * rhs_eta) * inv_det, order)
        dv_dt = _truncate((a11 * rhs_eta - a21 * rhs_xi) * inv_det, order)
        p_cur = _truncate(du_dt - u * lambda1, order)
        q_cur = _truncate(dv_dt - v * lambda2, order)

    transform_x, transform_y = u, v
    for _k, hx_k, hy_k in reversed(corrections):
        if hx_k.is_zero() and hy_k.is_zero():
            continue
        new_x = _truncate(transform_x + hx_k.compose([transform_x, transform_y]), order)
        new_y = _truncate(transform_y + hy_k.compose([transform_x, transform_y]), order)
        transform_x, transform_y = new_x, new_y

    return ResonantNormalForm(
        diagonal,
        order,
        p_cur,
        q_cur,
        transform_x,
        transform_y,
        tuple(sorted(set(resonant))),
    )


def _conjugacy_rows(normal_form: ResonantNormalForm) -> list[dict[str, object]]:
    """Flat rational rows replaying the truncated conjugacy identity.

    Checks, for both components, that the pushforward of the diagonal field
    through ``(transform_x, transform_y)`` equals the Jacobian of the
    transform applied to the normal-form field, truncated to ``order``. This
    is the exact chain-rule identity
    ``D(transform) @ (lambda*u + normal_form) == diagonal(transform)``.
    """
    order = normal_form.order
    diag = normal_form.diagonal
    tx, ty = normal_form.transform_x, normal_form.transform_y
    lhs_x = _truncate(diag.p.compose([tx, ty]) + tx * diag.lambda1, order)
    lhs_y = _truncate(diag.q.compose([tx, ty]) + ty * diag.lambda2, order)
    u, v = PolyN.var(2, 0), PolyN.var(2, 1)
    rhs_u = _truncate(u * diag.lambda1 + normal_form.p_star, order)
    rhs_v = _truncate(v * diag.lambda2 + normal_form.q_star, order)
    jac_xu, jac_xv = tx.partial(0), tx.partial(1)
    jac_yu, jac_yv = ty.partial(0), ty.partial(1)
    rhs_x = _truncate(jac_xu * rhs_u + jac_xv * rhs_v, order)
    rhs_y = _truncate(jac_yu * rhs_u + jac_yv * rhs_v, order)

    rows: list[dict[str, object]] = []
    for component, (lhs, rhs) in enumerate(((lhs_x, rhs_x), (lhs_y, rhs_y))):
        monomials = sorted(set(lhs.terms) | set(rhs.terms))
        for mon in monomials:
            left = lhs.terms.get(mon, Fraction(0))
            right = rhs.terms.get(mon, Fraction(0))
            diff = left - right
            rows.append(
                {
                    "component": component,
                    "monomial": list(mon),
                    "lhs": [[left.numerator, left.denominator]],
                    "rhs": [[right.numerator, right.denominator]],
                    "terms": [[diff.numerator, diff.denominator]] if diff else [[0, 1]],
                    "rhs_eq": 0,
                }
            )
    return rows or [{"lhs": [[0, 1]], "rhs": [[0, 1]], "terms": [[0, 1]], "rhs_eq": 0}]


@dataclass(frozen=True)
class ResonantNormalFormCertificate:
    """A sealed truncated conjugacy identity for one resonant normal form."""

    field: PlanarPolynomialField
    saddle: HyperbolicSaddle
    normal_form: ResonantNormalForm
    source_digest: str
    formal_seal: Cert


@dataclass(frozen=True)
class ResonantNormalFormFormalVerification:
    result: LeanCheckResult
    theorem_prover_verified: bool


def certify_resonant_normal_form(
    field: PlanarPolynomialField,
    saddle: HyperbolicSaddle,
    *,
    order: int,
) -> ResonantNormalFormCertificate:
    """Diagonalize, normalize, and seal the truncated conjugacy identity."""
    diagonal = diagonalize_saddle(field, saddle)
    normal_form = resonant_normal_form(diagonal, order=order)
    rows = _conjugacy_rows(normal_form)
    seal = make_certificate(
        claim="Exact-Q truncated Poincare-Dulac conjugacy identity at a hyperbolic saddle.",
        payload={
            "type": "polynomial_identity_q",
            "source_digest": field.digest,
            "equations": rows,
            "positive": [],
            "truncation_order": order,
            "resonant_monomial_count": len(normal_form.resonant_monomials),
            "formal_scope": (
                "finite truncated formal conjugacy over Q; not a proof the singular "
                "Dulac map of the analytic field matches beyond this order, and not "
                "a cyclicity theorem"
            ),
        },
        honesty={
            "poincare_dulac_normal_form_exact_q": True,
            "resonant_monomials_derived_not_declared": True,
            "physical_return_membership_proved": False,
            "graphic_finite_cyclicity_proved": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return ResonantNormalFormCertificate(field, saddle, normal_form, field.digest, seal)


def verify_resonant_normal_form(certificate: ResonantNormalFormCertificate) -> bool:
    """Replay the diagonalization, normalization, and sealed conjugacy rows from source."""
    if (
        certificate.source_digest != certificate.field.digest
        or not verify_certificate_digest(certificate.formal_seal)
    ):
        return False
    try:
        expected = certify_resonant_normal_form(
            certificate.field,
            certificate.saddle,
            order=certificate.normal_form.order,
        )
    except (ArithmeticError, TypeError, ValueError):
        return False
    return expected == certificate


def verify_resonant_normal_form_formally(
    certificate: ResonantNormalFormCertificate,
) -> ResonantNormalFormFormalVerification:
    """Run the Mathlib-free kernel on the exact conjugacy identity."""
    result = check_certificate(certificate.formal_seal)
    return ResonantNormalFormFormalVerification(result, result.verified)


# --------------------------------------------------------------------------- #
# Exact rational power/log series: coefficient of ``x**exponent *
# log(x)**log_power``, keyed by ``(exponent, log_power)``. Used only to
# express the first-order corner-map correction; unrelated to (and
# independent of) the ``power_compensator`` interval machinery in
# ``omnibias.core.verified``.
# --------------------------------------------------------------------------- #
SeriesKey = tuple[Fraction, int]
Series = dict[SeriesKey, Fraction]


def _k_series(normal_form: ResonantNormalForm, r: Fraction, max_offset: Fraction) -> Series:
    """The exact linear log-correction generator ``K(x)`` (see
    :func:`dulac_corner_expansion`), keyed with ``log_power`` fixed at ``0``.

    Every resonant monomial of ``p_star`` (component ``0``, indices ``(i,
    j)`` with ``j = r*(i-1)`` by the resonance condition) contributes
    ``coefficient/lambda2 * x**(r*(i-1))`` to ``K``, and every resonant
    monomial of ``q_star`` (``j = r*i + 1``) contributes ``coefficient*r/
    lambda2 * x**(r*i)``. Both exponent formulas collapse to a value
    *independent of ``j``* -- a direct consequence of the resonance
    condition -- which is the algebraic fact that makes the first-order
    correction below exact (see :func:`dulac_corner_expansion`).
    """
    diag = normal_form.diagonal
    out: Series = {}
    for (i, _j), coefficient in normal_form.p_star.terms.items():
        exponent = r * (i - 1)
        if exponent <= max_offset:
            key = (exponent, 0)
            out[key] = out.get(key, Fraction(0)) + coefficient / diag.lambda2
    for (i, _j), coefficient in normal_form.q_star.terms.items():
        exponent = r * i
        if exponent <= max_offset:
            key = (exponent, 0)
            out[key] = out.get(key, Fraction(0)) + coefficient * r / diag.lambda2
    return {key: value for key, value in out.items() if value != 0}


@dataclass(frozen=True)
class DerivedDulacExpansion:
    """A Dulac expansion mechanically derived from a resonant normal form.

    Wraps a plain :class:`~omnibias.dynamics.dulac.DulacExpansion`
    (``expansion``) with honest provenance, kept off ``DulacExpansion``
    itself so every existing sealed digest stays byte-identical.
    ``residual_verified`` records that the closed-form derivation's
    resonance-condition collapse (every contributing monomial landing on a
    single ``x``-exponent, see :func:`_k_series`) was checked while building
    ``expansion`` -- it is always ``True`` for a value actually returned by
    :func:`dulac_corner_expansion` (that function raises instead of
    returning a value for which the collapse fails), kept as an explicit
    field rather than silently assumed.
    """

    expansion: DulacExpansion
    normal_form: ResonantNormalForm
    ratio: Fraction
    corner_order: int
    residual_verified: bool


def dulac_corner_expansion(
    normal_form: ResonantNormalForm,
    r: Fraction,
    *,
    order: int = 3,
) -> DerivedDulacExpansion:
    """Derive the corner map at a hyperbolic saddle from its resonant normal form.

    **Derivation.** Let ``H(xi,eta) = xi**lambda2 * eta**mu1`` (``mu1 =
    -lambda1``), the exact first integral of the *linear* part of the
    diagonal field. Tracking a single orbit from the incoming transversal
    ``(xi,eta)=(1,x)`` to the outgoing transversal ``(xi,eta)=(y,1)``, the
    linear part alone gives the classical leading order ``y = x**r`` (``r =
    mu1/lambda2``, matching :class:`~omnibias.dynamics.graphic.HyperbolicSaddle.ratio`
    exactly). The resonant nonlinear terms perturb ``log(H)`` by
    ``d(log H)/dt = lambda2*p_star(xi,eta)/xi + mu1*q_star(xi,eta)/eta``;
    changing the integration variable to ``eta`` and substituting the
    *leading-order* orbit relation ``xi = (x/eta)**r`` inside that already-
    small correction gives, after integrating from ``eta=x`` to ``eta=1``:

        ``log(y) = r*log(x) - K(x)*log(x)``,  ``K(x)`` from :func:`_k_series`,

    i.e. ``y = x**r - K(x)*log(x)*x**r + (uncomputed higher-order terms)``.
    A resonant monomial's contribution to ``K`` lands at a *single*
    ``x``-exponent (never spread over several), a direct algebraic
    consequence of the resonance condition itself (see :func:`_k_series`).

    **This module deliberately stops at first order and does not
    exponentiate ``exp(-K(x)*log(x))``.** A hand-solvable one-resonant-term
    check (``lambda1=-1``, ``lambda2=1``, ``p_star=c*xi**2*eta``, a Bernoulli
    equation with the closed form ``y = x/(1+c*x*log(x))``) shows the
    ``exp`` resummation reproduces the exact answer's *linear* term in
    ``c*x*log(x))`` correctly but gets the coefficient of the next term
    wrong (``1/2`` instead of the true ``1``) -- so exponentiating would
    silently claim second-order accuracy this derivation does not have.
    Truncating at plain first order, as implemented here, is exactly what it
    claims to be: the coefficient of the single leading ``x**r*log(x)``-type
    correction, for every resonant monomial through the declared ``order``,
    and no more. This is cross-checked against that exact solvable case (and
    against direct numerical integration of the source ODE) in the
    regression suite, not merely asserted.
    """
    if type(order) is not int or order < 1:
        raise ValueError("order must be a positive integer")
    if type(r) is not Fraction or r <= 0:
        raise ValueError("the hyperbolicity ratio r must be a positive Fraction")
    k_series = _k_series(normal_form, r, Fraction(order))
    eta: Series = {(r, 0): Fraction(1)}
    for (offset, _log_power), coefficient in k_series.items():
        key = (r + offset, 1)
        eta[key] = eta.get(key, Fraction(0)) - coefficient
    max_exponent = r + order
    terms = tuple(
        DulacTerm.create(exponent, log_power, coefficient)
        for (exponent, log_power), coefficient in sorted(eta.items())
        if exponent <= max_exponent and coefficient != 0
    )
    expansion = DulacExpansion(terms, order, Fraction(0))
    return DerivedDulacExpansion(expansion, normal_form, r, order, True)


def derive_return_map(
    corner: DerivedDulacExpansion,
    regular_arc: StoppedEventResult,
) -> DerivedDulacExpansion:
    """Compose a derived corner expansion with a certified regular-arc return.

    ``regular_arc`` must be a certified :func:`~omnibias.dynamics.return_maps.certify_stopped_event`
    result on a single scalar transversal (``return_jacobian`` a ``1x1``
    matrix); every corner coefficient is scaled by that Jacobian's interval
    enclosure, landing on :class:`~omnibias.dynamics.dulac.RationalInterval`
    coefficients since the arc Jacobian is a sound enclosure, not an exact
    rational.
    """
    if not regular_arc.certified or regular_arc.return_jacobian is None:
        raise ValueError("derive_return_map requires a certified regular arc with a return Jacobian")
    if len(regular_arc.return_jacobian) != 1 or len(regular_arc.return_jacobian[0]) != 1:
        raise ValueError("derive_return_map currently supports a single scalar transversal only")
    jac = regular_arc.return_jacobian[0][0]
    jac_lo = Fraction(jac.lo).limit_denominator(10**12)
    jac_hi = Fraction(jac.hi).limit_denominator(10**12)
    new_terms = []
    for term in corner.expansion.terms:
        candidates = (
            jac_lo * term.coefficient.lo,
            jac_lo * term.coefficient.hi,
            jac_hi * term.coefficient.lo,
            jac_hi * term.coefficient.hi,
        )
        new_terms.append(
            DulacTerm(term.exponent, term.log_power, RationalInterval(min(candidates), max(candidates)))
        )
    expansion = DulacExpansion(
        tuple(new_terms),
        corner.expansion.truncation_order,
        corner.expansion.remainder_bound,
    )
    return DerivedDulacExpansion(
        expansion,
        corner.normal_form,
        corner.ratio,
        corner.corner_order,
        corner.residual_verified,
    )
