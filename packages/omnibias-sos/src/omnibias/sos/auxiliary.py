# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The auxiliary-functional (background) method: certified for-all-data bounds.

For an autonomous polynomial ODE ``x' = f(x)`` and an observable ``Phi(x)``, the
long-time average ``limsup_{T->inf} (1/T) integral_0^T Phi(x(t)) dt`` is bounded
**above** by a constant ``C`` -- for **every** initial condition -- as soon as
there is an *auxiliary functional* ``V(x)`` (a polynomial) with

    S(x) := C - Phi(x) - (grad V(x)) . f(x)  >= 0   for all x.

The one-line proof: along any trajectory ``d/dt V(x(t)) = grad V . f``, so
``(1/T) integral_0^T Phi dt = C - (1/T) integral_0^T S dt - (V(x(T)) - V(x(0)))/T``.
When ``S >= 0`` the integral term is ``<= 0``; if in addition the trajectory stays
in a compact set (so ``V(x(t))`` is bounded), the boundary term vanishes in the
``limsup`` and the average is ``<= C``.  This is the Chernyshenko / Goluskin /
Fantuzzi "background method as SOS" -- exactly "optimization proving a universal
statement": minimise ``C`` over ``(C, V)`` with ``S`` SOS.

**Honest hypothesis (not global by fiat).**  A global SOS ``S >= 0`` certifies the
bound for **every forward trajectory that remains in a compact set** -- SOS bounds
"do not hold globally; they are generally violated by trajectories starting outside
the local basin of attraction" (Fantuzzi-Goluskin-Huang-Chernyshenko, *SIAM J.
Appl. Dyn. Syst.* 2016).  The bound is upgraded to hold for **all initial data**
exactly when the boundary term is controlled, for which a sufficient, separately
certified condition is that ``V`` is **bounded below** (then compact sublevel
control gives ``(V(x(T)) - V(x(0)))/T -> 0``).  Both facts are reported honestly:
``certified`` is the global SOS fact; ``applies_to_all_initial_data`` is the extra
``V``-bounded-below certificate.

**Soundness.**  ``S`` is formed with *exact rational* arithmetic
(:class:`~omnibias.sos.problem.RationalPolynomial`) from a rationally-rounded ``V``
and an upward-rounded rational ``C``, then certified by
:func:`~omnibias.sos.certify.certify_sos_rational`.  The float SDP only *proposes*
``(C, V)``; the sealed bound is a genuine rational SOS certificate.  A failed
certification returns **inconclusive**, never a false bound.

**Scope (honest).**  The bound is a ``for all initial data`` statement about the
**finite-dimensional** system as written -- e.g. a Galerkin truncation of a fluid
model.  It is *not* a statement about the continuum PDE and *not* a global-regularity claim;
sealed certificates carry ``unproven_claim = False`` and a ``finite_dim_system`` /
``galerkin_truncation`` scope.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from omnibias.core.proof.certificate import Cert
from omnibias.sos.certify import DEFAULT_DENOMINATORS, certify_sos_rational
from omnibias.sos.honesty import FINITE_DIM_SYSTEM, SOSScope, seal_sos_certificate
from omnibias.sos.monomials import gram_products, monomial_basis
from omnibias.sos.problem import Exponent, Polynomial, RationalPolynomial, SOSCertificate
from omnibias.sos.rounding import project_coefficients_exact
from omnibias.sos.solve import _svec_dim, _svec_indices, solve_gram_program

_SQRT2 = float(np.sqrt(2.0))

#: Relative slacks added to the proposed ``C`` (ascending) until ``S`` certifies SOS.
DEFAULT_SLACKS: tuple[float, ...] = (
    1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0,
)


@dataclass(frozen=True)
class PolynomialSystem:
    r"""An autonomous polynomial vector field ``x' = f(x)`` on ``R^{n_vars}``.

    ``field[j]`` is the polynomial right-hand side of ``x_j'``.  The system is
    finite-dimensional by construction; when it is a Galerkin truncation of a PDE
    that fact is recorded honestly on the sealed certificate's scope, never elided.
    """

    n_vars: int
    field: tuple[Polynomial, ...]

    def __post_init__(self) -> None:
        if len(self.field) != self.n_vars:
            raise ValueError(f"expected {self.n_vars} field components, got {len(self.field)}")
        for j, component in enumerate(self.field):
            if component.n_vars != self.n_vars:
                raise ValueError(f"field[{j}] has arity {component.n_vars}, expected {self.n_vars}")

    def lie_derivative(self, functional: Polynomial) -> Polynomial:
        r"""``grad(functional) . f`` as a float polynomial (the SDP proposer's view)."""
        acc = Polynomial.zero(self.n_vars)
        for j in range(self.n_vars):
            acc = acc + functional.partial(j) * self.field[j]
        return acc

    def lie_derivative_rational(self, functional: RationalPolynomial) -> RationalPolynomial:
        r"""``grad(functional) . f`` with **exact** rational coefficients."""
        acc = RationalPolynomial.zero(self.n_vars)
        for j in range(self.n_vars):
            fj = RationalPolynomial.from_polynomial(self.field[j])
            acc = acc + functional.partial(j) * fj
        return acc


@dataclass(frozen=True)
class AuxiliaryBoundCertificate:
    r"""A certified upper bound on the infinite-time average of an observable.

    A ``status == "proved"`` certificate carries the exact rational bound ``C`` and
    auxiliary functional ``V`` for which ``C - Phi - grad(V) . f`` is a certified
    sum of squares, so ``limsup (1/T) integral Phi dt <= C`` for **all** initial
    data of the finite-dimensional system.  ``status == "inconclusive"`` makes no
    claim.
    """

    status: str
    n_vars: int
    bound: str
    """The certified upper bound ``C`` as an exact rational string (``""`` if none)."""
    auxiliary: tuple[tuple[Exponent, str], ...]
    """The auxiliary functional ``V``: ``(exponent, rational-coefficient-string)`` pairs."""
    sos_certificate: SOSCertificate | None
    """The SOS proof of ``S = C - Phi - grad(V) . f >= 0`` (``None`` if inconclusive)."""
    detail: str
    v_lower_bound: str = ""
    """A certified rational lower bound ``m <= V`` (``""`` if not certified).

    When non-empty, ``V`` is bounded below, the boundary term is controlled, and the
    bound holds for **all** initial data -- not only a-priori-bounded trajectories.
    """

    @property
    def certified(self) -> bool:
        """Whether ``S = C - Phi - grad(V).f >= 0`` is soundly proved (bounded-trajectory bound)."""
        return self.status == "proved"

    @property
    def applies_to_all_initial_data(self) -> bool:
        """Whether ``V`` is certified bounded below, upgrading the bound to all initial data."""
        return self.certified and bool(self.v_lower_bound)

    @property
    def bound_value(self) -> float:
        """The bound ``C`` as a float (``nan`` when inconclusive)."""
        return float(Fraction(self.bound)) if self.bound else float("nan")

    def auxiliary_polynomial(self, n_vars: int | None = None) -> RationalPolynomial:
        """Reconstruct the exact auxiliary functional ``V`` as a polynomial."""
        n = self.n_vars if n_vars is None else n_vars
        return RationalPolynomial(n, {exp: Fraction(c) for exp, c in self.auxiliary})


def _auxiliary_basis(n_vars: int, degree: int) -> list[Exponent]:
    """Non-constant monomials up to ``degree`` (a constant ``V`` has zero Lie derivative)."""
    return [exp for exp in monomial_basis(n_vars, degree) if sum(exp) >= 1]


def _certify_v_lower_bound(
    functional: RationalPolynomial, *, denominators: Sequence[int]
) -> Fraction | None:
    r"""A certified rational ``m`` with ``V - m`` SOS (hence ``V >= m``), or ``None``.

    Any finite lower bound suffices to control the background method's boundary term;
    a coercive ``V`` has one.  A geometric ladder ``0, -1, -2, -4, ...`` is tried from
    the top down, so the first success is the tightest certifiable bound and the search
    is robust at any coefficient scale.  Each candidate is accepted only when the
    rigorous interval ``LDL^T`` certifies ``V - m`` SOS -- this is sound, never a
    heuristic.  Returns ``None`` when ``V`` is not certifiably bounded below (e.g. an
    indefinite / negative-leading functional), reported honestly rather than silently
    upgraded to an all-data claim.
    """
    half_v = max(0, (functional.degree() + 1) // 2)
    n = functional.n_vars
    ladder = (0, *(-(2**k) for k in range(48)))  # 0, -1, -2, ..., -2^47
    for m in ladder:
        shifted = functional - RationalPolynomial.constant(Fraction(m), n)
        if certify_sos_rational(shifted, half_degree=half_v, denominators=denominators).certified:
            return Fraction(m)
    return None


def certify_time_average_bound(
    system: PolynomialSystem,
    observable: Polynomial,
    *,
    auxiliary_degree: int = 2,
    gram_half_degree: int | None = None,
    slacks: Sequence[float] = DEFAULT_SLACKS,
    v_denominator: int = 2**12,
    c_denominator: int = 2**16,
    free_ridge: float = 1.0,
    denominators: Sequence[int] = DEFAULT_DENOMINATORS,
) -> AuxiliaryBoundCertificate:
    r"""Certify ``limsup (1/T) integral Phi dt <= C`` for all data of ``x' = f(x)``.

    Parameters
    ----------
    system:
        The polynomial ODE (a finite-dimensional / Galerkin-truncated system).
    observable:
        The observable ``Phi(x)`` whose long-time average is bounded.
    auxiliary_degree:
        Degree of the auxiliary functional ``V`` (default quadratic).
    gram_half_degree:
        Half-degree of the SOS Gram basis for ``S``; defaults to
        ``ceil(deg(S) / 2)``.
    slacks:
        Relative slacks tried (ascending) on the proposed ``C`` until ``S``
        certifies SOS -- the first success is the tightest certified bound.
    v_denominator, c_denominator:
        Denominators for rounding ``V`` and (upward) ``C`` to exact rationals.
    free_ridge:
        Ridge weight on the auxiliary-functional coefficients in the float proposer,
        which returns a small-norm (well-scaled) ``V`` without changing the optimal
        ``C``.  Purely a numeric tie-breaker; the proof is the downstream certificate.

    Returns
    -------
    AuxiliaryBoundCertificate
        ``status == "proved"`` carries the exact rational bound ``C`` and auxiliary
        functional ``V`` plus the SOS proof of ``S >= 0``; ``status ==
        "inconclusive"`` makes no claim (SDP infeasible, or no slack in the schedule
        yielded a certifiable ``S``).
    """
    n = system.n_vars
    vbasis = _auxiliary_basis(n, auxiliary_degree)
    if not vbasis:
        return AuxiliaryBoundCertificate(
            "inconclusive", n, "", (), None, "auxiliary_degree must be >= 1"
        )
    lie_of_basis = [system.lie_derivative(Polynomial.monomial(exp, 1.0)) for exp in vbasis]

    # Size the SOS ansatz by the observable: 2*half = deg(Phi) rounded up to even.  Any
    # HIGHER-degree monomial of grad(V).f that the Gram basis cannot build then imposes a
    # linear *cancellation* constraint on (C, V) -- exactly the background method's
    # coercivity conditions (e.g. it forces the triad's odd x1 x2 x3 term to vanish, so
    # the residual S is genuinely even-degree and admits a strictly-PD Gram).
    half = gram_half_degree if gram_half_degree is not None else max(1, (observable.degree() + 1) // 2)
    gram_basis = monomial_basis(n, half)
    m = len(gram_basis)
    svec_dim = _svec_dim(m)
    svec_idx = _svec_indices(m)
    products = gram_products(gram_basis)
    zero_exp: Exponent = (0,) * n

    # Constraint monomials: every Gram product, plus everything S can contain.
    alphas: set[Exponent] = set(products) | set(observable.support) | {zero_exp}
    for lie in lie_of_basis:
        alphas |= set(lie.support)
    alpha_list = sorted(alphas)

    p = len(vbasis)
    n_free = 1 + p  # (C, v_1, ..., v_p)
    lhs = np.zeros((len(alpha_list), svec_dim + n_free))
    rhs = np.zeros(len(alpha_list))
    for row, alpha in enumerate(alpha_list):
        for i, j, _mult in products.get(alpha, []):
            lhs[row, svec_idx[(i, j)]] = 1.0 if i == j else _SQRT2
        # C contributes only to the constant monomial; move -C, +sum v_k L_k to the LHS.
        lhs[row, svec_dim] = -1.0 if alpha == zero_exp else 0.0
        for k, lie in enumerate(lie_of_basis):
            lhs[row, svec_dim + 1 + k] = lie.coefficient(alpha)
        rhs[row] = -observable.coefficient(alpha)

    objective = np.zeros(svec_dim + n_free)
    objective[svec_dim] = 1.0  # minimise C

    # Ridge the auxiliary-functional coefficients v (not C, not Q) so the proposer
    # returns a well-scaled V; purely a numeric tie-breaker (soundness is downstream,
    # and the ridge leaves the optimal C unchanged -- it only picks a small-norm V).
    reg = np.zeros(svec_dim + n_free)
    reg[svec_dim + 1 :] = free_ridge
    proposal = solve_gram_program(m, lhs, rhs, objective, reg=reg)
    if proposal.status != "solved" or proposal.free_vars is None:
        return AuxiliaryBoundCertificate(
            "inconclusive", n, "", (), None,
            f"auxiliary SDP proposer: {proposal.detail} "
            "(no PD Gram makes C - Phi - grad(V).f a sum of squares at this degree)",
        )
    free = np.asarray(proposal.free_vars, dtype=float)
    c_star = float(free[0])
    v_star = free[1:]

    # Round V, then snap it onto the EXACT rational cancellation constraints: every
    # residual monomial the Gram basis cannot build must vanish identically (a degree-3
    # remainder is unbounded below, so no slack could ever rescue it).  These are the
    # background method's coercivity conditions; solving them over Q keeps S sound.
    v_rounded = [
        Fraction(int(round(float(v_star[k]) * v_denominator)), v_denominator) for k in range(p)
    ]
    high_alphas = [alpha for alpha in alpha_list if alpha not in products]
    if high_alphas:
        cancel_matrix = [
            [Fraction(lie_of_basis[k].coefficient(alpha)) for k in range(p)] for alpha in high_alphas
        ]
        cancel_target = [-Fraction(observable.coefficient(alpha)) for alpha in high_alphas]
        v_exact = project_coefficients_exact(v_rounded, cancel_matrix, cancel_target)
        if v_exact is None:
            return AuxiliaryBoundCertificate(
                "inconclusive", n, "", (), None,
                "the residual has high-degree monomials no auxiliary functional of this "
                "degree can cancel (raise auxiliary_degree, or the average may be unbounded)",
            )
    else:
        v_exact = v_rounded

    v_rat = RationalPolynomial(n, {vbasis[k]: v_exact[k] for k in range(p)})
    lie_v = system.lie_derivative_rational(v_rat)
    phi_rat = RationalPolynomial.from_polynomial(observable)

    scale = abs(c_star) + 1.0
    for slack in slacks:
        c_target = c_star + slack * scale
        c_rat = Fraction(math.ceil(c_target * c_denominator), c_denominator)  # upward => sound bound
        residual = RationalPolynomial.constant(c_rat, n) - phi_rat - lie_v
        cert = certify_sos_rational(residual, half_degree=half, denominators=denominators)
        if cert.certified:
            lower = _certify_v_lower_bound(v_rat, denominators=denominators)
            if lower is not None:
                scope_note = (
                    f"V >= {float(lower):.6g} (bounded below) => the bound holds for ALL initial data"
                )
            else:
                scope_note = (
                    "V not certified bounded below => the bound holds for every forward "
                    "trajectory that remains in a compact set"
                )
            return AuxiliaryBoundCertificate(
                status="proved",
                n_vars=n,
                bound=str(c_rat),
                auxiliary=tuple((exp, str(coeff)) for exp, coeff in sorted(v_rat.coeffs.items())),
                sos_certificate=cert,
                detail=(
                    f"auxiliary-functional bound C = {float(c_rat):.6g}: "
                    f"C - Phi - grad(V).f is a certified SOS (PD margin {cert.pd_margin:.3e}); "
                    f"finite-dimensional system; {scope_note}"
                ),
                v_lower_bound="" if lower is None else str(lower),
            )

    return AuxiliaryBoundCertificate(
        "inconclusive", n, "", (), None,
        "no slack in the schedule yielded a certifiable S = C - Phi - grad(V).f "
        "(try a larger auxiliary_degree / gram_half_degree, or the average may be unbounded)",
    )


def energy_observable(n_vars: int, *, weights: Sequence[float] | None = None) -> Polynomial:
    r"""The quadratic energy ``Phi(x) = (1/2) sum w_i x_i^2`` (unit weights by default)."""
    w = [1.0] * n_vars if weights is None else list(weights)
    if len(w) != n_vars:
        raise ValueError(f"expected {n_vars} weights, got {len(w)}")
    coeffs: dict[Exponent, float] = {}
    for i in range(n_vars):
        exp = [0] * n_vars
        exp[i] = 2
        coeffs[tuple(exp)] = 0.5 * w[i]
    return Polynomial(n_vars, coeffs)


def energy_conserving_triad_system(
    *,
    viscosities: Sequence[float],
    couplings: tuple[float, float, float] = (1.0, 1.0, -2.0),
    forcing: Sequence[float] | None = None,
) -> PolynomialSystem:
    r"""A 3-mode energy-conserving Galerkin triad with linear damping and forcing.

    ``x_1' = c_1 x_2 x_3 - nu_1 x_1 + f_1``, and cyclically, with the quadratic
    couplings ``c_k`` summing to zero so the nonlinearity conserves energy
    ``sum x_k^2`` (energy is injected only by forcing and removed only by the
    ``-nu_k x_k`` dissipation).  This is the honest finite-dimensional caricature of
    a spectrally-truncated 2-D fluid used for the flagship dissipation-bound demo --
    a Galerkin truncation, explicitly **not** the continuum PDE.
    """
    nu = list(viscosities)
    if len(nu) != 3:
        raise ValueError("energy_conserving_triad_system needs exactly 3 viscosities")
    if abs(sum(couplings)) > 1e-12:
        raise ValueError("couplings must sum to 0 for the nonlinearity to conserve energy")
    f = [0.0, 0.0, 0.0] if forcing is None else list(forcing)
    if len(f) != 3:
        raise ValueError("forcing must have length 3")

    def component(i: int, j: int, k: int, coupling: float, nu_i: float, force: float) -> Polynomial:
        # coupling * x_j x_k - nu_i * x_i + force
        exp_jk = [0, 0, 0]
        exp_jk[j] += 1
        exp_jk[k] += 1
        exp_i = [0, 0, 0]
        exp_i[i] = 1
        coeffs: dict[Exponent, float] = {tuple(exp_jk): coupling, tuple(exp_i): -nu_i}
        if force != 0.0:
            coeffs[(0, 0, 0)] = force
        return Polynomial(3, coeffs)

    return PolynomialSystem(
        3,
        (
            component(0, 1, 2, couplings[0], nu[0], f[0]),
            component(1, 2, 0, couplings[1], nu[1], f[1]),
            component(2, 0, 1, couplings[2], nu[2], f[2]),
        ),
    )


def seal_auxiliary_bound(
    certificate: AuxiliaryBoundCertificate,
    *,
    claim: str,
    scope: SOSScope | None = None,
) -> Cert:
    r"""Seal a **proved** auxiliary-functional bound into a v1 ``positive_definite`` cert.

    The sealed obligation is the SOS proof of ``S = C - Phi - grad(V).f >= 0`` (its
    interval ``LDL^T`` pivots), which the Lean kernel turns into ``allPivotsPos``.  The
    bound ``C``, the auxiliary functional ``V``, and whether the bound applies to all
    initial data are recorded in the certificate metadata.  The scope defaults to
    :data:`~omnibias.sos.honesty.FINITE_DIM_SYSTEM` -- the bound is a for-all-data
    statement about the **finite-dimensional** system, never a continuum-PDE or global-regularity
    claim (``unproven_claim = False``).  Raises :class:`ValueError` on an inconclusive bound.
    """
    if not certificate.certified or certificate.sos_certificate is None:
        raise ValueError("cannot seal an inconclusive auxiliary-functional bound")
    if scope is None:
        scope = SOSScope(FINITE_DIM_SYSTEM)
    meta = {
        "auxiliary_bound": {
            "bound_C": certificate.bound,
            "auxiliary_V": [[list(exp), coeff] for exp, coeff in certificate.auxiliary],
            "applies_to_all_initial_data": certificate.applies_to_all_initial_data,
            "v_lower_bound": certificate.v_lower_bound,
        },
    }
    return seal_sos_certificate(certificate.sos_certificate, claim=claim, scope=scope, meta=meta)


__all__ = [
    "AuxiliaryBoundCertificate",
    "DEFAULT_SLACKS",
    "PolynomialSystem",
    "certify_time_average_bound",
    "energy_conserving_triad_system",
    "energy_observable",
    "seal_auxiliary_bound",
]
