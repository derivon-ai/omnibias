# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""End-to-end wiring: trained network / spectral ansatz -> certified PDE residual.

This module is the *adaptor* that turns the verified primitives into a sealed
:class:`~omnibias.core.proof.certificate.Cert`.  It connects three pieces that
already exist in :mod:`omnibias.core.verified`:

#. the **certified jet** (:func:`~omnibias.core.verified.jet_mv.certified_partials`)
   -- rigorous enclosures of every mixed partial of a deep MLP over an input box;
#. the **radii-polynomial / Newton-Kantorovich** closure
   (:func:`~omnibias.core.verified.kantorovich.radii_polynomial_certificate`);
#. the validated **Fourier algebra**
   (:class:`~omnibias.core.verified.fourier.ValidatedFourierSeries`) for the
   spectral / nonlocal route.

Two complementary, fully-rigorous certificates are produced.

A-posteriori error bound (physical-space / trained MLP)
------------------------------------------------------
For a *linear, well-posed* boundary-value problem ``L u = f`` in ``Omega`` with
``u = g`` on ``dOmega`` and a rigorous **stability constant** pair
``(C_Omega, C_dOmega)`` such that

.. math::

    \|v\|_\infty \le C_\Omega\,\|L v\|_{\infty,\Omega}
                   + C_{\partial\Omega}\,\|v\|_{\infty,\partial\Omega}
    \qquad \forall v,

applying the estimate to ``v = u_NN - u_true`` gives the certified bound

.. math::

    \|u_{NN} - u_{\text{true}}\|_\infty
      \le C_\Omega\,R_{\text{int}} + C_{\partial\Omega}\,R_{\text{bnd}},

where ``R_int = sup_Omega |L u_NN - f|`` and ``R_bnd = sup_{dOmega} |u_NN - g|``
are computed *rigorously* from the certified jet (box subdivision tightens both).
The stability constants are the caller's proof obligation -- they are recorded in
the certificate, not invented here.

Radii-polynomial existence (spectral / finite ansatz)
-----------------------------------------------------
:func:`radii_polynomial_residual_certificate` feeds a certified residual as the
Newton-Kantorovich defect ``Y0`` into the radii polynomial, proving a *true*
solution exists in an explicit ball around the approximation when the contraction
data ``(Z0, Z1, Z2)`` is supplied.  :func:`spectral_residual_norm` assembles the
residual ``F(a) = L a + Q(a) - rhs`` of a band-limited Fourier ansatz with the
validated algebra and returns its rigorous ``l1_nu`` norm -- the natural ``Y0``
for a steady / self-similar spectral profile.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omnibias.core.multi_index import MultiIndex
from omnibias.core.verified.fourier import Symbol, ValidatedFourierSeries
from omnibias.core.verified.interval import Interval, IntervalLike, hull
from omnibias.core.verified.jet_mv import BoxLike, Layer, certified_partials
from omnibias.core.verified.kantorovich import (
    RadiiCertificate,
    radii_polynomial_certificate,
)

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

#: A PDE coefficient / source: a constant or a function of the (sub-)box.
Coeff = IntervalLike | Callable[[Sequence[Interval]], IntervalLike]
#: Experimental nonlinear / custom residual callback.  It receives the current
#: sub-box and the certified partial dictionary produced for that box.
CertifiedResidualCallback = Callable[
    [Sequence[Interval], Mapping[MultiIndex, Sequence[Interval]]], IntervalLike
]


def _eval_coeff(c: Coeff, box: Sequence[Interval]) -> Interval:
    if callable(c):
        return Interval.from_value(c(box))
    return Interval.from_value(c)


def _promote(box: BoxLike) -> list[Interval]:
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


def _normalize_splits(splits: int | Sequence[int], dim: int) -> list[int]:
    if isinstance(splits, int):
        return [splits] * dim
    out = list(splits)
    if len(out) != dim:
        raise ValueError(f"splits must have one entry per axis ({dim}), got {len(out)}")
    return out


def _grid(box: Sequence[Interval], splits: Sequence[int]) -> list[list[Interval]]:
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


def _box_widths(box: Sequence[Interval]) -> list[float]:
    return [max(0.0, iv.hi - iv.lo) for iv in box]


def _as_payload_value(value: object) -> object:
    if isinstance(value, Interval):
        return [value.lo, value.hi]
    if isinstance(value, Mapping):
        return {str(k): _as_payload_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_as_payload_value(v) for v in value]
    if isinstance(value, str | bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return float(value)
    return str(value)


@dataclass(frozen=True)
class StabilityEstimate:
    r"""Provenance for the constants in the a-posteriori PDE estimate.

    The constants remain mathematical proof obligations.  This record prevents a
    certificate from carrying anonymous numbers by recording their source,
    assumptions, and whether they were supplied by the caller or a library helper.
    """

    interior: float
    boundary: float = 1.0
    source: str = "user_supplied"
    pde_family: str = "unspecified"
    domain: str = "unspecified"
    assumptions: tuple[str, ...] = ()
    library_provided: bool = False

    def __post_init__(self) -> None:
        if self.interior < 0.0 or self.boundary < 0.0:
            raise ValueError("stability constants must be non-negative")

    def to_payload(self) -> dict[str, object]:
        return {
            "interior": float(self.interior),
            "boundary": float(self.boundary),
            "source": self.source,
            "pde_family": self.pde_family,
            "domain": self.domain,
            "assumptions": list(self.assumptions),
            "library_provided": bool(self.library_provided),
        }


def user_stability_estimate(
    interior: float,
    boundary: float = 1.0,
    *,
    source: str = "user_supplied",
    pde_family: str = "unspecified",
    domain: str = "unspecified",
    assumptions: Sequence[str] = (),
) -> StabilityEstimate:
    """Build an explicit user-supplied stability estimate record."""
    return StabilityEstimate(
        float(interior),
        float(boundary),
        source=source,
        pde_family=pde_family,
        domain=domain,
        assumptions=tuple(str(a) for a in assumptions),
        library_provided=False,
    )


@dataclass(frozen=True)
class StructuralInvariant:
    """A structural identity recorded in a certificate payload."""

    kind: str
    expression: str
    certified: bool = True
    assumptions: tuple[str, ...] = ()
    method: str = "structural_identity"

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "expression": self.expression,
            "certified": bool(self.certified),
            "assumptions": list(self.assumptions),
            "method": self.method,
        }


def structural_invariant(
    kind: str,
    expression: str,
    *,
    assumptions: Sequence[str] = (),
    certified: bool = True,
    method: str = "structural_identity",
) -> StructuralInvariant:
    """Create a serialisable record for an invariant enforced by construction."""
    if not kind:
        raise ValueError("invariant kind must be non-empty")
    if certified and not expression:
        raise ValueError("certified structural invariants need an expression")
    return StructuralInvariant(
        kind=str(kind),
        expression=str(expression),
        certified=bool(certified),
        assumptions=tuple(str(a) for a in assumptions),
        method=str(method),
    )


@dataclass(frozen=True)
class AdaptiveResidualDiagnostics:
    """Diagnostics from adaptive residual subdivision."""

    residual: Interval
    residual_sup: float
    splits: tuple[int, ...]
    boxes: int
    iterations: int
    target: float | None
    reached_target: bool
    max_box_width: float

    def to_payload(self) -> dict[str, object]:
        return {
            "residual_enclosure": [self.residual.lo, self.residual.hi],
            "residual_sup": float(self.residual_sup),
            "splits": list(self.splits),
            "boxes": int(self.boxes),
            "iterations": int(self.iterations),
            "target": None if self.target is None else float(self.target),
            "reached_target": bool(self.reached_target),
            "max_box_width": float(self.max_box_width),
        }


# --------------------------------------------------------------------------- #
# Linear PDE operator DSL.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LinearPDE:
    r"""A linear differential operator ``L u - f`` as a residual functional.

    ``terms`` maps a multi-index ``alpha`` to its coefficient ``c_alpha`` (a
    constant or a ``box -> value`` callable for variable coefficients), encoding
    ``L u = sum_alpha c_alpha D^alpha u``.  ``source`` is ``f`` (constant or
    ``box -> value``).  :meth:`residual` evaluates a rigorous enclosure of
    ``L u(x) - f(x)`` over a box from a certified-partials dictionary.
    """

    terms: Mapping[MultiIndex, Coeff]
    source: Coeff = 0.0
    component: int = 0

    def required_order(self) -> int:
        """Lowest jet order that contains every derivative the operator needs."""
        if not self.terms:
            return 0
        return max(sum(alpha) for alpha in self.terms)

    def residual(
        self, box: Sequence[Interval], partials: Mapping[MultiIndex, Sequence[Interval]]
    ) -> Interval:
        """Enclosure of ``L u - f`` over ``box`` from the certified partials."""
        acc = Interval.point(0.0)
        for alpha, coeff in self.terms.items():
            row = partials[alpha]
            acc = acc + _eval_coeff(coeff, box) * row[self.component]
        return acc - _eval_coeff(self.source, box)


def _unit(i: int, dim: int) -> MultiIndex:
    return tuple(1 if j == i else 0 for j in range(dim))


def _double(i: int, dim: int) -> MultiIndex:
    return tuple(2 if j == i else 0 for j in range(dim))


def laplace(dim: int, *, component: int = 0) -> LinearPDE:
    r"""The Laplace operator residual ``Delta u`` (harmonic when zero)."""
    return LinearPDE({_double(i, dim): 1.0 for i in range(dim)}, 0.0, component)


def poisson(dim: int, source: Coeff, *, component: int = 0) -> LinearPDE:
    r"""The Poisson residual ``Delta u - f``."""
    return LinearPDE({_double(i, dim): 1.0 for i in range(dim)}, source, component)


def helmholtz(
    dim: int, wavenumber: float, source: Coeff = 0.0, *, component: int = 0
) -> LinearPDE:
    r"""The Helmholtz residual ``Delta u + k^2 u - f``."""
    terms: dict[MultiIndex, Coeff] = {_double(i, dim): 1.0 for i in range(dim)}
    terms[(0,) * dim] = float(wavenumber) ** 2
    return LinearPDE(terms, source, component)


def screened_poisson(
    dim: int, kappa: float, source: Coeff = 0.0, *, component: int = 0
) -> LinearPDE:
    r"""The screened-Poisson / modified-Helmholtz residual ``Delta u - kappa^2 u - f``."""
    terms: dict[MultiIndex, Coeff] = {_double(i, dim): 1.0 for i in range(dim)}
    terms[(0,) * dim] = -(float(kappa) ** 2)
    return LinearPDE(terms, source, component)


def advection_diffusion(
    dim: int,
    velocity: Sequence[float],
    diffusivity: float,
    source: Coeff = 0.0,
    *,
    component: int = 0,
) -> LinearPDE:
    r"""The steady advection-diffusion residual ``b . grad u - nu Delta u - f``."""
    if len(velocity) != dim:
        raise ValueError(f"velocity must have {dim} components, got {len(velocity)}")
    terms: dict[MultiIndex, Coeff] = {}
    for i in range(dim):
        terms[_unit(i, dim)] = float(velocity[i])
        terms[_double(i, dim)] = -float(diffusivity)
    return LinearPDE(terms, source, component)


# --------------------------------------------------------------------------- #
# Certified residuals from a trained network.
# --------------------------------------------------------------------------- #
def certified_interior_residual(
    layers: Sequence[Layer],
    domain: BoxLike,
    pde: LinearPDE,
    *,
    splits: int | Sequence[int] = 1,
) -> Interval:
    r"""Rigorous enclosure of ``{ L u_NN(x) - f(x) : x in domain }``.

    ``u_NN`` is the MLP described by ``layers`` (the
    :data:`~omnibias.core.verified.jet_mv.Layer` format).  The enclosure is the
    hull over a ``splits`` grid of sub-boxes; ``certified_interior_residual(...)
    .mag`` is a certified sup-norm bound on the interior PDE residual.
    """
    box = _promote(domain)
    dim = len(box)
    order = pde.required_order()
    per = _normalize_splits(splits, dim)
    pieces = [
        pde.residual(sub, certified_partials(sub, layers, order))
        for sub in _grid(box, per)
    ]
    return hull(list(pieces))


def certified_custom_residual(
    layers: Sequence[Layer],
    domain: BoxLike,
    residual: CertifiedResidualCallback,
    *,
    order: int,
    splits: int | Sequence[int] = 1,
) -> Interval:
    r"""Rigorous enclosure of a user-supplied certified residual callback.

    This is the v1 extension point for semilinear / nonlinear PDE terms.  The
    callback must be inclusion-isotonic in the interval inputs it receives; the
    helper supplies certified partials through ``order`` on every sub-box and
    returns the hull of the callback values.  General nonlinear PDE soundness is
    therefore the callback author's obligation.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    box = _promote(domain)
    dim = len(box)
    per = _normalize_splits(splits, dim)
    pieces = [
        Interval.from_value(residual(sub, certified_partials(sub, layers, order)))
        for sub in _grid(box, per)
    ]
    return hull(list(pieces))


def certified_quadratic_reaction_residual(
    layers: Sequence[Layer],
    domain: BoxLike,
    pde: LinearPDE,
    *,
    coefficient: Coeff = 1.0,
    splits: int | Sequence[int] = 1,
) -> Interval:
    r"""Experimental enclosure of ``L u - f + c u^2`` for scalar semilinear PDEs."""

    def residual(
        box: Sequence[Interval], partials: Mapping[MultiIndex, Sequence[Interval]]
    ) -> Interval:
        value = partials[(0,) * len(box)][pde.component]
        return pde.residual(box, partials) + _eval_coeff(coefficient, box) * value * value

    return certified_custom_residual(
        layers, domain, residual, order=pde.required_order(), splits=splits
    )


# --------------------------------------------------------------------------- #
# Incompressible streamfunction cage: rigorous nonlinear vorticity residual.
# --------------------------------------------------------------------------- #
r"""A 2-D incompressible velocity ``u = \nabla^\perp\psi = (\psi_y, -\psi_x)`` built
from a scalar streamfunction ``\psi`` is divergence free **by construction** --
``\nabla\cdot u = \psi_{xy} - \psi_{yx} \equiv 0`` (equality of mixed partials).
With ``\psi`` an MLP in the :data:`Layer` format the certified multivariate jet
encloses every partial of ``\psi`` over a box, so the steady vorticity-transport
residual of the velocity it induces is a rigorous interval -- a *whole-domain*
(between-grid-node) enclosure, the nonlinear counterpart of
:func:`certified_interior_residual`.
"""


def _neg(x: Interval) -> Interval:
    return Interval.point(0.0) - x


def certified_streamfunction_divergence(
    layers: Sequence[Layer],
    domain: BoxLike,
    *,
    splits: int | Sequence[int] = 1,
) -> Interval:
    r"""Rigorous enclosure of ``\nabla\cdot(\nabla^\perp\psi)`` over the box.

    For a scalar streamfunction ``\psi`` the induced velocity ``u = (\psi_y,
    -\psi_x)`` is divergence free identically; the divergence
    ``\psi_{xy} - \psi_{yx}`` is recomputed naively in interval arithmetic, so the
    returned enclosure always contains ``0`` and ``.mag`` is a certified bound on
    the (structurally zero) incompressibility defect that tightens to ``0`` as
    ``splits`` grows.  This is the numerical cross-check of the structural cage,
    not its proof (the proof is equality of mixed partials).
    """

    def residual(
        box: Sequence[Interval], partials: Mapping[MultiIndex, Sequence[Interval]]
    ) -> Interval:
        cross = partials[(1, 1)][0]
        return cross - cross

    return certified_custom_residual(layers, domain, residual, order=2, splits=splits)


def certified_vorticity_transport_residual(
    layers: Sequence[Layer],
    domain: BoxLike,
    *,
    viscosity: float = 0.0,
    forcing: Coeff = 0.0,
    splits: int | Sequence[int] = 1,
) -> Interval:
    r"""Rigorous enclosure of the steady 2-D vorticity-transport residual.

    The streamfunction ``\psi`` (an MLP in :data:`Layer` format, scalar output)
    induces ``u = (\psi_y, -\psi_x)`` and vorticity ``\omega = -\Delta\psi``.  The
    residual of the steady (Navier--)Stokes vorticity equation

    .. math::

        R = (u\cdot\nabla)\omega - \nu\,\Delta\omega - f_\omega

    is assembled from the certified partials of ``\psi`` (up to order 3 for the
    Euler advection ``\nu=0``, order 4 when the viscous term is included):

    .. math::

        u\cdot\nabla\omega
          = \psi_y\,\bigl[-(\psi_{xxx}+\psi_{xyy})\bigr]
          - \psi_x\,\bigl[-(\psi_{xxy}+\psi_{yyy})\bigr],\qquad
        \Delta\omega = -(\psi_{xxxx}+2\psi_{xxyy}+\psi_{yyyy}).

    The result encloses ``{ R(x) : x in domain }`` for *every* point of the box,
    so ``.mag`` is a certified sup-norm bound on the residual.  ``forcing`` is the
    target vorticity forcing ``f_\omega`` (a constant or a rigorous
    ``box -> enclosure`` callback -- its soundness is the caller's obligation, like
    a source term).  ``viscosity`` is ``\nu``; with ``\nu = 0`` and ``forcing = 0``
    this is the exact steady-Euler residual.
    """
    if viscosity < 0.0:
        raise ValueError("viscosity must be non-negative")
    order = 4 if viscosity != 0.0 else 3
    nu = Interval.point(float(viscosity))
    two = Interval.point(2.0)

    def residual(
        box: Sequence[Interval], partials: Mapping[MultiIndex, Sequence[Interval]]
    ) -> Interval:
        def d(alpha: MultiIndex) -> Interval:
            return partials[alpha][0]

        u1 = d((0, 1))  # psi_y  = u_x
        u2 = _neg(d((1, 0)))  # -psi_x = u_y
        dx_omega = _neg(d((3, 0)) + d((1, 2)))  # d_x omega
        dy_omega = _neg(d((2, 1)) + d((0, 3)))  # d_y omega
        res = u1 * dx_omega + u2 * dy_omega
        if viscosity != 0.0:
            lap_omega = _neg(d((4, 0)) + two * d((2, 2)) + d((0, 4)))
            res = res - nu * lap_omega
        return res - _eval_coeff(forcing, box)

    return certified_custom_residual(layers, domain, residual, order=order, splits=splits)


def adaptive_certified_interior_residual(
    layers: Sequence[Layer],
    domain: BoxLike,
    pde: LinearPDE,
    *,
    target: float | None = None,
    initial_splits: int | Sequence[int] = 1,
    max_splits: int = 16,
) -> AdaptiveResidualDiagnostics:
    """Refine uniform subdivisions until a residual target or split budget is hit."""
    box = _promote(domain)
    dim = len(box)
    splits = _normalize_splits(initial_splits, dim)
    if max_splits < 1:
        raise ValueError("max_splits must be >= 1")
    if any(s > max_splits for s in splits):
        raise ValueError("initial_splits cannot exceed max_splits")
    tgt = None if target is None else float(target)
    if tgt is not None and tgt < 0.0:
        raise ValueError("target must be non-negative")

    iterations = 0
    while True:
        iterations += 1
        residual = certified_interior_residual(layers, box, pde, splits=splits)
        reached = tgt is not None and residual.mag <= tgt
        saturated = all(s >= max_splits for s in splits)
        if reached or saturated or tgt is None:
            boxes = 1
            for s in splits:
                boxes *= s
            widths = _box_widths(box)
            max_width = max((w / s for w, s in zip(widths, splits, strict=True)), default=0.0)
            return AdaptiveResidualDiagnostics(
                residual=residual,
                residual_sup=residual.mag,
                splits=tuple(splits),
                boxes=boxes,
                iterations=iterations,
                target=tgt,
                reached_target=reached,
                max_box_width=max_width,
            )
        splits = [min(max_splits, s * 2) for s in splits]


@dataclass(frozen=True)
class BoundaryFace:
    """One boundary patch: a (typically degenerate) box plus the target ``g``."""

    box: BoxLike
    target: Coeff = 0.0


def certified_boundary_residual(
    layers: Sequence[Layer],
    faces: Sequence[BoundaryFace],
    *,
    component: int = 0,
    splits: int | Sequence[int] = 1,
) -> Interval:
    r"""Rigorous enclosure of ``{ u_NN(x) - g(x) : x in dOmega }`` over the faces.

    Reads the order-0 value row of the certified jet on each (subdivided) face;
    ``certified_boundary_residual(...).mag`` is a certified sup-norm bound on the
    boundary mismatch.
    """
    if not faces:
        return Interval.point(0.0)
    pieces: list[Interval] = []
    for face in faces:
        fbox = _promote(face.box)
        per = _normalize_splits(splits, len(fbox))
        for sub in _grid(fbox, per):
            value = certified_partials(sub, layers, 0)[(0,) * len(fbox)][component]
            pieces.append(value - _eval_coeff(face.target, sub))
    return hull(list(pieces))


# --------------------------------------------------------------------------- #
# Certificates.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PINNErrorCertificate:
    """A sealed a-posteriori sup-norm error bound for a trained network."""

    error_bound: float
    interior_residual: float
    boundary_residual: float
    certificate: Cert
    diagnostics: AdaptiveResidualDiagnostics | None = None


def aposteriori_error_certificate(
    layers: Sequence[Layer],
    domain: BoxLike,
    pde: LinearPDE,
    *,
    boundary: Sequence[BoundaryFace] = (),
    stability_interior: float = 1.0,
    stability_boundary: float = 1.0,
    stability: StabilityEstimate | None = None,
    invariants: Sequence[StructuralInvariant] = (),
    model_metadata: Mapping[str, object] | None = None,
    max_error: float | None = None,
    diagnostics: AdaptiveResidualDiagnostics | None = None,
    splits: int | Sequence[int] = 1,
    boundary_splits: int | Sequence[int] = 1,
) -> PINNErrorCertificate:
    r"""Certified a-posteriori bound ``||u_NN - u_true||_inf <= C_O R_int + C_b R_bnd``.

    ``stability_interior`` (``C_Omega``) and ``stability_boundary``
    (``C_{partial Omega}``) must be rigorous constants for the linear, well-posed
    BVP ``pde``; they are recorded in the certificate, which is sealed and
    digest-verifiable.  Both residuals are computed rigorously from the certified
    jet, so the resulting bound is a true upper bound on the network's error.
    """
    estimate = stability or user_stability_estimate(
        stability_interior,
        stability_boundary,
        source="legacy_arguments",
        pde_family="linear",
        domain="caller_supplied",
    )
    if diagnostics is not None:
        r_int = diagnostics.residual
    else:
        r_int = certified_interior_residual(layers, domain, pde, splits=splits)
    r_bnd = certified_boundary_residual(
        layers, boundary, component=pde.component, splits=boundary_splits
    )
    r_int_sup = r_int.mag
    r_bnd_sup = r_bnd.mag
    err = (
        Interval.point(estimate.interior) * Interval.point(r_int_sup)
        + Interval.point(estimate.boundary) * Interval.point(r_bnd_sup)
    )
    from omnibias.core.proof.certificate import make_certificate

    payload: dict[str, object] = {
        "type": "pinn_aposteriori_error",
        "error_bound": err.hi,
        "interior_residual_sup": r_int_sup,
        "boundary_residual_sup": r_bnd_sup,
        "stability_interior": float(estimate.interior),
        "stability_boundary": float(estimate.boundary),
        "stability": estimate.to_payload(),
        "interior_enclosure": [r_int.lo, r_int.hi],
        "boundary_enclosure": [r_bnd.lo, r_bnd.hi],
        "pde_order": pde.required_order(),
        "n_boundary_faces": len(boundary),
        "invariants": [inv.to_payload() for inv in invariants],
    }
    if model_metadata is not None:
        payload["model"] = _as_payload_value(model_metadata)
    if diagnostics is not None:
        payload["adaptive_diagnostics"] = diagnostics.to_payload()
    if max_error is not None:
        if max_error < 0.0:
            raise ValueError("max_error must be non-negative")
        margin = Interval.point(float(max_error)) - err
        payload["finite_obligation"] = {
            "type": "error_bound_le_threshold",
            "threshold": float(max_error),
            "margin": [margin.lo, margin.hi],
            "note": (
                "Lean can only check this finite numerical inequality; it does "
                "not formalize the analytic PDE stability theorem."
            ),
        }

    cert = make_certificate(
        claim=(
            "a-posteriori sup-norm error bound "
            "||u_NN - u_true||_inf <= C_Omega * R_interior + C_boundary * R_boundary"
        ),
        payload=payload,
        honesty={
            "unproven_claim": False,
            "continuum_claim": False,
            "interval_verified": True,
            "pde_stability_user_obligation": not estimate.library_provided,
        },
    )
    return PINNErrorCertificate(err.hi, r_int_sup, r_bnd_sup, cert, diagnostics)


def pinn_aposteriori_schema_errors(cert: Mapping[str, object]) -> list[str]:
    """Validate the v1 ``pinn_aposteriori_error`` certificate payload."""
    from omnibias.core.proof.certificate import schema_errors_v1

    errors = schema_errors_v1(cert)
    payload = cert.get("payload")
    if not isinstance(payload, Mapping):
        return errors + ["payload must be a mapping"]
    if payload.get("type") != "pinn_aposteriori_error":
        errors.append("payload.type must be 'pinn_aposteriori_error'")

    required = (
        "error_bound",
        "interior_residual_sup",
        "boundary_residual_sup",
        "stability",
        "interior_enclosure",
        "boundary_enclosure",
        "pde_order",
        "n_boundary_faces",
    )
    for key in required:
        if key not in payload:
            errors.append(f"payload missing required field {key!r}")

    for key in ("error_bound", "interior_residual_sup", "boundary_residual_sup"):
        try:
            if float(payload.get(key, -1.0)) < 0.0:
                errors.append(f"payload.{key} must be non-negative")
        except (TypeError, ValueError):
            errors.append(f"payload.{key} must be numeric")

    stability = payload.get("stability")
    if not isinstance(stability, Mapping):
        errors.append("payload.stability must be a mapping with provenance")
    else:
        for key in ("interior", "boundary", "source", "pde_family", "domain"):
            if key not in stability:
                errors.append(f"payload.stability missing {key!r}")
        for key in ("interior", "boundary"):
            try:
                if float(stability.get(key, -1.0)) < 0.0:
                    errors.append(f"payload.stability.{key} must be non-negative")
            except (TypeError, ValueError):
                errors.append(f"payload.stability.{key} must be numeric")

    invariants = payload.get("invariants", [])
    if not isinstance(invariants, Sequence) or isinstance(invariants, str | bytes):
        errors.append("payload.invariants must be a sequence")
    else:
        for idx, inv in enumerate(invariants):
            if not isinstance(inv, Mapping):
                errors.append(f"payload.invariants[{idx}] must be a mapping")
                continue
            if inv.get("certified") and not inv.get("expression"):
                errors.append(f"payload.invariants[{idx}] certified invariant needs expression")

    finite = payload.get("finite_obligation")
    if finite is not None:
        if not isinstance(finite, Mapping):
            errors.append("payload.finite_obligation must be a mapping")
        elif finite.get("type") != "error_bound_le_threshold":
            errors.append("payload.finite_obligation.type must be 'error_bound_le_threshold'")
        elif "margin" not in finite:
            errors.append("payload.finite_obligation missing 'margin'")

    honesty = cert.get("honesty", {})
    if isinstance(honesty, Mapping) and bool(honesty.get("unproven_claim", False)):
        errors.append("honesty.unproven_claim must be False")
    return errors


def replay_pinn_aposteriori_certificate(cert: Mapping[str, object]) -> bool:
    """Independent arithmetic replay of the sealed a-posteriori error formula."""
    if pinn_aposteriori_schema_errors(cert):
        return False
    payload = cert["payload"]
    assert isinstance(payload, Mapping)
    stability = payload["stability"]
    assert isinstance(stability, Mapping)
    expected = (
        Interval.point(float(stability["interior"]))
        * Interval.point(float(payload["interior_residual_sup"]))
        + Interval.point(float(stability["boundary"]))
        * Interval.point(float(payload["boundary_residual_sup"]))
    ).hi
    return abs(expected - float(payload["error_bound"])) <= max(1e-12, 1e-12 * abs(expected))


def radii_polynomial_residual_certificate(
    residual_sup: float,
    z0: float,
    z1: float,
    z2: float,
    *,
    a_norm: float = 1.0,
    r_max: float = float("inf"),
    claim: str = "unique zero in closed ball B(x_bar, r)",
) -> RadiiCertificate | None:
    r"""Radii-polynomial existence from a certified residual as the NK defect.

    The Newton-Kantorovich defect is ``Y0 = ||A F(x_bar)|| <= ||A|| * residual_sup``
    (``a_norm = ||A||``, the approximate-inverse operator norm).  ``residual_sup``
    is any rigorous bound on the residual norm -- e.g. from
    :func:`certified_interior_residual` (its ``.mag``) or
    :func:`spectral_residual_norm` (its ``.hi``).  ``(Z0, Z1, Z2)`` are the
    contraction bounds of the linearisation (the caller's obligation); the result
    is ``None`` when no contracting radius exists.
    """
    if residual_sup < 0.0 or a_norm < 0.0:
        raise ValueError("residual_sup and a_norm must be non-negative")
    y0 = (Interval.point(float(a_norm)) * Interval.point(float(residual_sup))).hi
    return radii_polynomial_certificate(y0, z0, z1, z2, r_max=r_max, claim=claim)


# --------------------------------------------------------------------------- #
# Spectral route: residual of a band-limited Fourier ansatz.
# --------------------------------------------------------------------------- #
def spectral_residual_norm(
    ansatz: ValidatedFourierSeries,
    linear_symbol: Symbol,
    linear_tail_factor: float,
    *,
    quadratic: Callable[[ValidatedFourierSeries], ValidatedFourierSeries] | None = None,
    rhs: ValidatedFourierSeries | None = None,
) -> Interval:
    r"""Rigorous ``l1_nu`` norm of the residual ``F(a) = L a + Q(a) - rhs``.

    ``L`` is the diagonal Fourier multiplier ``linear_symbol`` (with
    ``linear_tail_factor`` a sound bound on ``sup_{|k|>N}|symbol(k)|``); ``Q`` is
    an optional quadratic term built from the validated algebra (e.g.
    ``lambda a: a * a`` for ``u^2``, or a Riesz/convolution advection); ``rhs`` is
    the forcing.  The returned interval's ``.hi`` is the natural ``Y0`` for
    :func:`radii_polynomial_residual_certificate`.
    """
    residual = ansatz.apply_multiplier(linear_symbol, linear_tail_factor)
    if quadratic is not None:
        residual = residual + quadratic(ansatz)
    if rhs is not None:
        residual = residual - rhs
    return residual.norm()


__all__ = [
    "AdaptiveResidualDiagnostics",
    "BoundaryFace",
    "CertifiedResidualCallback",
    "Coeff",
    "LinearPDE",
    "PINNErrorCertificate",
    "StabilityEstimate",
    "StructuralInvariant",
    "adaptive_certified_interior_residual",
    "advection_diffusion",
    "aposteriori_error_certificate",
    "certified_boundary_residual",
    "certified_custom_residual",
    "certified_interior_residual",
    "certified_quadratic_reaction_residual",
    "certified_streamfunction_divergence",
    "certified_vorticity_transport_residual",
    "helmholtz",
    "laplace",
    "pinn_aposteriori_schema_errors",
    "poisson",
    "radii_polynomial_residual_certificate",
    "replay_pinn_aposteriori_certificate",
    "screened_poisson",
    "spectral_residual_norm",
    "structural_invariant",
    "user_stability_estimate",
]
