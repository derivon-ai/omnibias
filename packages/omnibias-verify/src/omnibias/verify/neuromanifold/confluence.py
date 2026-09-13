# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Sound collision acceptance over declared spatial and integral observations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from omnibias.core.confluence import (
    Activation,
    cluster_remainder,
    derivative_bound,
    initialize_moments,
    pair_derivative_error,
    pair_series_error,
)
from omnibias.core.proof.certificate import Cert, interval_certificate, verify_certificate_digest
from omnibias.core.realization.transition import TransitionProposal, snapshot_digest
from omnibias.core.verified.interval import Interval

_ZERO = Interval.point(0.0)
_ONE = Interval.point(1.0)


@dataclass(frozen=True)
class ConfluenceCertificate:
    status: Literal["proved", "inconclusive"]
    errors: tuple[tuple[int, Interval], ...]
    budget: float
    certificate: Cert
    source_digest: str | None = None
    dependencies: tuple[str, ...] = ("Taylor theorem for the declared activation",)

    @property
    def accepted(self) -> bool:
        return self.status == "proved"


def certify_pair_collapse(
    *,
    domain: Interval,
    rho: Interval,
    m0: Interval,
    m1: Interval,
    weight: Interval = _ONE,
    bias: Interval = _ZERO,
    direction: Interval = _ZERO,
    direction_bias: Interval = _ONE,
    activation: Activation = "sigmoid",
    spatial_orders: tuple[int, ...] = (0,),
    error_budget: float = 1e-6,
) -> ConfluenceCertificate:
    """Uniform error from a finite pair to m0*sigma(z)+m1*eta*sigma'(z).

    Includes derivatives of affine eta(x), unlike a bias-only bound. Analytic
    Taylor hypotheses remain explicit; finite endpoint arithmetic can be
    replayed independently by the formal bridge.
    """
    if error_budget < 0 or not spatial_orders or any(n < 0 for n in spatial_orders):
        raise ValueError("nonnegative budget and nonempty nonnegative derivative orders required")
    eta = direction * domain + direction_bias
    errors = tuple(
        (
            n,
            pair_derivative_error(
                rho=rho,
                eta=eta,
                m0=m0,
                m1=m1,
                weight=weight,
                direction=direction,
                spatial_order=n,
                activation=activation,
            ),
        )
        for n in spatial_orders
    )
    maximum = max(error.hi for _, error in errors)
    status: Literal["proved", "inconclusive"] = (
        "proved" if maximum <= error_budget else "inconclusive"
    )
    operands = {
        name: [value.lo, value.hi]
        for name, value in [
            ("domain", domain),
            ("rho", rho),
            ("m0", m0),
            ("m1", m1),
            ("weight", weight),
            ("bias", bias),
            ("direction", direction),
            ("direction_bias", direction_bias),
        ]
    }
    cert = interval_certificate(
        "uniform pair-to-derivative observation error",
        Interval(0.0, maximum),
        meta={
            "kind": "confluence",
            "operands": operands,
            "activation": activation,
            "orders": list(spatial_orders),
            "error_budget": error_budget,
            "status": status,
            "errors": [[n, e.lo, e.hi] for n, e in errors],
        },
    )
    return ConfluenceCertificate(status, errors, error_budget, cert)


def linear_residual_error(coefficients: Sequence[float], errors: Sequence[Interval]) -> Interval:
    """Propagate spatial-jet errors through a fixed linear PDE residual."""
    if len(coefficients) != len(errors):
        raise ValueError("one coefficient per derivative error required")
    total = _ZERO
    for coefficient, error in zip(coefficients, errors, strict=True):
        total += abs(coefficient) * Interval.point(error.mag)
    return Interval(0.0, total.hi)


def nonlinear_residual_error(
    residual: Callable[[tuple[Interval, ...]], Interval],
    reference: tuple[Interval, ...],
    errors: tuple[Interval, ...],
) -> Interval:
    """Sound interval product-rule consumer; dependency loss may be inconclusive."""
    if len(reference) != len(errors):
        raise ValueError("one error per residual input required")
    perturbed = tuple(x + Interval(-e.mag, e.mag) for x, e in zip(reference, errors, strict=True))
    difference = residual(perturbed) - residual(reference)
    return Interval(0.0, difference.mag)


def integral_error(
    error: Interval,
    mass: Interval,
    *,
    kind: Literal["activation_window", "domain_quadrature", "measure"],
) -> Interval:
    """Bound an integral error by a nonnegative total mass/window length.

    A quadrature result concerns the declared quadrature rule; a continuum
    integral additionally needs its quadrature error. Activation windows instead
    integrate the activation analytically and need no domain quadrature claim.
    """
    if kind not in ("activation_window", "domain_quadrature", "measure") or mass.lo < 0:
        raise ValueError("explicit integral semantics and nonnegative mass required")
    return Interval(0.0, (Interval.point(error.mag) * mass).hi)


def _values(snapshot: Mapping[str, Any], name: str) -> list[Any]:
    return list(snapshot[name]["values"])


def certify_transition(
    snapshot: dict[str, Any],
    proposal: TransitionProposal,
    *,
    domain: Interval,
    activation: Activation = "sigmoid",
    spatial_orders: tuple[int, ...] = (0,),
) -> ConfluenceCertificate:
    """Replay supported proposal operands; a resealed unrelated update is rejected.

    Source snapshots are supplied by ConfluentPackBank.portable_snapshot(). The
    source digest includes all tensors, IDs, modes, and the transition version.
    A certified acceptance callback must also compare the live snapshot at commit.
    """
    if snapshot_digest(snapshot) != proposal.source_digest:
        raise ValueError("proposal is not bound to the supplied source snapshot")
    if snapshot["__configuration__"]["activation"] != activation:
        raise ValueError("activation differs from the bound source configuration")
    if int(_values(snapshot, "transition_version")[0]) != proposal.source_version:
        raise ValueError("stale version")
    if not spatial_orders or any(n < 0 for n in spatial_orders):
        raise ValueError("nonempty nonnegative spatial derivative orders required")
    ids = [int(i) for i in _values(snapshot, "slot_ids")]
    slots = [ids.index(i) for i in proposal.slot_ids]
    active = _values(snapshot, "active")
    modes = _values(snapshot, "modes")
    weights = _values(snapshot, "weights")
    updates = {(u.name, u.index): u.value for u in proposal.updates}
    if len(updates) != len(proposal.updates):
        raise ValueError("duplicate update operands")
    errors: tuple[tuple[int, Interval], ...]
    switch_crossing = False

    def series_error(
        rho: Interval,
        m0: Interval,
        m1: Interval,
        weight: Interval,
        direction: Interval,
        direction_bias: Interval,
        n: int,
    ) -> Interval:
        nonlocal switch_crossing
        config = snapshot["__configuration__"]
        terms, radius = config["series_terms"], config["series_radius"]
        if (
            type(terms) is not int
            or terms < 1
            or not isinstance(radius, int | float)
            or not 0 < radius <= 0.1
        ):
            raise ValueError("invalid source series configuration")
        eta = direction * domain + direction_bias
        squared = rho * eta**2
        if squared.lo > radius**2 or rho.hi == 0:
            return _ZERO
        if (
            squared.lo <= radius**2 <= squared.hi
            and n > 0
            and not direction.lo == direction.hi == 0
        ):
            switch_crossing = True
        a, b = pair_series_error(
            rho=rho,
            eta=eta,
            terms=terms,
            activation=activation,
            spatial_order=n,
            weight=weight,
            direction=direction,
        )
        total = Interval.point(m0.mag) * a + Interval.point(m1.mag) * b
        return Interval(0, total.hi)

    if proposal.kind == "pair_to_derivative":
        if len(slots) != 1 or updates != {("rho", slots[0]): 0.0}:
            raise ValueError("unrecognized derivative transition operands")
        i = slots[0]
        if not active[i] or modes[i] != 1:
            raise ValueError("source is not a live centered pair")

        def point(name: str) -> Interval:
            return Interval.point(float(_values(snapshot, name)[i]))

        result = certify_pair_collapse(
            domain=domain,
            rho=point("rho"),
            m0=point("weights"),
            m1=point("odd_moments"),
            weight=point("scales"),
            bias=-point("scales") * point("centers"),
            direction=point("directions"),
            direction_bias=point("direction_bias"),
            activation=activation,
            spatial_orders=spatial_orders,
            error_budget=proposal.error_budget,
        )
        combined = []
        for n, error in result.errors:
            truncation = series_error(
                point("rho"),
                point("weights"),
                point("odd_moments"),
                point("scales"),
                point("directions"),
                point("direction_bias"),
                n,
            )
            # Adding the exact zero identity must preserve an attained,
            # zero-budget transition rather than introduce a rounding radius.
            combined.append((n, error + truncation if truncation.hi != 0 else error))
        errors = tuple(combined)
    elif proposal.kind in ("centered_pair", "duplicate_merge"):
        if len(slots) != 2:
            raise ValueError("two source slots required")
        i, j = slots
        orders = _values(snapshot, "orders")
        scales = _values(snapshot, "scales")
        centers = _values(snapshot, "centers")
        if (
            not active[i]
            or not active[j]
            or modes[i] != 0
            or modes[j] != 0
            or orders[i] != 0
            or orders[j] != 0
            or scales[i] != scales[j]
            or scales[i] == 0
        ):
            raise ValueError("incompatible source pair")
        scale = Fraction(scales[i])
        bi, bj = -scale * Fraction(centers[i]), -scale * Fraction(centers[j])
        ai, aj = Fraction(weights[i]), Fraction(weights[j])
        h, mean = abs(bi - bj) / 2, (bi + bj) / 2
        m0, m1 = ai + aj, (bi - bj) * (ai - aj) / 2
        expected = {
            ("centers", i): float(-mean / scale),
            ("weights", i): float(m0),
            ("rho", i): float(h * h),
            ("odd_moments", i): float(m1),
            ("modes", i): 1.0,
            ("directions", i): 0.0,
            ("direction_bias", i): 1.0,
            ("active", j): 0.0,
            ("weights", j): 0.0,
            ("odd_moments", j): 0.0,
        }
        if updates != expected:
            raise ValueError("proposal does not replay the centered coordinate transformation")
        # Exact coordinate identity followed by independently enclosed conversion
        # errors. Divided differences are smooth in rho, with global derivative
        # bounds M2/2 and M3/6 from the even Taylor integral representation.
        dm0 = Interval.from_rational(abs(m0 - Fraction(expected["weights", i])))
        dm1 = Interval.from_rational(abs(m1 - Fraction(expected["odd_moments", i])))
        dz = Interval.from_rational(abs(-scale * Fraction(expected["centers", i]) - mean))
        dr = Interval.from_rational(abs(Fraction(expected["rho", i]) - h * h))
        am0 = Interval.from_rational(max(abs(m0), abs(Fraction(expected["weights", i]))))
        am1 = Interval.from_rational(max(abs(m1), abs(Fraction(expected["odd_moments", i]))))
        errors_list = []
        for n in spatial_orders:
            if n < 0:
                raise ValueError("negative derivative order")

            def bound(k: int, order: int = n) -> Interval:
                return derivative_bound(activation, order + k)

            error = (
                dm0 * bound(0)
                + dm1 * bound(1)
                + dz * (am0 * bound(1) + am1 * bound(2))
                + dr * (am0 * bound(2) / 2 + am1 * bound(3) / 6)
            ) * Interval.from_rational(abs(scale) ** n)
            exact_conversion = all(e.lo == e.hi == 0 for e in (dm0, dm1, dz, dr))
            stored_error = _ZERO if exact_conversion else Interval(0.0, error.hi)
            truncation = series_error(
                Interval.point(expected["rho", i]),
                Interval.point(expected["weights", i]),
                Interval.point(expected["odd_moments", i]),
                Interval.from_rational(scale),
                _ZERO,
                _ONE,
                n,
            )
            total = stored_error + truncation if truncation.hi != 0 else stored_error
            errors_list.append((n, total))
        errors = tuple(errors_list)
    elif proposal.kind == "cluster_to_moments":
        if not slots:
            raise ValueError("nonempty cluster required")
        first = slots[0]
        scales = _values(snapshot, "scales")
        centers = _values(snapshot, "centers")
        orders = _values(snapshot, "orders")
        scale = Fraction(scales[first])
        if scale == 0 or any(
            not active[i] or modes[i] != 0 or orders[i] != 0 or scales[i] != scales[first]
            for i in slots
        ):
            raise ValueError("incompatible ordinary common-weight cluster")
        supplied_order = updates.get(("orders", first))
        if not isinstance(supplied_order, int | float) or int(supplied_order) != supplied_order:
            raise ValueError("integer moment order required")
        order = int(supplied_order)
        max_order = int(snapshot["__configuration__"]["max_order"])
        if not 0 <= order <= max_order:
            raise ValueError("moment order exceeds source capacity")
        center = sum((Fraction(centers[i]) for i in slots), Fraction()) / len(slots)
        initialization = initialize_moments(
            tuple(-scale * Fraction(centers[i]) for i in slots),
            tuple(Fraction(weights[i]) for i in slots),
            center=-scale * center,
            order=order,
        )
        row = initialization.moments + (0.0,) * (max_order - order)
        expected_cluster: dict[tuple[str, int], float | tuple[float, ...]] = {
            ("centers", first): float(center),
            ("modes", first): 2.0,
            ("orders", first): float(order),
            ("moment_coefficients", first): row,
        }
        for i in slots[1:]:
            expected_cluster["active", i] = expected_cluster["weights", i] = 0.0
        if updates != expected_cluster:
            raise ValueError("moment operands do not replay the original stored cluster")
        dz = Interval.from_rational(abs(scale * (Fraction(float(center)) - center)))
        errors_list = []
        for n in spatial_orders:
            error = cluster_remainder(
                initialization, activation=activation, spatial_order=n, weight=float(scale)
            )
            shift = _ZERO
            for k, moment in enumerate(initialization.moments):
                shift += abs(moment) * derivative_bound(activation, n + k + 1) * dz
            error = error + shift * Interval.from_rational(abs(scale) ** n)
            errors_list.append((n, Interval(0.0, error.hi)))
        errors = tuple(errors_list)
    elif proposal.kind == "zero_output_birth":
        if len(slots) != 1 or active[slots[0]]:
            raise ValueError("birth requires one inactive source slot")
        i = slots[0]
        required = {
            "centers",
            "weights",
            "scales",
            "orders",
            "ages",
            "active",
            "modes",
            "rho",
            "odd_moments",
            "directions",
            "direction_bias",
            "moment_coefficients",
        }
        if set(updates) != {(name, i) for name in required}:
            raise ValueError("birth must initialize every representation field")
        for name in ("weights", "modes", "rho", "odd_moments", "directions"):
            if updates[name, i] != 0:
                raise ValueError("birth must have zero output and ordinary mode")
        if (
            updates["active", i] != 1
            or updates["direction_bias", i] != 1
            or updates["moment_coefficients", i]
            != (0.0,) * (snapshot["__configuration__"]["max_order"] + 1)
        ):
            raise ValueError("invalid birth initialization")
        import math

        for name in ("centers", "scales", "orders", "ages"):
            value = updates[name, i]
            if not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError("finite scalar birth parameters required")
        birth_order, birth_age = updates["orders", i], updates["ages", i]
        if (
            not isinstance(birth_order, int | float)
            or not isinstance(birth_age, int | float)
            or int(birth_order) != birth_order
            or not 0 <= birth_order <= snapshot["__configuration__"]["max_order"]
            or int(birth_age) != birth_age
            or birth_age < 0
        ):
            raise ValueError("birth order/age exceed the supported integer range")
        errors = tuple((n, _ZERO) for n in spatial_orders)
    elif proposal.kind == "remove_zero":
        if (
            len(slots) != 1
            or updates != {("active", slots[0]): 0.0}
            or weights[slots[0]] != 0
            or modes[slots[0]] != 0
        ):
            raise ValueError("source is not an exact zero-output ordinary atom")
        errors = tuple((n, _ZERO) for n in spatial_orders)
    else:
        raise NotImplementedError("transition family is not yet certified")
    maximum = max(error.hi for _, error in errors)
    status: Literal["proved", "inconclusive"] = (
        "proved" if maximum <= proposal.error_budget and not switch_crossing else "inconclusive"
    )
    certificate = interval_certificate(
        "operand-bound structural transition error",
        Interval(0, maximum),
        meta={
            "kind": "confluence_transition",
            "source": snapshot,
            "proposal": proposal.to_json(),
            "domain": [domain.lo, domain.hi],
            "activation": activation,
            "orders": list(spatial_orders),
            "errors": [[n, e.lo, e.hi] for n, e in errors],
            "runtime_series_included": True,
            "spatial_series_switch_crossing": switch_crossing,
            "scope": "analytic operator and declared series truncation; backend rounding excluded",
            "status": status,
            "error_budget": proposal.error_budget,
        },
    )
    return ConfluenceCertificate(
        status, errors, proposal.error_budget, certificate, proposal.source_digest
    )


def replay_confluence_certificate(certificate: Mapping[str, Any]) -> bool:
    """Recompute the analytic error formula from sealed original operands."""
    if not verify_certificate_digest(certificate):
        return False
    try:
        meta = certificate["meta"]
        if meta["kind"] == "confluence_transition":
            rebuilt = certify_transition(
                meta["source"],
                TransitionProposal.from_json(meta["proposal"]),
                domain=Interval(*meta["domain"]),
                activation=meta["activation"],
                spatial_orders=tuple(meta["orders"]),
            )
        elif meta["kind"] == "confluence":
            operands = {name: Interval(*value) for name, value in meta["operands"].items()}
            rebuilt = certify_pair_collapse(
                **operands,
                activation=meta["activation"],
                spatial_orders=tuple(meta["orders"]),
                error_budget=meta["error_budget"],
            )
        else:
            return False
        return rebuilt.certificate == certificate
    except (KeyError, ValueError, TypeError, ZeroDivisionError, IndexError, NotImplementedError):
        return False


__all__ = [
    "ConfluenceCertificate",
    "certify_pair_collapse",
    "certify_transition",
    "integral_error",
    "linear_residual_error",
    "nonlinear_residual_error",
    "replay_confluence_certificate",
]
