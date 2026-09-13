# SPDX-License-Identifier: Apache-2.0
"""Exact finite controls for stochastic Yang--Mills research.

These computations distinguish a numerical sampler, a reduced matrix model,
and a continuum gauge-orbit process. They do not construct the latter.
All derivatives below are polynomial identities, with rational arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest


def rational(value: int | Q, name: str) -> Q:
    """Accept exact data only; bool and rounded floating proposals are refused."""
    if type(value) is not int and not isinstance(value, Q):
        raise TypeError(f"{name} must be an integer or Fraction")
    return Q(value)


def _positive_integer(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seal(kind: str, inputs: dict[str, Any], arithmetic: dict[str, Any]) -> dict[str, Any]:
    payload = {"type": kind, "inputs": inputs, "arithmetic": arithmetic}
    return {
        **payload,
        "certificate": make_certificate(claim="exact finite stochastic control arithmetic", payload=payload, meta={"transcend_backend": "not_used"}),
        "method": "EXACT_RATIONAL",
        "theorem_prover_verified": False,
        "mathlib_verified": False,
        "continuum_claim": False,
        "yang_mills_mass_gap_claim": False,
    }


def ou_euler_stationarity(rate: int | Q, temperature: int | Q, step: int | Q) -> dict[str, Any]:
    """Exact stationary second moment of X'=(1-h*lambda)X+sqrt(2*h*T)Z.

    Z has mean zero, variance one, and is independent of X. This is a
    covariance identity, not a simulation-based invariant-measure check.
    """
    lam, temp, h = (rational(x, name) for x, name in
                    ((rate, "rate"), (temperature, "temperature"), (step, "step")))
    if lam <= 0 or temp <= 0 or h <= 0:
        raise ValueError("rate, temperature and step must be positive")
    multiplier = 1-h*lam
    stable = abs(multiplier) < 1
    invariant = temp/lam
    variance = 2*h*temp/(1-multiplier**2) if stable else None
    return _seal("ou_euler_stationarity_v1",
                 {"rate": str(lam), "temperature": str(temp), "step": str(h)}, {
        "stable": stable, "multiplier": str(multiplier),
        "continuous_stationary_variance": str(invariant),
        "euler_stationary_variance": str(variance) if variance is not None else None,
        "variance_bias": str(variance-invariant) if variance is not None else None,
        "stationarity_residual": str(variance-multiplier**2*variance-2*h*temp)
        if variance is not None else None,
        "target_variance_one_step_defect": str(multiplier**2*invariant+2*h*temp-invariant),
        "finite_step_target_invariant": False,
    })


def _vectors(connection: Sequence[Sequence[int | Q]]) -> tuple[tuple[Q, ...], ...]:
    vectors = tuple(tuple(rational(v, "connection entry") for v in a) for a in connection)
    if len(vectors) < 2 or any(len(a) != 3 for a in vectors):
        raise ValueError("at least two su(2) coordinate vectors of length three are required")
    return vectors


def _dot(a: Sequence[Q], b: Sequence[Q]) -> Q:
    return sum((x*y for x, y in zip(a, b, strict=True)), Q(0))


def _cross(a: Sequence[Q], b: Sequence[Q]) -> tuple[Q, Q, Q]:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def homogeneous_su2_action(connection: Sequence[Sequence[int | Q]]) -> Q:
    """S=1/2 sum_(i<j)|A_i cross A_j|² on the unreduced R^(3d) matrix model."""
    vectors = _vectors(connection)
    return sum((_dot(c, c)/2 for i, a in enumerate(vectors)
                for b in vectors[i+1:] for c in [_cross(a, b)]), Q(0))


def homogeneous_su2_gradient(connection: Sequence[Sequence[int | Q]]) -> tuple[tuple[Q, ...], ...]:
    """Closed polynomial action gradient; no nested autodiff or differencing."""
    vectors = _vectors(connection)
    return tuple(tuple(sum((_dot(b, b)*a[k]-_dot(a, b)*b[k]
                            for j, b in enumerate(vectors) if i != j), Q(0))
                       for k in range(3)) for i, a in enumerate(vectors))


def homogeneous_su2_radial_generator(
    connection: Sequence[Sequence[int | Q]], *, power: int = 1, temperature: int | Q = 1,
) -> dict[str, Any]:
    """Evaluate L(1+|A|²)^p for L=T*Delta-grad(S).grad, exactly."""
    p = _positive_integer(power, "power")
    temp = rational(temperature, "temperature")
    if temp <= 0:
        raise ValueError("temperature must be positive")
    vectors = _vectors(connection)
    radius = sum((_dot(a, a) for a in vectors), Q(0))
    action = homogeneous_su2_action(vectors)
    gradient = homogeneous_su2_gradient(vectors)
    euler = sum((_dot(a, g) for a, g in zip(vectors, gradient, strict=True)), Q(0))
    noether = tuple(sum((_cross(a, g)[k] for a, g in zip(vectors, gradient, strict=True)), Q(0))
                    for k in range(3))
    laplacian = 2*p*len(vectors)*3*(1+radius)**(p-1)
    if p > 1:
        laplacian += 4*p*(p-1)*radius*(1+radius)**(p-2)
    generator = temp*laplacian-8*p*action*(1+radius)**(p-1)
    return _seal("homogeneous_su2_radial_generator_v1", {
        "connection": [[str(v) for v in a] for a in vectors], "power": p, "temperature": str(temp),
    }, {"squared_radius": str(radius), "action": str(action), "value": str((1+radius)**p),
        "generator": str(generator), "euler_homogeneity_residual": str(euler-4*action),
        "global_gauge_noether_residual": [str(v) for v in noether],
        "scope": "unreduced homogeneous matrix model; not lattice holonomies or continuum gauge orbits"})


def reject_radial_lyapunov(
    *, dimension: int = 3, power: int = 1, constant: int | Q = 1,
    contraction: int | Q = 1, temperature: int | Q = 1,
) -> dict[str, Any]:
    """Construct a commuting rational counterexample to LW <= C-cW.

    This rejects this particular unreduced radial ansatz for any supplied
    C>=0,c>0. It does not reject a compact-holonomy or gauge-orbit functional.
    """
    d = _positive_integer(dimension, "dimension")
    p = _positive_integer(power, "power")
    if d < 2:
        raise ValueError("dimension must be at least two")
    c0, c, temp = (rational(x, name) for x, name in
                   ((constant, "constant"), (contraction, "contraction"), (temperature, "temperature")))
    if c0 < 0 or c <= 0 or temp <= 0:
        raise ValueError("constant>=0, contraction>0 and temperature>0 are required")
    r = 1
    while c*(1+r*r)**p <= c0:
        r *= 2
    vectors = [[r, 0, 0], *[[0, 0, 0] for _ in range(d-1)]]
    evaluated = homogeneous_su2_radial_generator(vectors, power=p, temperature=temp)["arithmetic"]
    margin = Q(evaluated["generator"])-c0+c*Q(evaluated["value"])
    return _seal("radial_lyapunov_counterexample_v1", {
        "dimension": d, "power": p, "constant": str(c0), "contraction": str(c), "temperature": str(temp),
    }, {"point": vectors, "action": evaluated["action"], "value": evaluated["value"],
        "generator": evaluated["generator"], "violation_margin": str(margin),
        "candidate_status": "DISPROVED" if margin > 0 else "INCONCLUSIVE",
        "continuum_orbit_lyapunov_disproved": False})


def stochastic_power_count(dimension: int, *, loss: int | Q = Q(1, 1000)) -> dict[str, Any]:
    """Formal local-subcriticality test, not existence of a stochastic model."""
    d = _positive_integer(dimension, "dimension")
    eps = rational(loss, "loss")
    if eps <= 0:
        raise ValueError("loss must be positive")
    alpha = Q(2-d, 2)-eps
    quadratic_gain, cubic_gain = alpha+1, 2*alpha+2
    return _seal("stochastic_ym_power_count_v1", {"dimension": d, "loss": str(eps)}, {
        "noise_homogeneity": str(-Q(d+2, 2)-eps), "field_homogeneity": str(alpha),
        "integrated_quadratic_gain": str(quadratic_gain), "integrated_cubic_gain": str(cubic_gain),
        "locally_subcritical_power_count": quadratic_gain > 0 and cubic_gain > 0,
        "regularity_structure_constructed": False,
    })


def replay_finite_stochastic_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute all arithmetic; a resealed altered result still fails replay."""
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        raw, kind = payload["inputs"], payload["type"]
        if kind == "ou_euler_stationarity_v1":
            expected = ou_euler_stationarity(Q(raw["rate"]), Q(raw["temperature"]), Q(raw["step"]))
        elif kind == "homogeneous_su2_radial_generator_v1":
            expected = homogeneous_su2_radial_generator([[Q(x) for x in a] for a in raw["connection"]],
                         power=raw["power"], temperature=Q(raw["temperature"]))
        elif kind == "radial_lyapunov_counterexample_v1":
            expected = reject_radial_lyapunov(dimension=raw["dimension"], power=raw["power"],
                         constant=Q(raw["constant"]), contraction=Q(raw["contraction"]), temperature=Q(raw["temperature"]))
        elif kind == "stochastic_ym_power_count_v1":
            expected = stochastic_power_count(raw["dimension"], loss=Q(raw["loss"]))
        else:
            return False
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = ["homogeneous_su2_action", "homogeneous_su2_gradient", "homogeneous_su2_radial_generator",
           "ou_euler_stationarity", "reject_radial_lyapunov", "replay_finite_stochastic_certificate",
           "stochastic_power_count"]
