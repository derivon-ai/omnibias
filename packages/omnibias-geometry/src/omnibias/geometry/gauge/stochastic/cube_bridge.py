# SPDX-License-Identifier: Apache-2.0
"""Original-link cube geometry, exact controls and conditional feedback budgets.

The linear-character control is a positive compact-group probability law.
It is deliberately distinct from the heat-kernel trial and the physical vacuum.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.stochastic.finite import rational
from omnibias.geometry.gauge.stochastic.lattice import (
    Quaternion,
    _product,
    _unit,
    quaternion_inverse,
    quaternion_product,
)


def cube_bridge_geometry() -> dict[str, Any]:
    """A cube with the z=0 bottom held old and eight fresh edges attached."""
    edges = [(v, v ^ bit) for v in range(8) for bit in (1, 2, 4) if not v & bit]
    index = {edge: i+1 for i, edge in enumerate(edges)}
    faces = []
    for a, b in ((1, 2), (1, 4), (2, 4)):
        for v in range(8):
            if v & (a | b):
                continue
            vertices = [v, v ^ a, v ^ a ^ b, v ^ b]
            faces.append([index[(u, w)] if u < w else -index[(w, u)]
                          for u, w in zip(vertices, [*vertices[1:], vertices[0]], strict=True)])
    old = [i+1 for i, (u, v) in enumerate(edges) if u < 4 and v < 4]
    new = [i+1 for i in range(12) if i+1 not in old]
    incidences = {"old": sum(abs(e) in old for face in faces[1:] for e in face),
                  "new": sum(abs(e) in new for face in faces[1:] for e in face)}
    return {"vertices": 8, "edges": edges, "bottom": faces[0], "new_faces": faces[1:],
            "old_edges": old, "new_edges": new, "face_incidences": incidences,
            "independent_new_holonomies": 4, "new_vertices": 4}


def cube_disk_geometry(old_face_count: int = 2) -> dict[str, Any]:
    """Corner-adjacent old faces and their complementary disk on one cube.

    Counts1,2,3 retain one, two adjacent, or three corner-adjacent old faces.
    The boundary is a simple original-link loop, not a product face metric.
    """
    if type(old_face_count) is not int or old_face_count not in (1, 2, 3):
        raise ValueError("old_face_count must be1,2 or3")
    g = cube_bridge_geometry()
    faces = [g["bottom"], *g["new_faces"]]
    chosen = [0, 2, 4][:old_face_count]
    old_faces = [faces[j] for j in chosen]
    new_faces = [face for j, face in enumerate(faces) if j not in chosen]
    old_edges = sorted({abs(e) for face in old_faces for e in face})
    new_edges = [e for e in range(1, 13) if e not in old_edges]
    boundary_edges = [e for e in old_edges if sum(abs(token) == e for face in old_faces for token in face) == 1]
    edges = g["edges"]
    neighbors: dict[int, list[tuple[int, int]]] = {}
    for e in boundary_edges:
        u, v = edges[e-1]
        neighbors.setdefault(u, []).append((v, e))
        neighbors.setdefault(v, []).append((u, -e))
    start = min(neighbors)
    vertex, previous = start, -1
    boundary = []
    while True:
        nxt, token = min((v, token) for v, token in neighbors[vertex] if v != previous)
        boundary.append(token)
        previous, vertex = vertex, nxt
        if vertex == start:
            break
    old_vertices = {v for e in old_edges for v in edges[e-1]}
    return {"edges": edges, "old_faces": old_faces, "new_faces": new_faces,
            "old_edges": old_edges, "new_edges": new_edges,
            "boundary": boundary, "new_vertices": 8-len(old_vertices),
            "face_incidences": {"old": sum(abs(e) in old_edges for face in new_faces for e in face),
                                "new": sum(abs(e) in new_edges for face in new_faces for e in face)}}


def conditional_two_face_seam_budget(*, kappa_max: int | Q = Q(1, 64)) -> dict[str, Any]:
    """Exact m=4,b=6 consequence of the written two-face seam proof.

    This finite rational obligation leaves the underlying Haar, heat and
    two-boundary-action comparisons explicitly outside the certificate.
    """
    km = rational(kappa_max, "kappa_max")
    if not 0 < km <= Q(1, 64):
        raise ValueError("kappa_max must be in(0,1/64]")
    theta = Q(35, 512)*Q(484, 49)
    constant = Q(759, 32)+Q(35, 8)*km
    return _seal("conditional_two_face_seam_budget_v1", {"kappa_max": str(km)}, {
        "geometry": cube_disk_geometry(2), "theta_upper": str(theta),
        "constant_upper": str(constant), "contractive_arithmetic": theta < Q(3, 4),
        "constant_below24": constant < 24,
        "fisher_identity": "I=4(F-a)+(3/2)(a-G)",
        "boundary_action_bound": "u_D^2 <= (pi^2/4)(A(B1)+A(B2))",
        "external_premises": ["four-face Haar convolution and original-link Fisher identity",
            "uniform small-time heat-score theorem and SU2 Li-Yau",
            "two-face boundary product and angle-action inequality"],
        "physical_form_comparison_verified": False})


def three_face_strict_feedback_obstruction(
    *, constant: int | Q = 100, theta: int | Q = Q(3, 4),
) -> dict[str, Any]:
    """Refute the constant-plus-strict-action budget for a normalized real fiber.

    At boundary -I, three old and three added faces each have minimum total
    action3. The written geometric proof gives W>=6/kappa, independent of
    the trial profile. This certificate replays the exact violating budget.
    It makes no assertion about differences of optimized ground energies.
    """
    c, th = rational(constant, "constant"), rational(theta, "theta")
    if c < 0 or not 0 <= th < 1:
        raise ValueError("constant>=0 and 0<=theta<1 required")
    kap = min(Q(1, 64), 3*(1-th)/(c+1))
    lower, upper = 6/kap, c+6*th/kap
    return _seal("three_face_strict_feedback_obstruction_v1", {
        "constant": str(c), "theta": str(th)}, {
        "candidate_status": "DISPROVED",
        "candidate_scope": "uniform scalar compression budget for smooth normalized real fibers; unchanged old multiplication terms",
        "geometry": cube_disk_geometry(3), "boundary_trace": "-2",
        "old_total_action": "3", "new_total_action_minimum": "3",
        "counterexample_kappa": str(kap), "energy_lower": str(lower),
        "proposed_energy_upper": str(upper), "violation_margin": str(lower-upper),
        "written_geometric_input": "X1 X2 X3=-I implies sum(2-Tr Xi)>=3; equal pi/3 rotations attain3",
        "optimized_ground_energy_comparison_disproved": False,
        "mass_gap_disproved": False})


def _seal(kind: str, inputs: dict[str, Any], arithmetic: dict[str, Any]) -> dict[str, Any]:
    payload = {"type": kind, "inputs": inputs, "arithmetic": arithmetic}
    return {**payload,
        "certificate": make_certificate(claim="exact cube control or explicitly conditional budget",
            payload=payload, meta={"transcend_backend": "not_used"}),
        "register": "EXACT_RATIONAL", "theorem_prover_verified": False,
        "mathlib_verified": False, "continuum_claim": False,
        "yang_mills_mass_gap_claim": False}


def conditional_cube_feedback_budget(
    *, scale: int | Q = 1, kappa_max: int | Q = Q(1, 64),
    lower_loss: int | Q = 3, upper_loss: int | Q = Q(3, 2),
    upper_constant: int | Q = Q(1, 2),
) -> dict[str, Any]:
    """Rational consequence of uniform heat time-score and Li--Yau estimates.

    Assumptions on 0<s<=5*scale*kappa_max are
    u²/s²-L/s <= partial_s log K_s(u) <= u²/s²-D/s+U.
    This builder checks arithmetic, not those pointwise analytic assumptions.
    The angle comparison uses pi²<(22/7)² in the accompanying written proof.
    """
    c, km, low, loss, upper = (rational(x, name) for x, name in (
        (scale, "scale"), (kappa_max, "kappa_max"), (lower_loss, "lower_loss"),
        (upper_loss, "upper_loss"), (upper_constant, "upper_constant")))
    if c <= 0 or km <= 0 or low < 0 or loss < 0 or upper < 0 or low < loss/5:
        raise ValueError("positive scales and nonnegative compatible score budgets required")
    theta = Q(484, 49)*(Q(2, 5)+1/(50*c*c))/8
    constant = 10*(low-loss/5)*c+(Q(15, 4)-loss/10)/c+km*upper*(10*c*c+Q(1, 2))
    return _seal("conditional_cube_feedback_budget_v1", {
        "scale": str(c), "kappa_max": str(km), "lower_loss": str(low),
        "upper_loss": str(loss), "upper_constant": str(upper)}, {
        "theta_upper": str(theta), "constant_upper": str(constant),
        "heat_time_max": str(5*c*km), "contractive_arithmetic": theta < 1,
        "target_theta_three_quarters_passed": theta <= Q(3, 4),
        "target_constant_31_passed": constant <= 31,
        "external_premises": ["global heat time-score bounds on the declared interval",
            "SU2 Li-Yau inequality with Casimir normalization3/4",
            "five-face Haar normalization and original-link Fisher identity",
            "angle-action comparison pi²<(22/7)²"],
        "physical_form_comparison_verified": False})


def linear_character_cube_control(
    boundary_half_trace: int | Q, *, tilt: int | Q = Q(1, 3), kappa: int | Q = 1,
) -> dict[str, Any]:
    """Exact integrated cube trial with f(U)=1+c TrU, c=r/(1+r²), |r|<1.

    The square root sqrt(1-4c²)=(1-r²)/(1+r²) is rational. Character
    orthogonality and the SU2 semicircle integral give the complete Fisher
    and magnetic expectation, not a truncated character proposal.
    """
    x, r, kap = (rational(z, name) for z, name in (
        (boundary_half_trace, "boundary_half_trace"), (tilt, "tilt"), (kappa, "kappa")))
    if not -1 <= x <= 1 or not -1 < r < 1 or kap <= 0:
        raise ValueError("half trace in[-1,1], tilt in(-1,1) and kappa>0 required")
    c = r/(1+r*r)
    root = (1-r*r)/(1+r*r)
    inverse_mean = 2/(1+root)
    r0 = Q(1, 4)+(c*c-Q(1, 4))*inverse_mean
    r1 = (3*c*c/4-r0)/c if c else Q(0)
    chi, coeff = 2*x, c**5/16
    z = 1+coeff*chi
    fisher = (r0+r1*c**4*chi/16)/z
    lap = -Q(3, 4)*coeff*chi/z
    score = coeff**2*(1-chi*chi/4)/(z*z)
    mu = (2-c+(2*c-1)*c**4*chi/16)/z
    inew, iold = 4*(fisher-lap), fisher-score
    return _seal("linear_character_cube_control_v1", {
        "boundary_half_trace": str(x), "tilt": str(r), "kappa": str(kap)}, {
        "profile": "f(U)=1+c TrU; compact positive control, not heat kernel or actual vacuum",
        "character_coefficient": str(c), "pointwise_kernel_lower": str(1-2*abs(c)),
        "normalizer": str(z), "inverse_kernel_haar_mean": str(inverse_mean),
        "magnetic_one_face": str(mu), "one_face_fisher": str(fisher),
        "normalizer_laplacian_ratio": str(lap), "normalizer_score_squared": str(score),
        "new_link_fisher": str(inew), "old_link_fisher": str(iold),
        "total_fisher": str(inew+iold), "energy": str(kap*(inew+iold)/2+10*mu/kap),
        "conditional_normalization": "f convolution5(B)=1+c^5 Tr(B)/16",
        "physical_heat_feedback_verified": False})


def _face_trace_gradient(field: Sequence[Quaternion], cycle: Sequence[int]) -> tuple[Q, list[list[Q]]]:
    factors = [field[e-1] if e > 0 else quaternion_inverse(field[-e-1]) for e in cycle]
    derivatives = [[Q(0) for _ in range(3)] for _ in field]
    for j, token in enumerate(cycle):
        e = abs(token)-1
        for axis in range(3):
            tangent = tuple(Q(1, 2) if k == axis+1 else Q(0) for k in range(4))
            if token > 0:
                d = quaternion_product(tangent, field[e])
            else:
                v = quaternion_product(factors[j], tangent)
                d = (-v[0], -v[1], -v[2], -v[3])
            derivatives[e][axis] = 2*_product([*factors[:j], d, *factors[j+1:]])[0]
    return 2*_product(factors)[0], derivatives


def linear_character_cube_point(
    links: Sequence[Sequence[int | Q]], *, tilt: int | Q = Q(1, 3),
) -> dict[str, Any]:
    """Evaluate density and all12 original-link log-amplitude derivatives over Q."""
    if len(links) != 12:
        raise ValueError("exactly twelve links required")
    field = [_unit(q) for q in links]
    r = rational(tilt, "tilt")
    if not -1 < r < 1:
        raise ValueError("tilt must lie in(-1,1)")
    c, geometry = r/(1+r*r), cube_bridge_geometry()
    bottom, db = _face_trace_gradient(field, geometry["bottom"])
    z, coeff = 1+c**5*bottom/16, c**5/16
    score = [[-coeff*v/z for v in row] for row in db]
    density = 1/z
    for face in geometry["new_faces"]:
        chi, gradient = _face_trace_gradient(field, face)
        f = 1+c*chi
        density *= f
        for i in range(12):
            for axis in range(3):
                score[i][axis] += c*gradient[i][axis]/f
    old = sum((v*v/4 for i in geometry["old_edges"] for v in score[i-1]), Q(0))
    new = sum((v*v/4 for i in geometry["new_edges"] for v in score[i-1]), Q(0))
    return _seal("linear_character_cube_point_v1", {
        "links": [[str(v) for v in q] for q in field], "tilt": str(r)}, {
        "density": str(density), "bottom_trace": str(bottom),
        "old_log_amplitude_gradient_squared": str(old),
        "new_log_amplitude_gradient_squared": str(new),
        "log_amplitude_gradient": [[str(v/2) for v in row] for row in score],
        "source_scope": "one rational cube configuration, not an ensemble"})


def replay_cube_bridge_certificate(certificate: dict[str, Any]) -> bool:
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        p = certificate["payload"]
        raw, kind = p["inputs"], p["type"]
        if kind == "conditional_cube_feedback_budget_v1":
            row = conditional_cube_feedback_budget(**{k: Q(v) for k, v in raw.items()})
        elif kind == "conditional_two_face_seam_budget_v1":
            row = conditional_two_face_seam_budget(kappa_max=Q(raw["kappa_max"]))
        elif kind == "three_face_strict_feedback_obstruction_v1":
            row = three_face_strict_feedback_obstruction(constant=Q(raw["constant"]), theta=Q(raw["theta"]))
        elif kind == "linear_character_cube_control_v1":
            row = linear_character_cube_control(Q(raw["boundary_half_trace"]),
                tilt=Q(raw["tilt"]), kappa=Q(raw["kappa"]))
        elif kind == "linear_character_cube_point_v1":
            row = linear_character_cube_point([[Q(v) for v in q] for q in raw["links"]], tilt=Q(raw["tilt"]))
        else:
            return False
        return bool(certificate == row["certificate"])
    except (TypeError, ValueError, KeyError, ZeroDivisionError, OverflowError):
        return False


__all__ = [
    "conditional_cube_feedback_budget",
    "conditional_two_face_seam_budget",
    "cube_bridge_geometry",
    "cube_disk_geometry",
    "linear_character_cube_control",
    "linear_character_cube_point",
    "replay_cube_bridge_certificate",
    "three_face_strict_feedback_obstruction",
]
