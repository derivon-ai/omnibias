# SPDX-License-Identifier: Apache-2.0
"""Exact SU(2) noncommuting trace jets and finite-block weak-field bounds.

The magnetic trace is the actual Wilson trace, not a scalar surrogate.
The electric chart is unreduced: a tree gauge changes its physical metric.
No renormalization or continuum implication is inferred from these bounds.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from math import factorial
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational
from omnibias.geometry.gauge.transfer.wilson_large_field import su2_wilson_large_field

Quaternion = tuple[Q, Q, Q, Q]
Vectors = Sequence[Sequence[int | Q]]
_ZERO: Quaternion = (Q(0), Q(0), Q(0), Q(0))
_ONE: Quaternion = (Q(1), Q(0), Q(0), Q(0))


def _multiply(p: Quaternion, q: Quaternion) -> Quaternion:
    """Product for q0*I+i*q_vector.sigma: vector cross term has minus sign."""
    a, x, y, z = p
    b, u, v, w = q
    return (a*b-x*u-y*v-z*w, a*u+b*x-y*w+z*v,
            a*v+b*y-z*u+x*w, a*w+b*z-x*v+y*u)


def _convolve(left: list[Quaternion], right: list[Quaternion]) -> list[Quaternion]:
    result: list[Quaternion] = []
    for n in range(len(left)):
        terms = [_multiply(left[k], right[n-k]) for k in range(n+1)]
        result.append((sum((p[0] for p in terms), Q(0)), sum((p[1] for p in terms), Q(0)),
                       sum((p[2] for p in terms), Q(0)), sum((p[3] for p in terms), Q(0))))
    return result


def _report(kind: str, claim: str, witness: dict[str, Any], earned: dict[str, bool]) -> dict[str, Any]:
    scope = {
        "physical_gauge_fixed_electric_metric_verified": False,
        "all_state_localization_operator_bound_verified": earned.get("all_state_localization_operator_bound_verified", False),
        "conditional_large_field_bound_verified": False,
        "exponential_polymer_bound_verified": False,
        "spectral_gap_claim": False, "infinite_volume_claim": False,
        "uniform_in_a_claim": False, "continuum_claim": False,
        "yang_mills_claim": False, "yang_mills_mass_gap_claim": False,
    }
    cert = make_certificate(
        claim=claim, payload={"type": kind, "witness": witness},
        honesty={**earned, **scope},
        meta={"analytic_implication": "docs/api/gauge-nonabelian-small-field.md",
              "transcend_backend": "not_used"},
    )
    return {"status": "PASS", "witness": witness, "certificate": cert,
            "digest_verified": verify_certificate_digest(cert), **earned, **scope,
            "theorem_prover_verified": False, "mathlib_verified": False}


def su2_plaquette_trace_jet(
    vectors: Vectors, *, degree: int = 3, parameter_radius: int | Q = 1,
    norm_bounds: Sequence[int | Q] | None = None,
) -> dict[str, Any]:
    """Exact Taylor coefficients of Tr(prod_i exp(i*t*A_i.sigma/2)).

    Four ordered, oriented vectors are required. Coefficients are Taylor
    coefficients (derivatives divided by n!), obtained by quaternion jet
    composition. Rational Euclidean norm upper bounds are checked by squares;
    absent bounds use the safe L1 norm. If R=sum(bounds)/2, the degree-N
    remainder for every real |t|<=T is at most 2*(R*T)^(N+1)/(N+1)!.
    """
    a = [[_rational(x, "vector coordinate") for x in row] for row in vectors]
    if len(a) != 4 or any(len(row) != 3 for row in a):
        raise ValueError("vectors must have shape (4,3)")
    n = _integer(degree, "degree")
    radius = _rational(parameter_radius, "parameter_radius")
    if n < 0 or radius < 0:
        raise ValueError("degree and parameter_radius must be nonnegative")
    bounds = ([sum((abs(x) for x in row), Q(0)) for row in a] if norm_bounds is None
              else [_rational(x, "norm bound") for x in norm_bounds])
    if len(bounds) != 4 or any(b < 0 or b*b < sum((x*x for x in row), Q(0))
                               for b, row in zip(bounds, a, strict=False)):
        raise ValueError("four nonnegative bounds must enclose Euclidean vector norms")
    product = [_ONE] + [_ZERO] * n
    for row in a:
        generator: Quaternion = (Q(0), row[0]/2, row[1]/2, row[2]/2)
        jet = [_ONE]
        for order in range(1, n+1):
            term = _multiply(jet[-1], generator)
            jet.append((term[0]/order, term[1]/order, term[2]/order, term[3]/order))
        product = _convolve(product, jet)
    coefficients = [2*p[0] for p in product]
    r = sum(bounds, Q(0))/2
    remainder = 2*(r*radius)**(n+1)/factorial(n+1)
    endpoint = sum((c*radius**k for k, c in enumerate(coefficients)), Q(0))
    witness = {
        "inputs": {"vectors": [[str(x) for x in row] for row in a], "degree": n,
                   "parameter_radius": str(radius), "norm_bounds": [str(x) for x in bounds]},
        "object": "chi(t)=Tr(exp(i*t*A1.sigma/2)...exp(i*t*A4.sigma/2))",
        "convention": "ordered oriented vectors; quaternion scalar+ i*vector.sigma; minus cross product",
        "calculus_path": "CLOSED_FORM exact rational quaternion Taylor jets; no nested autodiff",
        "proof": "multinomial Leibniz plus unitarity gives |chi^(k)(t)|<=2*R^k for real t; integral Taylor remainder",
        "arithmetic": {"coefficients": [str(x) for x in coefficients],
                       "generator_norm_sum_upper": str(r), "uniform_remainder_upper": str(remainder),
                       "endpoint_polynomial": str(endpoint),
                       "endpoint_trace_lower": str(max(Q(-2), endpoint-remainder)),
                       "endpoint_trace_upper": str(min(Q(2), endpoint+remainder))},
    }
    return _report("su2_plaquette_trace_jet_v1", "exact finite noncommuting SU2 Wilson trace jet with a real-parameter Taylor enclosure",
                   witness, {"exact_trace_jet_verified": True, "uniform_trace_remainder_verified": True})


def su2_wilson_small_field_budget(kappa: int | Q, edge_radius: int | Q) -> dict[str, Any]:
    """Bound the actual magnetic remainder and unreduced electric chart.

    Assumes every one of four oriented log vectors has norm<=edge_radius<=2.
    Electric identity acts on compactly supported scalar functions in an
    original link-log chart, before gauge fixing or physical localization.
    """
    coupling, eps = _rational(kappa, "kappa"), _rational(edge_radius, "edge_radius")
    if coupling <= 0 or not 0 <= eps <= 2:
        raise ValueError("kappa>0 and 0<=edge_radius<=2 are required")
    h = 1-eps*eps/24
    witness = {
        "inputs": {"kappa": str(coupling), "edge_radius": str(eps)},
        "hamiltonian": "aH=kappa/2*sum_e C_e+2/kappa*sum_p(2-Tr U_p)",
        "trace_expansion": "2-|sum A_i|^2/4+sum_(i<j<k) A_i.dot(A_j.cross(A_k))/4+R4",
        "electric_chart": "unreduced scalar local Dirichlet chart in original links; not a physical tree-gauge operator",
        "halfdensity_identity": "h*C*h^-1=-div(Ginv grad)-1/4; h=sinc(r/2), Ginv=Pr+(r/(2*sin(r/2)))^2*Pt",
        "commutator_witness": "A,B,-A,-B with A perpendicular B and |A|=|B|=epsilon: quadratic and cubic vanish; action=4*sin(epsilon/2)^4",
        "arithmetic": {
            "halfdensity_lower_per_edge": str(h),
            "electric_derivative_relative_error_upper": str(h**-2-1),
            "electric_constant_per_edge": str(-coupling/8),
            "character_cubic_absolute_upper": str(eps**3),
            "character_remainder_after_cubic_upper": str(4*eps**4/3),
            "magnetic_cubic_absolute_upper_per_plaquette": str(2*eps**3/coupling),
            "magnetic_remainder_after_cubic_upper_per_plaquette": str(8*eps**4/(3*coupling)),
            "orthogonal_commutator_action_lower": str(eps**4*h**4/4),
        },
    }
    return _report("su2_wilson_small_field_budget_v1", "SU2 local Wilson magnetic remainder and unreduced half-density electric bounds",
                   witness, {"actual_local_magnetic_remainder_verified": True,
                             "unreduced_electric_chart_bound_verified": True,
                             "commutator_quartic_obstruction_verified": eps > 0})


def su2_wilson_block_small_field(
    kappa: int | Q, edge_radius: int | Q, *, block_side: int = 1,
) -> dict[str, Any]:
    """Bound failure of an axial-tree small-field patch in the actual vacuum.

    A fixed open block has vertices {0,...,b}^3, embedded without wrapping
    in any periodic cubic lattice of side L>=max(3,b+1). Tree paths run
    x then y then z. Each chord is a ribbon of at most 2*b plaquettes.
    This controls a gauge-invariant patch event, actual-vacuum localization
    and a universal single-block IMS error, not the transformed conditional
    kinetic operator or many-block polymer weights.
    """
    coupling, eps = _rational(kappa, "kappa"), _rational(edge_radius, "edge_radius")
    b = _integer(block_side, "block_side")
    if coupling <= 0 or not 0 < eps <= 2 or b < 1:
        raise ValueError("kappa>0, 0<edge_radius<=2 and integer block_side>=1 are required")
    count = 3*b*b*(b+1)
    threshold = (7*eps/(44*b))**2
    source = su2_wilson_large_field(coupling, threshold=threshold, marked_count=count)
    local = su2_wilson_small_field_budget(coupling, eps)
    bound = Q(source["large_field_probability_upper"])
    mean = Q(source["plaquette_action_mean_upper"])
    source_form = Q(source["witness"]["arithmetic"]["marked_average_action_form_energy_upper"])
    good_norm = max(Q(0), 1-2*count*mean/threshold)
    slope = Q(22, 7)/threshold
    localization_cost = slope**2*count**2*source_form
    weak_cost_coefficient = Q(1452, 49)*count*min(4, count)*(Q(44*b, 7))**4
    universal_ims = Q(968, 49)*coupling*min(4, count)/threshold
    witness = {
        "inputs": {"kappa": str(coupling), "edge_radius": str(eps), "block_side": b},
        "family": f"every periodic isotropic cubic SU2 vacuum with integer L>=max(3,{b+1}), every embedded unwrapped block with vertices 0..{b} in each axis",
        "event": "failure to have all canonical axial-tree-gauged block link log norms<=edge_radius",
        "tree": "all z links; y links at z=0; x links at y=z=0; root paths x then y then z",
        "geometric_implication": "v_p<=s for every block plaquette implies every tree-gauge link log norm<=2*b*(22/7)*sqrt(s)=edge_radius",
        "distance_bound": "SU2 log norm<=pi*sqrt(2-Tr U)<=22/7*sqrt(2-Tr U); bi-invariant triangle inequality",
        "arithmetic": {"block_plaquette_count": count, "max_ribbon_plaquettes": 2*b,
                       "plaquette_threshold": str(threshold), "bad_block_probability_upper": str(bound),
                       "uncapped_weak_probability_upper": str(Q(8712, 49)*b**4*(b+1)*coupling/eps**2),
                       "probability_bound_nontrivial": bound < 1,
                       "good_localized_vacuum_norm_squared_lower": str(good_norm),
                       "partition_angle_lipschitz_upper": str(slope),
                       "total_vacuum_localization_form_cost_upper": str(localization_cost),
                       "weak_localization_cost_coefficient": str(weak_cost_coefficient),
                       "weak_localization_cost_upper": str(weak_cost_coefficient*coupling**2/eps**4),
                       "universal_single_block_ims_error_upper": str(universal_ims),
                       "bad_support_local_magnetic_potential_lower": str(threshold/coupling),
                       "good_normalized_energy_above_vacuum_upper": str(localization_cost/good_norm) if good_norm > 0 else None},
        "physical_localization": {
            "sum_action": "S=sum_block(2-Tr U_p)",
            "partition": "chi_good=cos(theta(S)), chi_bad=sin(theta(S)); theta=0 below s/2, pi/2 above s, linear in between",
            "support": "chi_good nonzero implies all block plaquette actions<s, hence axial-tree small-link patch",
            "state": "chi_good*psi0 is gauge invariant; no knowledge of the vacuum density is assumed",
            "norm_bound": "norm^2>=1-Pr(S>s/2)>=1-2*count*mean/s",
            "form_bound": "E0(chi_good)+E0(chi_bad)<=slope^2*E0(S); this is an expectation in the actual vacuum, not an all-state operator bound",
            "universal_ims": "Gamma(S)<=4*min(4,count)*S and theta' vanishes outside s/2<S<s; hence kappa/2*sum Gamma(chi)<=968*kappa*min(4,count)/(49*s) pointwise for every state",
            "bad_support_energy": "chi_bad nonzero implies S>s/2, so local magnetic potential 2*S/kappa>=s/kappa; no vacuum energy subtraction",
            "spectral_scope": "normalized energy upper concerns a vacuum approximation, not an orthogonal excitation or a gap",
        },
        "actual_vacuum_moment_certificate": source["certificate"],
        "local_chart_certificate": local["certificate"],
        "missing_step": "tree gauge changes conditional electric metric; single-block IMS and uncentered potential bounds do not supply exterior-compatible vacuum-subtracted comparison or RG composition",
    }
    return _report("su2_wilson_block_small_field_v1", "actual SU2 block patch probability, gauge-invariant vacuum localization and universal single-block IMS bound",
                   witness, {"actual_block_patch_probability_bound_verified": True,
                             "actual_gauge_invariant_vacuum_localization_verified": True,
                             "all_state_localization_operator_bound_verified": True,
                             "volume_uniform_fixed_block_probability_bound_verified": True})


def replay_su2_nonabelian_small_field_certificate(certificate: dict[str, Any]) -> bool:
    """Canonical replay of trace, chart or block certificates and nested sources."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        inputs, kind = payload["witness"]["inputs"], payload["type"]
        if kind == "su2_plaquette_trace_jet_v1":
            result = su2_plaquette_trace_jet(
                [[Q(x) for x in row] for row in inputs["vectors"]], degree=inputs["degree"],
                parameter_radius=Q(inputs["parameter_radius"]), norm_bounds=[Q(x) for x in inputs["norm_bounds"]])
        elif kind == "su2_wilson_small_field_budget_v1":
            result = su2_wilson_small_field_budget(Q(inputs["kappa"]), Q(inputs["edge_radius"]))
        elif kind == "su2_wilson_block_small_field_v1":
            result = su2_wilson_block_small_field(Q(inputs["kappa"]), Q(inputs["edge_radius"]), block_side=inputs["block_side"])
        else:
            return False
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_nonabelian_small_field_certificate",
    "su2_plaquette_trace_jet",
    "su2_wilson_block_small_field",
    "su2_wilson_small_field_budget",
]
