# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 04-03: Fermi occupancy and thermodynamic potentials.

Smoke earns G1 (entropy identity + high-precision derivative match + exact
``d omega/d mu = -f``), G2 (enclosure soundness: dense grid plus random
sample), G3 (certified chemical potential ball contains a fine-bisection
root; an empty ball is a reported halt, not a failure), G4 (certified
Sommerfeld coefficients match the closed-form Ashcroft & Mermin values and
beat a finite-difference numerical-integration arm on tightness), and G5
(honesty non-vacuity: permanently-false keys never flip, and the collapse
registry refuses a ``beta``/``indicator`` rebrand). ``beta -> inf`` is the
named founding **temperature collapse**; it is never taken by this module
and is not requested as a new collapse-registry slot.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.collapse import (  # noqa: E402
    CollapseSpec,
    register_collapse,
    reset_collapse_registry,
)
from omnibias.core.occupancy import (  # noqa: E402
    FermiModel,
    entropy_per_state,
    honesty_payload,
    occupancy,
    occupancy_derivatives,
    occupancy_window,
    sommerfeld_coefficient,
)
from omnibias.core.verified.occupancy import (  # noqa: E402
    certified_chemical_potential,
    constant_density_of_states,
    sommerfeld_coefficient_enclosure,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
_CASES = (
    (2.0, 0.5, 0.3),
    (0.7, -1.0, 2.5),
    (5.0, 0.0, 0.0),
    (10.0, 2.0, 1.9),
    (0.3, -3.0, -2.5),
)


# ----- G1: exactness ----------------------------------------------------------


def _run_g1() -> dict[str, Any]:
    mpmath = __import__("mpmath")
    mpmath.mp.dps = 50

    def mp_sigmoid(z: object) -> object:
        return 1 / (1 + mpmath.exp(-z))

    def mp_softplus(z: object) -> object:
        return mpmath.log1p(mpmath.exp(z))

    worst_entropy_identity = 0.0
    worst_occ_deriv = 0.0
    worst_domega = 0.0
    for beta, mu, energy in _CASES:
        model = FermiModel(beta=beta, mu=mu)
        f = occupancy(model, energy)
        s = entropy_per_state(model, energy)
        if 0.0 < f < 1.0:
            ref_s = -f * math.log(f) - (1.0 - f) * math.log(1.0 - f)
            worst_entropy_identity = max(worst_entropy_identity, abs(s - ref_s))

        def f_of_e(e: object, beta: float = beta, mu: float = mu) -> object:
            z = -beta * (e - mu)
            return mp_sigmoid(z)

        tower = occupancy_derivatives(model, energy, order=6)
        for n in range(7):
            ref = float(mpmath.diff(f_of_e, mpmath.mpf(str(energy)), n))
            rel = abs(tower[n] - ref) / max(1.0, abs(ref))
            worst_occ_deriv = max(worst_occ_deriv, rel)

        def omega_of_mu(m: object, beta: float = beta, energy: float = energy) -> object:
            z = -beta * (energy - m)
            return -mp_softplus(z) / beta

        d_omega_d_mu = float(mpmath.diff(omega_of_mu, mpmath.mpf(str(mu)), 1))
        worst_domega = max(worst_domega, abs(d_omega_d_mu - (-f)))

    ok = (
        worst_entropy_identity <= 1e-12
        and worst_occ_deriv <= 1e-6
        and worst_domega <= 1e-9
    )
    return {
        "name": "g1_exactness",
        "passed": ok,
        "worst_entropy_identity_abs": worst_entropy_identity,
        "worst_occupancy_derivative_rel": worst_occ_deriv,
        "worst_domega_dmu_abs": worst_domega,
        "detail": "entropy identity, d^n f/de^n vs mpmath dps=50, d omega/d mu = -f",
    }


# ----- G2: enclosure soundness ------------------------------------------------


def _run_g2() -> dict[str, Any]:
    from omnibias.core.verified.occupancy import occupancy_enclosure

    rng = np.random.default_rng(0)
    n_checked = 0
    for beta, mu, _ in _CASES:
        model = FermiModel(beta=beta, mu=mu)
        grid = np.linspace(-6.0, 6.0, 49)
        sample = list(grid) + list(rng.uniform(-6.0, 6.0, size=30))
        for e in sample:
            enc_tower = occupancy_enclosure(beta, mu, float(e), order=3)
            float_tower = occupancy_derivatives(model, float(e), order=3)
            for enc, val in zip(enc_tower, float_tower, strict=True):
                if not (enc.lo <= val <= enc.hi):
                    return {
                        "name": "g2_soundness",
                        "passed": False,
                        "detail": f"enclosure miss at beta={beta}, mu={mu}, e={e}",
                    }
                n_checked += 1
    return {
        "name": "g2_soundness",
        "passed": True,
        "n_checked": n_checked,
        "detail": "occupancy_enclosure contains dense grid + random sample, orders 0..3",
    }


# ----- G3: certified chemical potential --------------------------------------


def _bisection_root(beta: float, g0: float, e_lo: float, e_hi: float, n_target: float) -> float:
    def n_of_mu(mu: float) -> float:
        model = FermiModel(beta=beta, mu=mu)
        return g0 * occupancy_window(model, e_lo, e_hi)

    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if n_of_mu(mid) < n_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _run_g3() -> dict[str, Any]:
    cases = (
        (2.0, 0.5, -3.0, 3.0, 0.1),
        (0.8, -1.2, -6.0, 6.0, 0.2),
        (5.0, 0.7, -1.0, 1.0, 0.1),
        (1.5, 0.0, -4.0, 4.0, 0.15),
        (3.0, 1.3, -2.5, 2.5, 0.1),
    )
    g0 = 1.0
    dos = constant_density_of_states(g0)
    n_accepted = 0
    worst_containment_slack = math.inf
    for beta, mu_true, e_lo, e_hi, r_max in cases:
        model = FermiModel(beta=beta, mu=mu_true)
        n_target = g0 * occupancy_window(model, e_lo, e_hi)
        mu_bar = mu_true + 0.02
        decision = certified_chemical_potential(
            beta, dos, e_lo, e_hi, n_target, mu_bar=mu_bar, r_max=r_max
        )
        if not decision.accepted:
            continue
        n_accepted += 1
        assert decision.certificate is not None
        root = _bisection_root(beta, g0, e_lo, e_hi, n_target)
        radius = decision.certificate.radius
        contains_root = (mu_bar - radius) <= root <= (mu_bar + radius)
        contains_truth = (mu_bar - radius) <= mu_true <= (mu_bar + radius)
        if not (contains_root and contains_truth):
            return {
                "name": "g3_certified_chemical_potential",
                "passed": False,
                "detail": f"ball did not contain root/truth at beta={beta}",
            }
        slack = min(root - (mu_bar - radius), (mu_bar + radius) - root)
        worst_containment_slack = min(worst_containment_slack, slack)

    # An empty ball for a too-tight r_max is a reported halt, never a failure.
    halt = certified_chemical_potential(
        2.0, dos, -3.0, 3.0, 2.0 * occupancy_window(FermiModel(beta=2.0, mu=0.5), -3.0, 3.0),
        mu_bar=0.51, r_max=0.01,
    )
    halt_ok = (not halt.accepted) and halt.reason == "empty" and halt.certificate is None

    ok = n_accepted >= 4 and halt_ok
    return {
        "name": "g3_certified_chemical_potential",
        "passed": ok,
        "n_accepted": n_accepted,
        "n_cases": len(cases),
        "worst_containment_slack": (
            worst_containment_slack if math.isfinite(worst_containment_slack) else None
        ),
        "halt_reason": halt.reason,
        "detail": "nonempty balls contain bisection root and truth; too-tight r_max halts honestly",
    }


# ----- G4: certified Sommerfeld coefficients ---------------------------------


def _run_g4() -> dict[str, Any]:
    a1_true = math.pi**2 / 6.0
    a2_true = 7.0 * math.pi**4 / 360.0
    enc1 = sommerfeld_coefficient_enclosure(1)
    enc2 = sommerfeld_coefficient_enclosure(2)
    contains = (enc1.lo <= a1_true <= enc1.hi) and (enc2.lo <= a2_true <= enc2.hi)
    tight = enc1.width < 1e-9 and enc2.width < 1e-9

    matches_float = (
        abs(sommerfeld_coefficient(1) - a1_true) < 1e-12
        and abs(sommerfeld_coefficient(2) - a2_true) < 1e-12
    )

    # A finite-difference / truncated-domain numerical-integration arm of
    # the same moment, as the non-certified baseline this beats on tightness.
    def sigma_prime(z: float) -> float:
        s = 1.0 / (1.0 + math.exp(-z))
        return s * (1.0 - s)

    xs = np.linspace(-40.0, 40.0, 20001)
    ys = np.array([sigma_prime(float(x)) * x**4 for x in xs])
    fd_moment = float(np.trapezoid(ys, xs))
    fd_a2 = fd_moment / math.factorial(4)
    fd_error = abs(fd_a2 - a2_true)
    beats_fd = fd_error > enc2.width

    ok = contains and tight and matches_float and beats_fd
    return {
        "name": "g4_sommerfeld",
        "passed": ok,
        "a1_enclosure_width": enc1.width,
        "a2_enclosure_width": enc2.width,
        "fd_a2_abs_error": fd_error,
        "detail": "closed-form zeta_even enclosure vs Ashcroft & Mermin + FD baseline",
    }


# ----- G5: honesty non-vacuity ------------------------------------------------


def _run_g5() -> dict[str, Any]:
    payload = honesty_payload()
    all_false = all(v is False for v in payload.values())

    forbidden_keys = (
        "dft_solved_claim",
        "many_body_solved_claim",
        "interacting_system_claim",
        "thermodynamic_limit_taken",
        "phase_transition_proved",
        "founding_bias_collapse",
        "temperature_collapse",
        "requests_new_collapse_registry_slot",
        "theorem_prover_verified",
    )
    pattern = re.compile(r'"(' + "|".join(forbidden_keys) + r')"\s*:\s*True')
    sources = [
        REPO_ROOT / "packages/omnibias-core/src/omnibias/core/occupancy.py",
        REPO_ROOT / "packages/omnibias-core/src/omnibias/core/verified/occupancy.py",
    ]
    static_scan_clean = True
    for path in sources:
        if pattern.search(path.read_text(encoding="utf-8")):
            static_scan_clean = False

    reset_collapse_registry()
    refused = False
    try:
        register_collapse(
            CollapseSpec(
                name="occupancy_zero_temperature",
                parameter="beta",
                limit="inf",
                surviving_object="indicator",
                failure="not a certificate",
                home="omnibias.core.occupancy",
                register="differentiable",
            )
        )
    except ValueError as exc:
        refused = "rebrand of 'temperature'" in str(exc)
    finally:
        reset_collapse_registry()

    ok = all_false and static_scan_clean and refused
    return {
        "name": "g5_honesty_non_vacuity",
        "passed": ok,
        "all_keys_false": all_false,
        "static_scan_clean": static_scan_clean,
        "registry_refused_rebrand": refused,
        "detail": "permanently-false keys stay false; registry refuses a beta/indicator rebrand",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "occupancy.json" if full else "occupancy_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for e in entries:
        print(e["name"], "ok" if e["passed"] else "FAIL")
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.occupancy.v1",
            config={"family": "occupancy", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "occupancy"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
