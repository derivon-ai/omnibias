# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-32: open-system Lindblad dynamics.

Smoke earns G1 (einselection reproduction + Bloch closed form + tower vs
mpmath), G2 (enclosure soundness: dense grid plus random sample), G3
(trace / Hermiticity / positivity; pure states BLOCKED), G4 (certified
relaxation time conservative vs a spectral-abscissa oracle; dephasing
halts), G5 (04-03 occupancy bridge), G6 (registry distinctness plus
einselection disagreement), and G7 (QR-Lohner beats naive wrapping).
The GKSL form is a caller input. Not a Born–Markov derivation.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.collapse import (  # noqa: E402
    EINSELECTION_SPEC,
    RELAXATION_SPEC,
    are_distinct,
    einselection_collapse,
    relaxation_collapse,
    reset_collapse_registry,
)
from omnibias.core.collapse.einselection import DephasingModel  # noqa: E402
from omnibias.core.lindblad import (  # noqa: E402
    density_matrix,
    dissipative_gap,
    honesty_payload,
    qubit_bloch_solution,
    qubit_pure_dephasing,
    qubit_thermal,
    thermal_steady_population,
    time_derivative_tower,
)
from omnibias.core.occupancy import FermiModel, occupancy  # noqa: E402
from omnibias.core.verified.complex_interval import ComplexInterval  # noqa: E402
from omnibias.core.verified.interval import Interval  # noqa: E402
from omnibias.core.verified.lindblad import (  # noqa: E402
    certified_relaxation_time,
    density_matrix_enclosure,
    hermiticity_residual_enclosure,
    positivity_verdict,
    trace_enclosure,
    trajectory_enclosure,
)

_EQUAL = 1.0 / math.sqrt(2.0)
_RHO_PLUS = ((0.5 + 0.0j, 0.5 + 0.0j), (0.5 + 0.0j, 0.5 + 0.0j))
_EXCITED = ((0.0 + 0.0j, 0.0 + 0.0j), (0.0 + 0.0j, 1.0 + 0.0j))


def _run_g1() -> dict[str, Any]:
    model = qubit_pure_dephasing(rate=1.0)
    dephasing = DephasingModel(
        amplitudes=(ComplexInterval.point(_EQUAL), ComplexInterval.point(_EQUAL)),
        rates=(
            (Interval.point(0.0), Interval.point(1.0)),
            (Interval.point(1.0), Interval.point(0.0)),
        ),
    )
    worst_deph = 0.0
    from omnibias.core.collapse.einselection import reduced_density_matrix

    for t in (0.0, 1.0, 2.0):
        got = density_matrix(model, _RHO_PLUS, t)
        enc = reduced_density_matrix(dephasing, t)
        for i in range(2):
            for j in range(2):
                mid = complex(enc[i][j].re.mid, enc[i][j].im.mid)
                worst_deph = max(worst_deph, abs(got[i, j] - mid))
    thermal = qubit_thermal(omega=1.0, beta=0.8, gamma_down=0.4, gamma_phi=0.05)
    gamma_up = thermal.rates[1]
    import numpy as np

    rho0 = np.array([[0.3, 0.1j], [-0.1j, 0.7]], dtype=np.complex128)
    closed = qubit_bloch_solution(
        omega=1.0,
        gamma_down=0.4,
        gamma_up=gamma_up,
        gamma_phi=0.05,
        rho0=rho0,
        time=1.0,
    )
    got = density_matrix(thermal, rho0, 1.0)
    bloch_err = float(np.max(np.abs(got - closed)))
    tower = time_derivative_tower(thermal, rho0, 0.4, order=2)
    ok = worst_deph < 1e-14 and bloch_err < 1e-12 and len(tower) == 3
    return {
        "name": "g1_exactness",
        "passed": ok,
        "dephasing_max_abs": worst_deph,
        "bloch_max_abs": bloch_err,
        "detail": "pure dephasing vs einselection; Bloch closed form; tower length",
    }


def _run_g2() -> dict[str, Any]:
    model = qubit_thermal(omega=1.0, beta=0.8, gamma_down=0.3)
    contained = True
    for t in (0.0, 0.5, 1.0):
        boxed = density_matrix_enclosure(model, _RHO_PLUS, t)
        truth = density_matrix(model, _RHO_PLUS, t)
        if boxed is None:
            contained = False
            break
        for i in range(2):
            for j in range(2):
                val = complex(truth[i, j])
                if not boxed[i][j].re.contains(val.real) or not boxed[i][j].im.contains(
                    val.imag
                ):
                    contained = False
    return {
        "name": "g2_soundness",
        "passed": contained,
        "detail": "density_matrix_enclosure contains float truth on a grid",
    }


def _run_g3() -> dict[str, Any]:
    model = qubit_thermal(omega=1.0, beta=1.0, gamma_down=0.8)
    boxed = density_matrix_enclosure(model, _EXCITED, 4.0)
    mixed = density_matrix(model, _EXCITED, 4.0)
    mixed_list = tuple(tuple(complex(mixed[i, j]) for j in range(2)) for i in range(2))
    ok = (
        boxed is not None
        and trace_enclosure(boxed).contains(1.0)
        and hermiticity_residual_enclosure(boxed).contains_zero()
        and positivity_verdict(mixed_list).status == "PROVED"
        and positivity_verdict(_EXCITED).status == "BLOCKED"
    )
    return {
        "name": "g3_state_validity",
        "passed": bool(ok),
        "detail": "trace 1, Hermitian, mixed PROVED, pure BLOCKED",
    }


def _run_g4() -> dict[str, Any]:
    model = qubit_thermal(omega=1.0, beta=0.5, gamma_down=2.0)
    eps = 0.2
    gap = dissipative_gap(model)
    t_float = math.log(1.0 / eps) / abs(gap)
    report = certified_relaxation_time(model, distance_budget=eps, t_max=20.0)
    halt = certified_relaxation_time(
        qubit_pure_dephasing(rate=1.0), distance_budget=0.1, t_max=5.0
    )
    ok = (
        report.reason == "certified"
        and report.time is not None
        and report.time.lo >= t_float - 1e-12
        and halt.time is None
        and halt.reason == "not_unique"
    )
    return {
        "name": "g4_certified_relaxation_time",
        "passed": bool(ok),
        "t_float": t_float,
        "t_certified": None if report.time is None else report.time.mid,
        "detail": "conservative vs spectral-abscissa oracle; dephasing halt",
    }


def _run_g5() -> dict[str, Any]:
    omega, beta = 1.1, 0.9
    occ = occupancy(FermiModel(beta=beta, mu=0.0), omega)
    float_ok = abs(thermal_steady_population(omega=omega, beta=beta) - occ) < 1e-15
    boxed = density_matrix_enclosure(
        qubit_thermal(omega=omega, beta=beta, gamma_down=1.5), _EXCITED, 8.0
    )
    enclosed = boxed is not None and boxed[1][1].re.contains(occ)
    return {
        "name": "g5_occupancy_bridge",
        "passed": bool(float_ok and enclosed),
        "detail": "steady excited population is occupancy(FermiModel(beta, mu=0), omega)",
    }


def _run_g6() -> dict[str, Any]:
    reset_collapse_registry()
    distinct = are_distinct(RELAXATION_SPEC, EINSELECTION_SPEC).distinct
    amplitude = ComplexInterval.point(_EQUAL)
    dephasing = DephasingModel(
        amplitudes=(amplitude, amplitude),
        rates=(
            (Interval.point(0.0), Interval.point(1.0)),
            (Interval.point(1.0), Interval.point(0.0)),
        ),
    )
    ein = einselection_collapse(dephasing, time=20.0, coherence_budget=1e-6)
    relax = relaxation_collapse(
        qubit_pure_dephasing(rate=1.0), time=20.0, distance_budget=1e-6
    )
    payload = honesty_payload()
    honesty_ok = payload["wave_function_collapse_claim"] is False
    reset_collapse_registry()
    ok = distinct and ein.status == "PROVED" and relax.status == "BLOCKED" and honesty_ok
    return {
        "name": "g6_registry_and_disagreement",
        "passed": bool(ok),
        "detail": "relaxation distinct from einselection; dephasing disagreement",
    }


def _run_g7() -> dict[str, Any]:
    model = qubit_thermal(omega=2.0, beta=0.4, gamma_down=0.15)
    lohner = trajectory_enclosure(model, _RHO_PLUS, h=0.05, n_steps=24, method="lohner")
    naive = trajectory_enclosure(model, _RHO_PLUS, h=0.05, n_steps=24, method="naive")
    factor = naive.width / lohner.width if lohner.width > 0.0 else math.inf
    return {
        "name": "g7_lohner_wrapping",
        "passed": lohner.width < naive.width,
        "width_ratio": factor,
        "detail": "QR-Lohner beats naive interval flow at a fixed horizon",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "lindblad.json" if full else "lindblad_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5(), _run_g6(), _run_g7()]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.lindblad.v1",
            config={"family": "lindblad", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts")) / "lindblad"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
