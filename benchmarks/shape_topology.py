# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shape topology (theory 05-02 G6/G7).

G6: the soft Euler gap bound contains the true integer in 100% of
cases, and the API cannot return the value without the bound.
G7: Euler-regularized reconstruction recovers the correct planar
genus in >= 90% of the cases where a named single-disk implicit
baseline is wrong, without degrading mean IoU by more than 5%.

Composes 03-09 ``SoftCount`` / ``soft_euler_characteristic``. Does not
reimplement those gates. G4 stays unearned; G5 stays failed/retired.
``beta -> inf`` is temperature collapse, not founding bias collapse.
No differentiable function equals a Betti number.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.shape.topology import (  # noqa: E402
    SoftCount,
    cubical_faces_2d,
    digital_genus,
    euler_value_bound,
    field_euler_characteristic,
    field_euler_pair,
    field_to_occupancy,
    honesty_payload,
    occupancy_euler,
    regularize_occupancy,
    soft_euler_characteristic,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

BASELINE_NAME = "thresholded soft-disk implicit (2-D marching-cubes stand-in)"


def _hard_face_euler(masses: list[float], dims: list[int]) -> float:
    hard = 0.0
    for mass, dim in zip(masses, dims, strict=True):
        sign = -1.0 if int(dim) % 2 else 1.0
        hard += sign * (1.0 if float(mass) >= 0.5 else 0.0)
    return hard


def _iou(pred: np.ndarray, truth: np.ndarray, *, threshold: float = 0.5) -> float:
    a = np.asarray(pred) >= threshold
    b = np.asarray(truth) >= threshold
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(inter) / float(max(union, 1))


def _fit_soft_disk(
    xx: np.ndarray,
    yy: np.ndarray,
    pos: np.ndarray,
    exterior: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    best_err = float("inf")
    best = np.ones_like(xx)
    xs = xx[0]
    ys = yy[:, 0]
    for cx in np.linspace(float(xs[2]), float(xs[-3]), 7):
        for cy in np.linspace(float(ys[2]), float(ys[-3]), 7):
            rho2 = (xx - cx) ** 2 + (yy - cy) ** 2
            for radius in np.linspace(0.45, 0.88, 9):
                pred = 1.0 / (1.0 + np.exp(-float(beta) * (radius**2 - rho2)))
                err = float(np.mean((pred[pos] - 1.0) ** 2) + np.mean(pred[exterior] ** 2))
                if err < best_err:
                    best_err = err
                    best = pred
    return best


def _run_g6(*, n_faces: int, n_fields: int) -> dict[str, Any]:
    rng = np.random.default_rng(6)
    grid = np.linspace(-1.0, 1.0, 13)
    violations = 0
    n = 0
    for _ in range(int(n_faces)):
        masses = [float(v) for v in rng.uniform(0.0, 1.0, size=10)]
        dims = [int(d) for d in rng.integers(0, 3, size=10)]
        sc = soft_euler_characteristic(masses, dims)
        true = _hard_face_euler(masses, dims)
        n += 1
        if abs(sc.value - true) > sc.gap_bound + 1e-12:
            violations += 1
    for _ in range(int(n_fields)):
        field = rng.normal(0.0, 0.4, size=(13, 13))
        sc = field_euler_characteristic(field, beta=7.0, grid=grid)
        occ = field_to_occupancy(field, beta=7.0, grid=grid)
        masses, dims = cubical_faces_2d(occ)
        true = _hard_face_euler(masses, dims)
        n += 1
        if abs(sc.value - true) > sc.gap_bound + 1e-12:
            violations += 1
    for field in (
        lambda x, y: 0.36 - x**2 - y**2,
        lambda x, y: np.minimum(0.49 - x**2 - y**2, x**2 + y**2 - 0.09),
    ):
        sc = field_euler_characteristic(field, beta=14.0, grid=grid)
        pair = field_euler_pair(field, beta=14.0, grid=grid)
        n += 1
        if abs(sc.value - pair[0]) > 1e-15 or abs(sc.gap_bound - pair[1]) > 1e-15:
            violations += 1
        occ = field_to_occupancy(field, beta=14.0, grid=grid)
        masses, dims = cubical_faces_2d(occ)
        true = _hard_face_euler(masses, dims)
        if abs(sc.value - true) > sc.gap_bound + 1e-12:
            violations += 1
    occ_sc = occupancy_euler(np.ones((4, 4), dtype=np.float64))
    api_ok = (
        isinstance(occ_sc, SoftCount)
        and isinstance(euler_value_bound(occ_sc), tuple)
        and len(euler_value_bound(occ_sc)) == 2
    )
    try:
        float(occ_sc)  # type: ignore[arg-type]
        api_ok = False
    except TypeError:
        pass
    passed = violations == 0 and api_ok
    return {
        "name": "g6_topology_bound",
        "passed": passed,
        "n": n,
        "violations": violations,
        "api_returns_pair": api_ok,
        "detail": "100% gap containment; SoftCount / (value, bound) only",
    }


def _one_shape(seed: int, *, kind: str, n: int = 25) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    xs = np.linspace(-1.0, 1.0, int(n))
    xx, yy = np.meshgrid(xs, xs, indexing="xy")
    cx = float(rng.uniform(-0.08, 0.08))
    cy = float(rng.uniform(-0.08, 0.08))
    r_out = float(rng.uniform(0.64, 0.76))
    r_in = float(rng.uniform(0.26, 0.36))
    rho2 = (xx - cx) ** 2 + (yy - cy) ** 2
    beta = 18.0
    disk = 1.0 / (1.0 + np.exp(-beta * (r_out**2 - rho2)))
    if kind == "annulus":
        hole = 1.0 / (1.0 + np.exp(-beta * (r_in**2 - rho2)))
        truth = disk * (1.0 - hole)
        target_chi = 0.0
    elif kind == "disk":
        truth = disk
        target_chi = 1.0
    else:
        raise ValueError("kind must be 'annulus' or 'disk'")
    pos = truth >= 0.5
    exterior = rho2 >= (r_out + 0.08) ** 2
    baseline = _fit_soft_disk(xx, yy, pos, exterior, beta=beta)
    regularized = regularize_occupancy(
        baseline,
        pos_mask=pos,
        exterior_mask=exterior,
        target_chi=target_chi,
    )
    true_g = digital_genus(truth)
    return {
        "seed": int(seed),
        "kind": kind,
        "true_genus": true_g,
        "baseline_genus": digital_genus(baseline),
        "regularized_genus": digital_genus(regularized),
        "iou_baseline": _iou(baseline, truth),
        "iou_regularized": _iou(regularized, truth),
    }


def _run_g7(*, n_annulus: int, n_disk: int) -> dict[str, Any]:
    rows = [_one_shape(s, kind="annulus") for s in range(int(n_annulus))]
    rows.extend(_one_shape(100 + s, kind="disk") for s in range(int(n_disk)))
    wrong = [r for r in rows if r["baseline_genus"] != r["true_genus"]]
    recovered = [r for r in wrong if r["regularized_genus"] == r["true_genus"]]
    rate = float(len(recovered) / len(wrong)) if wrong else 0.0
    mean_base = float(np.mean([r["iou_baseline"] for r in wrong])) if wrong else 0.0
    mean_reg = float(np.mean([r["iou_regularized"] for r in wrong])) if wrong else 0.0
    iou_ratio = mean_reg / max(mean_base, 1e-12)
    passed = bool(wrong) and rate >= 0.90 and mean_reg >= 0.95 * mean_base
    return {
        "name": "g7_shape_quality",
        "passed": passed,
        "n": len(rows),
        "unreg_wrong": len(wrong),
        "recovered": len(recovered),
        "recovery_rate": rate,
        "mean_iou_baseline": mean_base,
        "mean_iou_regularized": mean_reg,
        "iou_ratio": iou_ratio,
        "baseline": BASELINE_NAME,
        "detail": "genus recovery on cases the single-disk implicit gets wrong",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "shape_topology.json" if full else "shape_topology_smoke.json"
    t0 = time.perf_counter()
    g6 = _run_g6(n_faces=256 if full else 64, n_fields=128 if full else 32)
    g7 = _run_g7(n_annulus=32 if full else 12, n_disk=8 if full else 3)
    entries = [g6, g7]
    for e in entries:
        print(e["name"], "ok" if e["passed"] else "FAIL")
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.shape_topology.v1",
            config={
                "family": "shape_topology",
                "full": full,
                "honesty": {
                    **honesty_payload(),
                    "g4_unearned": True,
                    "g5_failed": True,
                    "equals_betti": False,
                    "named_baseline": BASELINE_NAME,
                },
            },
        ),
        "baseline": {"name": BASELINE_NAME},
        "seeds": list(range(int(g7["n"]))),
        "g6_earned": bool(g6["passed"]),
        "g7_earned": bool(g7["passed"]),
        "g4_unearned": True,
        "g5_failed": True,
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "beyond_pde"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
