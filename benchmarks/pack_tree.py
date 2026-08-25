# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: hierarchical pack tree (theory 02-07). 1-D offsets; eta=0 dense."""

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

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

# G3: near-linear in N+M over two decades of M, with a dense crossover.
G3_MS = (32, 320, 3200)
G3_N_EVAL = 16
G3_P = 6
G3_ETA = 2.0
G3_WARMUP = 1
G3_REPEATS = 3
G3_EXPONENT_MAX = 1.3
G3_CROSSOVER_RATIO_MAX = 1.0


def _median_seconds(fn: Any, *, warmup: int, repeats: int) -> float:
    for _ in range(int(warmup)):
        fn()
    samples: list[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _run_g3() -> dict[str, Any]:
    """Time dense vs hierarchical over two decades of bank size ``M``.

    ``far_eval`` still walks every source (per-source Taylor). That is
    ``O(M)`` per evaluation, so a near-linear ``N+M`` win against dense
    cannot appear. Record the miss; do not stub ``passed: true``.
    """
    from omnibias.core.hierarchy import build_pack_tree, dense_scan, hierarchical_value

    zs = tuple(8.0 + 0.15 * i for i in range(G3_N_EVAL))
    rows: list[dict[str, Any]] = []
    for m in G3_MS:
        offsets = tuple(float(i) * 2.0 / float(m) - 1.0 for i in range(int(m)))
        weights = tuple(1.0 / float(m) for _ in offsets)
        orders = tuple(1 for _ in offsets)
        tree = build_pack_tree(offsets, leaf_size=8)

        def _dense(
            offs: tuple[float, ...] = offsets,
            wts: tuple[float, ...] = weights,
            ords: tuple[int, ...] = orders,
        ) -> None:
            for z in zs:
                dense_scan(z, offs, wts, ords)

        def _hier(
            tr=tree,
            offs: tuple[float, ...] = offsets,
            wts: tuple[float, ...] = weights,
            ords: tuple[int, ...] = orders,
        ) -> None:
            for z in zs:
                hierarchical_value(z, tr, offs, wts, ords, p=G3_P, eta=G3_ETA)

        dense_s = _median_seconds(_dense, warmup=G3_WARMUP, repeats=G3_REPEATS)
        hier_s = _median_seconds(_hier, warmup=G3_WARMUP, repeats=G3_REPEATS)
        rows.append(
            {
                "m": int(m),
                "dense_seconds": dense_s,
                "hier_seconds": hier_s,
                "hier_over_dense": hier_s / max(dense_s, 1e-12),
            }
        )
    log_m = [float(np.log(row["m"])) for row in rows]
    log_h = [float(np.log(max(row["hier_seconds"], 1e-18))) for row in rows]
    slope, _intercept = np.polyfit(np.asarray(log_m), np.asarray(log_h), 1)
    ratio_hi = float(rows[-1]["hier_over_dense"])
    crossover_m = next(
        (int(row["m"]) for row in rows if row["hier_over_dense"] <= G3_CROSSOVER_RATIO_MAX),
        None,
    )
    earned = bool(
        float(slope) <= G3_EXPONENT_MAX
        and crossover_m is not None
        and ratio_hi <= G3_CROSSOVER_RATIO_MAX
    )
    return {
        "name": "g3_complexity",
        "passed": earned,
        "earned": earned,
        "in_ci_all_passed": False,
        "m": list(G3_MS),
        "n_eval": G3_N_EVAL,
        "p": G3_P,
        "eta": G3_ETA,
        "rows": rows,
        "hier_time_exponent_vs_m": float(slope),
        "exponent_max": G3_EXPONENT_MAX,
        "hier_over_dense_at_m_hi": ratio_hi,
        "crossover_m": crossover_m,
        "note": (
            "far_eval is a per-source Taylor (O(M) per z), not an O(p) "
            "multipole. Measured hierarchical wall does not beat dense "
            "over two decades of M; no crossover. 1-D offsets only."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.core.hierarchy import build_pack_tree, dense_scan, hierarchical_value

    offsets = tuple(float(i) * 0.1 - 0.8 for i in range(16))
    weights = tuple(0.05 for _ in offsets)
    orders = tuple(1 for _ in offsets)
    tree = build_pack_tree(offsets, leaf_size=4)
    z = 0.2
    g1 = dense_scan(z, offsets, weights, orders) == hierarchical_value(
        z, tree, offsets, weights, orders, eta=0.0
    )
    g3 = _run_g3()
    entries: list[dict[str, Any]] = [
        {"name": "g1_eta0_bit_identical", "passed": g1, "in_ci_all_passed": True},
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.pack_tree.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "g3_in_all_passed": False,
            "gates_in_scope": ["g1"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["g3"] = g3
    payload["honesty"] = {
        "axis": "1-D offsets",
        "far_field": "truncation with a bound",
        "g3_earned": bool(g3["earned"]),
        "g3_in_ci_all_passed": False,
        "far_eval_is_per_source_taylor": True,
    }
    if args.full:
        dest = SCRATCH / "hierarchy" / "pack_tree.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('pack_tree_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
