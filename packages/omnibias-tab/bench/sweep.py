# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Multi-seed omnibias-tab vs LightGBM sweep over the full tabular suite (the cluster job).

This is the heavier companion to the CPU-smoke ``docs/examples/tab_validate.py``: it runs
:func:`omnibias.tab.bench.head_to_head` over the full suite for ``K >= 5`` seeds, writes the
per-dataset metrics as JSON to the ``--out`` directory (never large binaries in the tree), and
prints a markdown table to transcribe into ``docs/benchmarks.md`` (the empirical-validation
report artifact). The network-only datasets (``california_housing`` / ``adult`` / ``higgs``)
are skipped automatically when they cannot be fetched, so the same script runs offline on a
CPU-dev host or on a cluster node with network.

Usage (local CPU smoke of the *harness* itself)::

    python packages/omnibias-tab/bench/sweep.py --datasets breast_cancer diabetes --seeds 3

Usage (full sweep; submit to your GPU scheduler)::

    python packages/omnibias-tab/bench/sweep.py --seeds 10 --out ./omnibias_runs/tab

Terminology: ``tab``'s gate ``sigmoid(beta (w.x - t))`` hardens as ``beta -> inf`` (the
feasibility / temperature sense of "collapse"), distinct from the founding ``delta -> 0``
bias collapse; the sweep only trains and scores, it invokes no collapse limit itself.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

# Tiny float64 tree ops are latency-bound: one BLAS/OMP thread per core thrashes and can
# turn a ~20-minute sweep into a many-hour one that the cluster then kills. Pin to a single
# thread *before* numpy / torch import (mirrors tests/conftest.py). Must precede any import
# that pulls torch.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from omnibias.tab.bench import FULL_SUITE, TabConfig, head_to_head  # noqa: E402


def _pin_torch_threads() -> None:
    r"""Best-effort single-thread torch (import is lazy so tab stays backend-free)."""
    try:
        import torch

        torch.set_num_threads(1)
    except ModuleNotFoundError:  # pragma: no cover - torch is an extra
        pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(FULL_SUITE),
                   help=f"datasets to run (default: {list(FULL_SUITE)})")
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (>= 5 for the gate)")
    p.add_argument("--max-rows", type=int, default=None,
                   help="optional per-dataset row cap (speeds up the big datasets)")
    p.add_argument("--method", choices=("boost", "joint"), default="boost")
    p.add_argument("--out", type=str, default="./omnibias_runs/tab",
                   help="artifact directory (JSON summary is written here)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _pin_torch_threads()
    out_dir = Path(os.path.expanduser(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = TabConfig(method=args.method)

    rows: list[dict[str, object]] = []
    print(f"# omnibias-tab vs LightGBM -- {args.seeds} seeds, method={args.method}\n", flush=True)
    for name in args.datasets:
        t0 = time.time()
        # A per-seed heartbeat so the cluster log shows live progress (and survives a kill).
        done = [0]

        def _tick(_s: int, name: str = name, done: list[int] = done, t0: float = t0) -> None:
            done[0] += 1
            print(f"    {name}: seed {done[0]}/{args.seeds} done ({time.time() - t0:.0f}s)",
                  flush=True)

        try:
            h = head_to_head(
                name, seeds=args.seeds, tab_cfg=cfg, max_rows=args.max_rows, on_seed=_tick
            )
        except RuntimeError as exc:
            print(f"  ({name}: skipped -- {exc})", flush=True)
            continue
        s = h.summary()
        s["seconds"] = round(time.time() - t0, 1)
        rows.append(s)
        print(f"  {name:20s} tab={s['tab_mean_primary']:+.4f}+/-{s['tab_std_primary']:.4f}  "
              f"lgbm={s['lgbm_mean_primary']:+.4f}+/-{s['lgbm_std_primary']:.4f}  "
              f"not_worse={s['not_worse']}  ({s['seconds']}s)", flush=True)
        # Persist incrementally so a late kill never loses completed datasets.
        (out_dir / "tab_benchmark.json").write_text(
            json.dumps({"n_seeds": args.seeds, "method": args.method, "results": rows}, indent=2)
        )

    payload = {"n_seeds": args.seeds, "method": args.method, "results": rows}
    (out_dir / "tab_benchmark.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_dir / 'tab_benchmark.json'}", flush=True)

    # A markdown table to paste into docs/benchmarks.md.
    print("\n## transcribe to docs/benchmarks.md\n")
    print("| Dataset | Task | Metric | omnibias-tab | LightGBM | not-worse |")
    print("|---|---|---|---|---|---|")
    for s in rows:
        higher = "acc" if s["task"] != "regression" else "-rmse"
        print(f"| {s['dataset']} | {s['task']} | {higher} | "
              f"{s['tab_mean_primary']:+.4f} +/- {s['tab_std_primary']:.4f} | "
              f"{s['lgbm_mean_primary']:+.4f} +/- {s['lgbm_std_primary']:.4f} | "
              f"{'yes' if s['not_worse'] else 'NO'} |")
    n_ok = sum(1 for s in rows if s["not_worse"])
    print(f"\nnot-worse-than-LightGBM on {n_ok}/{len(rows)} datasets.")


if __name__ == "__main__":
    main()
