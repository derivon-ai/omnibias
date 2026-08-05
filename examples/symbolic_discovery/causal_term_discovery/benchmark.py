# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Controlled causal parent-ranking benchmark on a known linear SEM.

Ground-truth structural causal model (a chain plus an irrelevant variable):

    x0 ~ N(0, 1)
    x1 = 2.0 * x0 + eps1
    x2 = -1.5 * x1 + eps2
    z  ~ N(0, 1)            # spectator, causes nothing

with equal small noise variances on un-standardised data -- the
direction-identifiable regime for linear-Gaussian SEMs.  We check that

* :func:`omnibias.symbolic.causal.causal_discovery_report` recovers the
  *directed* chain ``x0 -> x1 -> x2`` (and no edge to/from the spectator), and
* :func:`omnibias.symbolic.causal.term_parent_ranking` ranks the true parents of
  ``x2`` (namely ``x1``, and ``x0`` indirectly) above the spectator.

Honest scope: NOTEARS-lite returns a *ranking*, not a certified DAG.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from omnibias.symbolic.causal import causal_discovery_report, term_parent_ranking

VARIABLE_NAMES = ["x0", "x1", "x2", "z"]


def make_dataset(*, n_samples: int = 4000, noise_std: float = 0.3, seed: int = 0) -> np.ndarray:
    """Sample the ground-truth chain SEM with one spectator variable."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n_samples)
    x1 = 2.0 * x0 + noise_std * rng.standard_normal(n_samples)
    x2 = -1.5 * x1 + noise_std * rng.standard_normal(n_samples)
    z = rng.standard_normal(n_samples)
    return np.stack([x0, x1, x2, z], axis=1)


def evaluate_benchmark(
    *, n_samples: int = 4000, noise_std: float = 0.3, seed: int = 0
) -> dict[str, Any]:
    """Run the structure + parent-ranking recovery and return a JSON-able report."""
    data = make_dataset(n_samples=n_samples, noise_std=noise_std, seed=seed)

    report = causal_discovery_report(data, VARIABLE_NAMES, w_threshold=0.5)
    recovered_edges = {(src, dst) for src, dst, _ in report["edges"]}
    true_edges = {("x0", "x1"), ("x1", "x2")}

    # Rank parents of x2 among {x0, x1, z}.
    parents_of_x2 = term_parent_ranking(
        data[:, [0, 1, 3]], data[:, 2], ["x0", "x1", "z"], target_name="x2"
    )
    combined_order = [name for name, _ in parents_of_x2["combined_ranking"]]

    return {
        "true_edges": sorted(true_edges),
        "recovered_edges": sorted(recovered_edges),
        "structure_exact": recovered_edges == true_edges,
        "acyclicity_residual": report["acyclicity"],
        "edge_weights": [
            {"src": s, "dst": d, "weight": w} for s, d, w in report["edges"]
        ],
        "x2_parent_combined_ranking": parents_of_x2["combined_ranking"],
        "x2_parent_weights": parents_of_x2["notears_parent_weights"],
        "spectator_ranked_last": combined_order[-1] == "z",
        "honesty_note": report["note"],
        "fairness_protocol": {
            "data_standardized": False,
            "identifiability": "linear-Gaussian equal-noise, raw scale",
            "claim": "directed parent RANKING, not a certified DAG",
        },
    }


def write_artifacts(results: dict[str, Any], out_dir: Path) -> None:
    """Write the benchmark report to ``out_dir/report.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(results, indent=2, sort_keys=True))
