# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Build ``docs/benchmarks/full_scope.json``: one row per measured cell against every baseline.

Every row names its source record, the baseline it was compared with, the ratio ``baseline time / omnibias time``,
the interval that was recorded and the verdict. Nothing is measured here: the script reads ``docs/benchmarks`` and
re-states each record's result in one scheme (no imports of the packages). ``--check`` fails when the file or the
generated blocks of the docs are stale.

Verdict (ratio = baseline / omnibias, larger is faster for omnibias):

* ``big_win``  lower bound of the interval >= 2
* ``win``      lower bound > 1
* ``parity``   the interval contains 1
* ``loss``     upper bound < 1 (omnibias slower with confidence)
* ``incomplete`` the record has no qualified baseline for the cell (the baseline failed, was refused or did not qualify)

``basis`` tells how the interval was obtained: ``t95_3proc`` (three fresh processes: the recorded one-sided 95 % t bound
``lower95 = exp(mean(log r) - 2.920 sd(log r) / sqrt(3))`` and its mirror ``exp(mean(log r) + 2.920 sd(log r) / sqrt(3))
= ratio^2 / lower95``; the verdict reads that interval, not the status written in the record), ``t95_seeds`` (the same bound over
the five seeds of a registered end-to-end suite, with ``t(0.95, 4)``), ``bootstrap_samples`` (95 % percentile bootstrap of the recorded timing samples), ``process_median``
(median of process medians, no interval) or ``point`` (one recorded median per cell; the interval is the ratio times
and divided by 1.10, a noise band wider than the 5 % that same-tree repeats showed in the 2026-10 audit, not a bootstrap).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "docs" / "benchmarks"
OUT = BENCH / "full_scope.json"
SCHEMA = "omnibias/full-scope/v1"
VERDICTS = ("big_win", "win", "parity", "loss", "incomplete")
POINT_BAND = 0.10
BOOTSTRAP = 4000

RULES = {
    "ratio": "baseline time / omnibias time (memory rows: baseline bytes / omnibias bytes)",
    "big_win": "lower bound of the interval >= 2",
    "win": "lower bound > 1",
    "parity": "the interval contains 1",
    "loss": "upper bound < 1",
    "incomplete": "no qualified baseline for the cell in the record",
    "t95_3proc": "three fresh processes: the recorded one-sided 95% t bound of the log ratio (lower95) and its mirror ratio^2 / lower95",
    "t95_seeds": "per-seed ratios of a registered suite (five seeds): the one-sided 95% t bound of the log ratio, exp(mean(log r) - t(0.95, 4) sd(log r) / sqrt(5)), and its mirror ratio^2 / lower",
    "bootstrap_samples": "95% percentile bootstrap (4,000 resamples) of the recorded timing samples of both methods",
    "bootstrap_rounds": "95% percentile hierarchical bootstrap over fresh-process rounds and the samples within each round",
    "process_median": "median of process medians; the interval is the ratio divided and multiplied by 1.10",
    "point": "one recorded median per cell; the interval is the ratio divided and multiplied by 1.10 (a noise band, not a bootstrap)",
}

# record stem -> (operator, backend, device, dtype)
ROW_RECORDS: dict[str, tuple[str, str, str, str]] = {
    "closed_ridge_grid_gauss_newton_a100": ("tensor-grid Gauss-Newton step", "jax", "A100", "float64"),
    "closed_ridge_grid_options_a100": ("tensor-grid loss", "jax", "A100", "float64"),
    "closed_ridge_grid_training_a100": ("tensor-grid training step", "jax", "A100", "float64"),
    "closed_ridge_training_a100": ("ridge residual loss", "jax", "A100", "float64"),
    "deep_residual_gram_a100": ("residual Gram and Gauss-Newton step", "jax", "A100", "float64"),
    "deep_residual_ntk_diagonal_a100": ("NTK diagonal", "jax", "A100", "float64"),
    "fourth_order_residual_gram_a100": ("fourth-order residual Gram", "jax", "A100", "float64"),
    "fused_closed_ridge_training_a100": ("ridge value-and-gradient (fused)", "jax", "A100", "float64"),
    "projected_deep_bilaplacian_a100": ("bi-Laplacian", "jax", "A100", "float64"),
    "projected_deep_bilaplacian_time_a100": ("bi-Laplacian (time-dependent)", "jax", "A100", "float64"),
    "projected_deep_divergence_a100": ("divergence", "jax", "A100", "float64"),
    "projected_deep_laplacian_a100": ("Laplacian (deep)", "jax", "A100", "float64"),
    "projected_deep_laplacian_point_diffusion_a100": ("Laplacian (point diffusion)", "jax", "A100", "float64"),
    "projected_deep_laplacian_time_a100": ("Laplacian (time-dependent)", "jax", "A100", "float64"),
    "projected_deep_logdet_a100": ("log-determinant", "jax", "A100", "float64"),
    "projected_ntk_diagonal_a100": ("NTK diagonal (projected)", "jax", "A100", "float64"),
    "projected_residual_gram_a100": ("residual Gram (projected)", "jax", "A100", "float64"),
    "ridge_taylor_mode_a100": ("ridge operators against Taylor mode", "jax", "A100", "float64"),
    "separable_grid_gauss_newton_a100": ("separable-grid Gauss-Newton step", "jax", "A100", "float64"),
    "separable_grid_training_a100": ("separable-grid loss", "jax", "A100", "float64"),
}
STATUS_VERDICT = {
    "significant_2x": "big_win",
    "lower95_ge_1": "win",
    "parity": "parity",
    "loss": "loss",
    "partial": "incomplete",
    "suppressed": "incomplete",
    "PROMOTE": "win",
    "parity/uncertain": "parity",
    "LOSS": "loss",
}
# tiers that are omnibias's own alternative implementations, not another package
INTERNAL_TIERS = frozenset(
    {
        "per_point_closed_form_jacobian", "omnibias_per_point_closed_form", "omnibias_xla_closed_form",
        "explicit_jacobian_shared_streams", "identity_every_layer", "gram_then_diagonal",
        "explicit_rows_from_shared_streams", "previous_package_exact", "previous_interface", "explicit_selection",
        "projected_streams_explicit_first_layer_rows", "full_seed_stream_identity", "projected_per_row",
        "projected_without_identity", "explicit_jacobian_full_seed", "a26_stream_module", "previous_package",
        "full_input_recursion", "inner_proj", "ring_jvp",
    }
)
# generic engines (folx, collapsed Taylor mode, JVPs) given omnibias's first-layer projection: not independent baselines
PROJECTED_ENGINE_TIERS = frozenset({"projected_generic_engines", "projected_generic"})
PROJECTED_ENGINE_NAMES = frozenset({"folx3_proj"})  # the same, named per engine in the tri-Laplacian record
# tier members that are not another package: ``omnibias_a`` is the package's own earlier deep-Laplacian program (snapshot A of
# the audit) and ``fl_own`` the benchmark worker's hand-written forward Laplacian, the control the record names beside it
MEMBER_KINDS = {"omnibias_a": "omnibias_variant", "fl_own": "hand_written"}
OMNIBIAS_PREFIXES = ("omnibias",)
KINDS = ("external", "engine_on_omnibias_projection", "hand_written", "omnibias_variant")
GROUPS = {
    "activation derivative": "Activation derivatives",
    "Laplacian (one layer)": "One-layer Laplacian",
    "poly-Laplacian (one layer)": "Higher-order operators",
    "bi-Laplacian": "Higher-order operators",
    "bi-Laplacian (time-dependent)": "Higher-order operators",
    "bi-Laplacian (deep, per call)": "Higher-order operators",
    "tri-Laplacian / PFC residual": "Higher-order operators",
    "third-order directional derivative (per call)": "Higher-order operators",
    "Laplacian (deep)": "Deep Laplacian, projected",
    "Laplacian (point diffusion)": "Deep Laplacian, projected",
    "Laplacian (time-dependent)": "Deep Laplacian, projected",
    "Laplacian (deep, per call)": "Deep Laplacian, per-call grid",
    "divergence": "Divergence and log-determinant",
    "log-determinant": "Divergence and log-determinant",
    "residual solve (Woodbury)": "Residual solves",
    "residual inverse (auto)": "Residual solves",
    "residual Gram and Gauss-Newton step": "Residual Gram, Gauss-Newton step, NTK diagonal",
    "fourth-order residual Gram": "Residual Gram, Gauss-Newton step, NTK diagonal",
    "residual Gram (projected)": "Residual Gram, Gauss-Newton step, NTK diagonal",
    "NTK diagonal": "Residual Gram, Gauss-Newton step, NTK diagonal",
    "NTK diagonal (projected)": "Residual Gram, Gauss-Newton step, NTK diagonal",
    "ridge residual loss": "Ridge and tensor-grid losses",
    "ridge value-and-gradient (fused)": "Ridge and tensor-grid losses",
    "ridge operators against Taylor mode": "Ridge and tensor-grid losses",
    "tensor-grid loss": "Ridge and tensor-grid losses",
    "tensor-grid training step": "Ridge and tensor-grid losses",
    "tensor-grid Gauss-Newton step": "Ridge and tensor-grid losses",
    "separable-grid loss": "Ridge and tensor-grid losses",
    "separable-grid Gauss-Newton step": "Ridge and tensor-grid losses",
    "training step": "Training step",
    "time to target (end to end)": "End-to-end time to target",
    "folx drop-in (folx_compat)": "Interop adapters and drop-ins",
    "planner on a folx call": "Interop adapters and drop-ins",
    "DeepXDE adapter": "Interop adapters and drop-ins",
    "PINA adapter": "Interop adapters and drop-ins",
    "operator compiler": "Operator compiler",
}


def verdict_of(lower: float | None, upper: float | None, ratio: float) -> str:
    """The verdict of an interval ``[lower, upper]`` of ``baseline / omnibias``; a missing bound is the ratio's noise band."""
    lo = ratio / (1 + POINT_BAND) if lower is None and upper is None else lower
    hi = ratio * (1 + POINT_BAND) if lower is None and upper is None else upper
    if lo is not None and lo >= 2:
        return "big_win"
    if lo is not None and lo > 1:
        return "win"
    if hi is not None and hi < 1:
        return "loss"
    return "parity"


def upper95(ratio: float, lower: float) -> float | None:
    """The mirrored one-sided bound of a record that stores ``lower95 = exp(mean(log r) - 2.920 sd(log r) / sqrt(3))`` next to
    ``ratio = exp(mean(log r))`` (three fresh processes): ``exp(mean(log r) + 2.920 sd(log r) / sqrt(3)) = ratio^2 / lower95``."""
    return ratio * ratio / lower if ratio > 0 and lower > 0 else None


def band(ratio: float) -> tuple[float, float]:
    return ratio / (1 + POINT_BAND), ratio * (1 + POINT_BAND)


def compact(config: dict[str, Any]) -> str:
    skip = {"operator_definition", "projection_applies"}
    return ",".join(f"{k}={v}" for k, v in config.items() if k not in skip and v is not None)


def row(
    rid: str, record: str, operator: str, backend: str, device: str, dtype: str, endpoint: str, cell: str,
    baseline: str, kind: str, ratio: float | None, lower: float | None, upper: float | None,
    verdict: str, basis: str, note: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": rid, "record": record, "operator": operator, "backend": backend, "device": device, "dtype": dtype,
        "endpoint": endpoint, "cell": cell, "baseline": baseline, "baseline_kind": kind,
        "ratio": None if ratio is None else round(ratio, 4),
        "lower": None if lower is None else round(lower, 4),
        "upper": None if upper is None else round(upper, 4),
        "verdict": verdict, "basis": basis,
    }
    if note:
        out["note"] = note
    return out


def load(name: str) -> dict[str, Any]:
    result: dict[str, Any] = json.loads((BENCH / name).read_text())
    return result


# --------------------------------------------------------------------------------------------------------------------
# records with one row per (cell, tier) and a recorded status
# --------------------------------------------------------------------------------------------------------------------


def from_row_record(stem: str) -> list[dict[str, Any]]:
    operator, backend, device, dtype = ROW_RECORDS[stem]
    data = load(f"{stem}.json")
    rows = []
    # a candidate the record itself marks as "no shipped module" is not omnibias code a user can call
    unshipped = {name: text for name, text in (data.get("candidates") or {}).items() if "no shipped module" in text}
    for index, r in enumerate(data["rows"]):
        cfg = r.get("config", {})
        status = r["status"]
        verdict = STATUS_VERDICT[status]
        tier = r.get("tier") or r.get("strongest_control") or "tanh_ridge_control"
        kind = ("omnibias_variant" if tier in INTERNAL_TIERS
                else "engine_on_omnibias_projection" if tier in PROJECTED_ENGINE_TIERS else "external")
        kind = MEMBER_KINDS.get(str(r.get("strongest_control")), kind) if kind == "external" else kind
        ratio, lower = r.get("speed_ratio"), r.get("lower95")
        upper = None
        if ratio is None or lower is None:
            verdict = "incomplete"
        elif verdict != "incomplete":
            # one rule for every row: the verdict reads the interval [lower95, ratio^2 / lower95], not the record's own status
            upper = upper95(ratio, lower)
            verdict = verdict_of(lower, upper, ratio)
        if r.get("candidate") in unshipped:
            verdict = "incomplete"
        endpoint = str(cfg.get("regime") or r.get("regime") or "value_and_gradient")
        note = ""
        if verdict == "incomplete":
            failed = r.get("failed_tier_members") or {}
            note = "; ".join(f"{k}: {v}" for k, v in failed.items()) or status
            if r.get("candidate") in unshipped:
                note = f"program {r['candidate']} is not shipped: {unshipped[r['candidate']]}"
        cell = compact(cfg) if cfg else str(r.get("workload", ""))
        if r.get("role") not in (None, "held_out"):
            cell += f",role={r['role']}"
        out = row(
            f"{stem}/{index}", f"docs/benchmarks/{stem}.json", operator, backend, device, dtype, endpoint, cell,
            f"{tier} ({r.get('strongest_control', '')})" if r.get("strongest_control") else tier, kind,
            ratio, lower, upper, verdict, "t95_3proc", note,
        )
        rows.append(out)
    return rows


# --------------------------------------------------------------------------------------------------------------------
# shipped-module records with held-out rows per baseline
# --------------------------------------------------------------------------------------------------------------------


def from_trilaplacian() -> list[dict[str, Any]]:
    stem = "trilaplacian_a100"
    data = load(f"{stem}.json")
    rows = []
    for index, r in enumerate(data["held_out"]["rows"]):
        cell = f"op={r['op']},D={r['D']},H={r['H']},B={r['B']}"
        for name, rec in r["ratios"].items():
            kind = ("omnibias_variant" if name in INTERNAL_TIERS
                    else "engine_on_omnibias_projection" if name in PROJECTED_ENGINE_NAMES else "external")
            ratio, lower = rec["ratio"], rec["lower95"]
            rows.append(
                row(f"{stem}/{index}/{name}", f"docs/benchmarks/{stem}.json", "tri-Laplacian / PFC residual", "jax", "A100",
                    "float64", str(r["regime"]), cell, name, kind, ratio, lower, upper95(ratio, lower),
                    verdict_of(lower, upper95(ratio, lower), ratio), "t95_3proc")
            )
    return rows


def from_woodbury() -> list[dict[str, Any]]:
    stem = "woodbury_a100"
    data = load(f"{stem}.json")
    rows = []
    for index, r in enumerate(data["held_out"]["rows"]):
        cell = f"D={r['D']},H={r['H']},n={r['n']},B={r['B']}"
        for name, rec in r["measured"].items():
            kind = "omnibias_variant" if name in INTERNAL_TIERS else "external"
            ratio, lower = rec["ratio"], rec["lower95"]
            rows.append(
                row(f"{stem}/{index}/{name}", f"docs/benchmarks/{stem}.json", "residual solve (Woodbury)", "jax", "A100",
                    "float64", str(r["regime"]), cell, name, kind, ratio, lower, upper95(ratio, lower),
                    verdict_of(lower, upper95(ratio, lower), ratio), "t95_3proc")
            )
    return rows


# --------------------------------------------------------------------------------------------------------------------
# CPU records with timing samples
# --------------------------------------------------------------------------------------------------------------------


def bootstrap_ratio(num: list[float], den: list[float], seed: int) -> tuple[float, float, float]:
    """Ratio of medians ``median(num) / median(den)`` and its 95 % percentile interval."""
    rng = random.Random(seed)
    point = statistics.median(num) / statistics.median(den)
    draws = sorted(
        statistics.median(rng.choices(num, k=len(num))) / statistics.median(rng.choices(den, k=len(den)))
        for _ in range(BOOTSTRAP)
    )
    return point, draws[int(0.025 * BOOTSTRAP)], draws[int(0.975 * BOOTSTRAP) - 1]


def samples_of(rec: dict[str, Any]) -> list[float] | None:
    if rec.get("status", "ok") != "ok":
        return None
    s = rec.get("samples_ms")
    return [float(x) for x in s] if s else None


def sampled_rows(
    stem: str, operator: str, cell: str, candidate: dict[str, Any], baselines: dict[str, Any], endpoint: str, seed: int,
) -> list[dict[str, Any]]:
    out = []
    mine = samples_of(candidate)
    for name, rec in baselines.items():
        if name == "omnibias":
            continue
        theirs = samples_of(rec)
        rid = f"{stem}/{cell}/{name}"
        if mine is None or theirs is None:
            status = rec.get("status", "ok") if theirs is None else candidate.get("status", "ok")
            out.append(row(rid, f"docs/benchmarks/{stem}.json", operator, "jax/torch", "CPU", "float64", endpoint, cell,
                           name, "external", None, None, None, "incomplete", "bootstrap_samples", f"baseline status: {status}"))
            continue
        ratio, lo, hi = bootstrap_ratio(theirs, mine, seed)
        out.append(row(rid, f"docs/benchmarks/{stem}.json", operator, "jax/torch", "CPU", "float64", endpoint, cell,
                       name, "external", ratio, lo, hi, verdict_of(lo, hi, ratio), "bootstrap_samples"))
    return out


def from_derivative_order() -> list[dict[str, Any]]:
    stem = "derivative_order"
    data = load(f"{stem}.json")
    rows: list[dict[str, Any]] = []
    for r in data["rows"]:
        meas = r["measurements"]
        cand = meas.get("closed_form")
        if cand is None:
            continue
        rows += sampled_rows(stem, "activation derivative", f"order={r['n']}", cand, meas, "forward", 1000 + int(r["n"]))
    return rows


def from_laplacian_scaling() -> list[dict[str, Any]]:
    stem = "laplacian_scaling"
    data = load(f"{stem}.json")
    rows: list[dict[str, Any]] = []
    for r in data["rows"]:
        rows += sampled_rows(stem, "Laplacian (one layer)", f"D={r['D']}", r["methods"]["omnibias"], r["methods"],
                             "forward", 2000 + int(r["D"]))
    return rows


def from_polylaplacian_order() -> list[dict[str, Any]]:
    stem = "polylaplacian_order"
    data = load(f"{stem}.json")
    rows: list[dict[str, Any]] = []
    for r in data["rows"]:
        base = {k: v for k, v in r.items() if isinstance(v, dict) and "status" in v}
        rows += sampled_rows(stem, "poly-Laplacian (one layer)", f"k={r['k']}", base["omnibias"], base,
                             "forward", 3000 + int(r["k"]))
    return rows


# --------------------------------------------------------------------------------------------------------------------
# per-call grid (one recorded median per cell and method)
# --------------------------------------------------------------------------------------------------------------------

PARITY_TOL = {"f64": 1e-8, "f32h": 2e-3, "f32t": 5e-2}
DTYPE = {"f64": "float64", "f32h": "float32 (highest)", "f32t": "float32 (tf32)"}
ORDER_OPERATOR = {2: "Laplacian (deep, per call)", 3: "third-order directional derivative (per call)",
                  4: "bi-Laplacian (deep, per call)"}


def _time(rec: dict[str, Any], tag: str, mode: str, prec: str) -> float | None:
    if "error" in rec:
        return None
    parity = rec.get("parity_max_rel_err")
    if parity is not None and not parity <= PARITY_TOL[prec]:
        return None
    key = f"{tag}_ms" if mode == "eager" else f"{tag}_device_ms"
    v = rec.get(key)
    return float(v) if isinstance(v, int | float) and v > 0 else None


def _stalled(rec: dict[str, Any], tag: str) -> bool:
    host, device = rec.get(f"{tag}_ms"), rec.get(f"{tag}_device_ms")
    return bool(isinstance(host, int | float) and isinstance(device, int | float) and host > 1.0 and host > 10 * device)


def from_percall() -> list[dict[str, Any]]:
    stem = "percall_grid"
    data = load(f"{stem}.json")
    rows: list[dict[str, Any]] = []
    # host-timed medians that are more than twice their minimum (a block of two calls decided by one stall)
    suspect = {(t["file"], t["cell"], t["method"], t["tag"]) for t in data["verdicts"].get("suspect_timings", [])}
    for label, cells in data["cells"].items():
        backend, prec, ordr = label.split("/")
        order = int(ordr[1:])
        for c in cells:
            cell = f"D={c['D']},H={c['H']},L={c['L']},B={c['B']}"
            methods = c["methods"]
            for mode in ("eager", "device"):
                for tag, endpoint in (("fwd", "forward"), ("vag", "value_and_gradient")):
                    if order == 2:
                        cand_names = [n for n in ("omnibias_default_train", "omnibias_default") if n in methods]
                        omni = next(((n, _time(methods[n], tag, mode, prec)) for n in cand_names
                                     if _time(methods[n], tag, mode, prec)), None)
                        base_names = [n for n in methods if not n.startswith(OMNIBIAS_PREFIXES) and n != "folx_nested"]
                    else:
                        cands = [(n, _time(r, tag, mode, prec)) for n, r in methods.items() if n.startswith(OMNIBIAS_PREFIXES)]
                        cands = [(n, t) for n, t in cands if t]
                        omni = min(cands, key=lambda x: x[1]) if cands else None
                        base_names = [n for n in methods if not n.startswith(OMNIBIAS_PREFIXES)]
                    bases = [(n, _time(methods[n], tag, mode, prec)) for n in base_names]
                    bases = [(n, t) for n, t in bases if t]
                    rid = f"{stem}/{label}/{cell}/{tag}/{mode}"
                    if omni is None or not omni[1] or not bases:
                        rows.append(row(rid, f"docs/benchmarks/{stem}.json", ORDER_OPERATOR[order], backend, "A100",
                                        DTYPE[prec], f"{endpoint}/{mode}", cell, "none qualified", "external", None, None, None,
                                        "incomplete", "point", "no qualified omnibias or baseline timing"))
                        continue
                    name, t = min(bases, key=lambda x: x[1])
                    ratio = t / omni[1]
                    sfile = f"{backend}_{prec}_o{order}.json"
                    skey = f"D{c['D']}H{c['H']}L{c['L']}B{c['B']}"
                    stag = "forward" if tag == "fwd" else "value_and_grad"
                    stalled = [m for m in (omni[0], name) if _stalled(methods[m], tag)]
                    if mode == "eager" and ((sfile, skey, omni[0], stag) in suspect or (sfile, skey, name, stag) in suspect or stalled):
                        rows.append(row(rid, f"docs/benchmarks/{stem}.json", ORDER_OPERATOR[order], backend, "A100", DTYPE[prec],
                                        f"{endpoint}/{mode}", cell, name, "external", ratio, None, None, "incomplete", "point",
                                        "host-timed median is suspect (more than twice its minimum, or above 1 ms and 10 times "
                                        "the device time of the same program)"))
                        continue
                    rows.append(row(rid, f"docs/benchmarks/{stem}.json", ORDER_OPERATOR[order], backend, "A100", DTYPE[prec],
                                    f"{endpoint}/{mode}", cell, name, "external", ratio, *band(ratio),
                                    verdict_of(None, None, ratio), "point", f"omnibias program: {omni[0]}"))
    for step, doc in data["training_step"].items():
        _, backend, prec = step.split("_")
        for dim, methods in doc["dims"].items():
            omni = methods.get("omnibias_planned", {}).get("ms_per_step")
            for name, rec in methods.items():
                if name == "omnibias_planned":
                    continue
                cell = f"D={dim},width={doc['protocol']['width']},depth={doc['protocol']['depth']}"
                rid = f"{stem}/{step}/{cell}/{name}"
                t = rec.get("ms_per_step")
                if omni is None or t is None:
                    rows.append(row(rid, f"docs/benchmarks/{stem}.json", "training step", backend, "A100", DTYPE[prec], "step",
                                    cell, name, "external", None, None, None, "incomplete", "point",
                                    str(rec.get("skipped", rec.get("error", "not run")))))
                    continue
                ratio = t / omni
                rows.append(row(rid, f"docs/benchmarks/{stem}.json", "training step", backend, "A100", DTYPE[prec], "step",
                                cell, name, "external", ratio, *band(ratio), verdict_of(None, None, ratio), "point"))
    return rows


# --------------------------------------------------------------------------------------------------------------------
# operator compiler (CPU, median of process medians)
# --------------------------------------------------------------------------------------------------------------------


def from_operator_compiler() -> list[dict[str, Any]]:
    stem = "operator_compiler_v2_summary"
    data = load(f"{stem}.json")
    rows: list[dict[str, Any]] = []
    for rep in data["reports"]:
        cfg = rep.get("config", {})
        for index, comp in enumerate(rep.get("comparisons", [])):
            if comp["compiled_method"] != "compiler":
                continue  # the other compiled methods are baselines of the compiler, not candidates
            cell = (f"family={comp['family']},order={comp['order']},dim={cfg.get('dim')},hidden={cfg.get('hidden')},"
                    f"batch={cfg.get('batch')}")
            for name, over in comp["baseline_over_compiler"].items():
                kind = {"keras-jax": "omnibias_variant", "manual": "hand_written", "jax-manual": "hand_written"}.get(name, "external")
                rows.append(row(f"{stem}/{rep['archive_path']}/{index}/{name}", f"docs/benchmarks/{stem}.json",
                                "operator compiler", "jax/torch", "CPU", "float64", "derivative-only", cell, name, kind,
                                over, *band(over), verdict_of(None, None, over), "process_median"))
    return rows


FARM_SCHEMA = "omnibias/farm-ratios/v1"
FARM_FIELDS = ("operator", "backend", "device", "dtype", "endpoint", "cell", "baseline", "baseline_kind", "ratio", "lower", "upper")


def from_farm_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Records re-measured with replicate rounds (``farm_*.json``): the rows already hold the bootstrap interval."""
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in sorted(BENCH.glob("farm_*.json")):
        data = json.loads(path.read_text())
        if data.get("schema") != FARM_SCHEMA:
            raise ValueError(f"{path.name}: schema must be {FARM_SCHEMA}")
        part = []
        for index, r in enumerate(data["rows"]):
            missing = [k for k in FARM_FIELDS if k not in r]
            if missing:
                raise ValueError(f"{path.name} row {index}: missing {missing}")
            basis = r.get("basis", "bootstrap_rounds")
            if basis not in RULES:
                raise ValueError(f"{path.name} row {index}: unknown basis {basis!r}")
            if r["ratio"] is None or r["lower"] is None or r["upper"] is None:
                verdict = "incomplete"
            else:
                verdict = verdict_of(r["lower"], r["upper"], r["ratio"])
            part.append(row(f"{path.stem}/{index}", f"docs/benchmarks/{path.name}", r["operator"], r["backend"], r["device"],
                            r["dtype"], r["endpoint"], r["cell"], r["baseline"], r["baseline_kind"], r["ratio"], r["lower"],
                            r["upper"], verdict, basis, r.get("note", "")))
        rows += part
        source: dict[str, Any] = {"record": f"docs/benchmarks/{path.name}", "rows": len(part)}
        declared = data.get("supersedes", [])
        if declared:
            source["supersedes"] = [f"docs/benchmarks/{name}" for name in declared]
        sources.append(source)
    return rows, sources


def cell_key(r: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(r[k]) for k in ("operator", "backend", "device", "dtype", "endpoint", "cell"))


def build() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for stem in sorted(ROW_RECORDS):
        part = from_row_record(stem)
        rows += part
        sources.append({"record": f"docs/benchmarks/{stem}.json", "rows": len(part)})
    for name, fn in (
        ("trilaplacian_a100", from_trilaplacian), ("woodbury_a100", from_woodbury),
        ("derivative_order", from_derivative_order), ("laplacian_scaling", from_laplacian_scaling),
        ("polylaplacian_order", from_polylaplacian_order), ("percall_grid", from_percall),
        ("operator_compiler_v2_summary", from_operator_compiler),
    ):
        part = fn()
        rows += part
        sources.append({"record": f"docs/benchmarks/{name}.json", "rows": len(part)})
    # A record that kept an earlier measurement of its own module before a fix (role superseded_*) and re-measured the cell after
    # the fix (role fix_verification, same cell and baseline) holds one current measurement: the earlier row describes code that
    # no longer ships, so the re-measured row replaces it, and the count is recorded with the source.
    current = {(r["record"], r["baseline"], r["endpoint"], r["cell"]) for r in rows if "role=fix_verification" in r["cell"]}
    stale = [r for r in rows if "role=superseded_" in r["cell"] and (r["record"], r["baseline"], r["endpoint"],
             re.sub(r"role=superseded_\w+", "role=fix_verification", r["cell"])) in current]
    if stale:
        stale_ids = {r["id"] for r in stale}
        rows = [r for r in rows if r["id"] not in stale_ids]
        for source in sources:
            n = sum(1 for r in stale if r["record"] == source["record"])
            if n:
                source["replaced_by_fix_verification"] = n
    farm_rows, farm_sources = from_farm_records()
    # A farm record that re-measures a cell (same operator, back end, device, dtype, endpoint and cell) replaces the rows an older
    # record holds for it, so the matrix keeps one measurement per cell; the count is recorded with the source.
    # Two farm records may hold the same cell only when the newer one names the older in its ``supersedes`` header.
    declared = {src["record"]: set(src.get("supersedes", [])) for src in farm_sources}
    holders: dict[tuple[str, ...], set[str]] = {}
    for r in farm_rows:
        holders.setdefault(cell_key(r), set()).add(r["record"])
    farm_cells: dict[tuple[str, ...], str] = {}
    for key, records in holders.items():
        winners = [rec for rec in records if records - {rec} <= declared[rec]]
        if len(winners) != 1:
            raise ValueError(f"two farm records measure the same cell without a supersedes header: {key} {sorted(records)}")
        farm_cells[key] = winners[0]
    farm_dropped = [r for r in farm_rows if farm_cells[cell_key(r)] != r["record"]]
    farm_rows = [r for r in farm_rows if farm_cells[cell_key(r)] == r["record"]]
    kept = [r for r in rows if cell_key(r) not in farm_cells]
    replaced = kept != rows
    sources += farm_sources
    if replaced or farm_dropped:
        gone = [r for r in rows if cell_key(r) in farm_cells] + farm_dropped
        for source in sources:
            n = sum(1 for r in gone if r["record"] == source["record"])
            if n:
                source["superseded"] = n
    rows = kept + farm_rows
    for r in rows:
        r["family"] = GROUPS[r["operator"]]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        dup = [k for k, v in Counter(ids).items() if v > 1][:5]
        raise ValueError(f"duplicate row ids: {dup}")
    return {
        "schema": SCHEMA,
        "rules": RULES,
        "sources": sources,
        "summary": {
            "overall": tally(rows),
            "by_kind": by(rows, lambda r: r["baseline_kind"]),
            "by_family_independent": by(rows, group_key, "external"),
        },
        "rows": rows,
    }


def tally(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(r["verdict"] for r in rows)
    ratios = [r["ratio"] for r in rows if r["verdict"] != "incomplete" and r["ratio"] is not None]
    return {**{v: counts.get(v, 0) for v in VERDICTS}, "rows": len(rows),
            "ratio_min": min(ratios) if ratios else None, "ratio_max": max(ratios) if ratios else None}


def by(rows: list[dict[str, Any]], key: Any, kind: str | None = None) -> dict[str, dict[str, Any]]:
    keys = sorted({key(r) for r in rows})
    return {k: tally([r for r in rows if key(r) == k and (kind is None or r["baseline_kind"] == kind)]) for k in keys}


def fmt_ratio(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:,.0f}" if x >= 100 else f"{x:.2f}" if x < 10 else f"{x:.1f}"


def render_table(rows: list[dict[str, Any]], key: Any, title: str, kind: str | None) -> list[str]:
    lines = [f"| {title} | cells | big win | win | parity | loss | incomplete | ratio range |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    total = Counter()
    for name, t in by(rows, key, kind).items():
        if not t["rows"]:
            continue
        lines.append(f"| {name} | {t['rows']:,} | {t['big_win']:,} | {t['win']:,} | {t['parity']:,} | {t['loss']:,} | "
                     f"{t['incomplete']:,} | {fmt_ratio(t['ratio_min'])}-{fmt_ratio(t['ratio_max'])}× |")
        total.update({v: t[v] for v in VERDICTS} | {"rows": t["rows"]})
    lines.append(f"| **all** | {total['rows']:,} | {total['big_win']:,} | {total['win']:,} | {total['parity']:,} | "
                 f"{total['loss']:,} | {total['incomplete']:,} | |")
    return lines


def group_key(r: dict[str, Any]) -> str:
    return str(r["family"])


MATRIX_BEGIN = "<!-- BEGIN GENERATED FULL SCOPE -->"
MATRIX_END = "<!-- END GENERATED FULL SCOPE -->"


def render_baseline_wins(rows: list[dict[str, Any]]) -> list[str]:
    """Every (operator, backend, baseline) against which at least one independent cell is a loss, most losses first."""
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if r["baseline_kind"] == "external":
            cells.setdefault((r["operator"], r["backend"], r["baseline"].split(" (")[0]), []).append(r)
    lines = ["| operator | backend | baseline | cells | loss | parity | ratio range of the losses |", "|---|---|---|---:|---:|---:|---|"]
    ranked = []
    for (operator, backend, baseline), group in cells.items():
        losses = [r["ratio"] for r in group if r["verdict"] == "loss"]
        if losses:
            ranked.append((-len(losses), operator, backend, baseline, len(group), len(losses),
                           sum(r["verdict"] == "parity" for r in group), min(losses), max(losses)))
    for _, operator, backend, baseline, n, loss, parity, lo, hi in sorted(ranked):
        lines.append(f"| {operator} | {backend} | `{baseline}` | {n:,} | {loss:,} | {parity:,} | {lo:.2f}-{hi:.2f}× |")
    return lines


def render_doc_block(rows: list[dict[str, Any]]) -> str:
    """The generated matrix summary of ``docs/performance.md`` (counts per operator family and per operator)."""
    out = [
        "Ratio = baseline time / omnibias time; every row of "
        "[`docs/benchmarks/full_scope.json`](benchmarks/full_scope.json) names its record, interval and verdict "
        "(`big win`: lower bound >= 2, `win`: lower bound > 1, `parity`: the interval contains 1, `loss`: upper bound < 1, "
        "`incomplete`: no qualified baseline in the record). Rows are cells (an operator at one configuration against "
        "one baseline tier).",
        "",
        "**Against independent baselines** (nested autodiff, dense Hessian and Jacobian, `jax.experimental.jet`, `folx`, "
        "PyTorch `jet_collapsed`, DeepXDE, GMRES, explicit grids):",
        "",
        *render_table(rows, group_key, "family", "external"),
        "",
        "**Against engines fed omnibias's own first-layer projection** (the generic engine is `folx` or collapsed Taylor mode "
        "on the reduced function; the projection is omnibias's):",
        "",
        *render_table(rows, group_key, "family", "engine_on_omnibias_projection"),
        "",
        "**Against omnibias's own variants and hand-written closed forms** (a loss here is one omnibias program beating "
        "another):",
        "",
        *render_table([r for r in rows if r["baseline_kind"] in ("omnibias_variant", "hand_written")], group_key, "family", None),
        "",
        "**Per operator, independent baselines:**",
        "",
        *render_table(rows, lambda r: r["operator"], "operator", "external"),
        "",
        "**Where an independent baseline is faster:** every operator, back end and baseline with at least one loss cell. "
        "Cells that tie are counted under parity; the cell list with its record, interval and basis is in the JSON.",
        "",
        *render_baseline_wins(rows),
    ]
    return "\n".join(out)


def replace_block(text: str, content: str) -> str:
    if text.count(MATRIX_BEGIN) != 1 or text.count(MATRIX_END) != 1:
        raise ValueError("expected one full-scope marker pair")
    start, finish = text.index(MATRIX_BEGIN), text.index(MATRIX_END)
    return text[:start] + MATRIX_BEGIN + "\n\n" + content + "\n\n" + text[finish:]


def dumps(doc: dict[str, Any]) -> str:
    head = {k: v for k, v in doc.items() if k != "rows"}
    text = json.dumps(head, indent=1)[:-2] + ',\n "rows": [\n'
    text += ",\n".join("  " + json.dumps(r, separators=(",", ":")) for r in doc["rows"]) + "\n ]\n}\n"
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if docs/benchmarks/full_scope.json is stale")
    args = parser.parse_args(argv)
    doc = build()
    text = dumps(doc)
    page = ROOT / "docs" / "performance.md"
    page_text = replace_block(page.read_text(), render_doc_block(doc["rows"])) if page.exists() and MATRIX_BEGIN in page.read_text() else None
    if args.check:
        stale = [name for name, ok in (("docs/benchmarks/full_scope.json", OUT.exists() and OUT.read_text() == text),
                                       ("docs/performance.md", page_text is None or page.read_text() == page_text)) if not ok]
        if stale:
            print(f"stale: {', '.join(stale)}: run scripts/build_full_scope.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text)
    if page_text is not None:
        page.write_text(page_text)
    print(json.dumps(doc["summary"]["overall"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
