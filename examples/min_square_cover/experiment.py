# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sweep shapes x arms x seeds and summarise square counts vs the greedy baseline.

:func:`run_sweep` solves every ``(shape, arm, seed)`` from the shared greedy warm start (so
only the optimiser differs), optionally writing a JSON log + flat CSV. :func:`summarize` /
:func:`format_table` reduce runs to mean +/- std final square count per ``(shape, arm)`` and
compare against the greedy baseline and the certified area lower bound.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from examples.min_square_cover.arms import ARMS, get_arm
from examples.min_square_cover.data import SHAPES, Instance, make_instance
from examples.min_square_cover.train import CoverResult, solve_cover


def run_sweep(
    shapes: tuple[str, ...] = SHAPES,
    arms: tuple[str, ...] = ARMS,
    seeds: tuple[int, ...] = (0, 1, 2),
    *,
    size: int = 24,
    side: int | None = None,
    steps: int = 150,
    shape_kind: str = "square",
    warm_start: str = "greedy",
    device: str = "cpu",
    out_dir: str | None = None,
    log: bool = False,
) -> list[CoverResult]:
    """Solve all ``(shape, arm, seed)`` combinations and return the results."""
    results: list[CoverResult] = []
    for shape in shapes:
        for seed in seeds:
            instance = make_instance(shape, size=size, side=side, seed=seed)
            for arm_name in arms:
                result = solve_cover(
                    get_arm(arm_name),
                    instance,
                    steps=steps,
                    shape_kind=shape_kind,
                    warm_start=warm_start,
                    seed=seed,
                    device=device,
                    log=log,
                )
                results.append(result)
    if out_dir is not None:
        write_results(results, out_dir)
    return results


def run_sweep_sides(
    shapes: tuple[str, ...] = SHAPES,
    arms: tuple[str, ...] = ARMS,
    seeds: tuple[int, ...] = (0, 1, 2),
    sides: tuple[int, ...] = (5, 7),
    *,
    size: int = 24,
    steps: int = 150,
    shape_kind: str = "square",
    warm_start: str = "greedy",
    device: str = "cpu",
    out_dir: str | None = None,
    log: bool = False,
) -> list[CoverResult]:
    """Run :func:`run_sweep` at each square ``side`` and concatenate (the full shape x side sweep).

    Each :class:`CoverResult` already records its ``side``, so the combined list slices cleanly
    per square size. Results are written once (``results.{json,csv}``) if ``out_dir`` is given.
    """
    results: list[CoverResult] = []
    for side in sides:
        results.extend(
            run_sweep(
                shapes, arms, seeds, size=size, side=side, steps=steps,
                shape_kind=shape_kind, warm_start=warm_start, device=device,
                out_dir=None, log=log,
            )
        )
    if out_dir is not None:
        write_results(results, out_dir)
    return results


def run_instance(
    instance: Instance,
    arms: tuple[str, ...] = ARMS,
    seeds: tuple[int, ...] = (0, 1, 2),
    *,
    steps: int = 150,
    shape_kind: str = "square",
    warm_start: str = "greedy",
    device: str = "cpu",
    out_dir: str | None = None,
    log: bool = False,
) -> list[CoverResult]:
    """Sweep a single, already-built ``instance`` across ``arms x seeds``.

    Unlike :func:`run_sweep` (which generates the synthetic shape family), this takes an
    :class:`~examples.min_square_cover.data.Instance` verbatim -- e.g. one produced by
    :func:`~examples.min_square_cover.data.load_instance` from a real image file -- so the
    same reporting works on user-supplied inputs.
    """
    results: list[CoverResult] = []
    for seed in seeds:
        for arm_name in arms:
            results.append(
                solve_cover(
                    get_arm(arm_name), instance, steps=steps, shape_kind=shape_kind,
                    warm_start=warm_start, seed=seed, device=device, log=log,
                )
            )
    if out_dir is not None:
        write_results(results, out_dir, name="image_results")
    return results


def run_warm_start_variants(
    shapes: tuple[str, ...] = SHAPES,
    arms: tuple[str, ...] = ("adam", "cubic_newton", "trust_region"),
    seeds: tuple[int, ...] = (0, 1, 2),
    warm_starts: tuple[str, ...] = ("greedy", "lp"),
    *,
    size: int = 24,
    side: int | None = None,
    steps: int = 150,
    device: str = "cpu",
    out_dir: str | None = None,
    log: bool = False,
) -> list[CoverResult]:
    """Compare warm starts: the greedy corners vs the LP-relaxation register (omnibias-convex).

    Each :class:`CoverResult` records the warm start actually used (``"lp"`` falls back to
    ``"greedy"`` when ``omnibias-convex`` is unavailable), so the two registers can be compared
    on the same instances.
    """
    results: list[CoverResult] = []
    for warm in warm_starts:
        for shape in shapes:
            for seed in seeds:
                instance = make_instance(shape, size=size, side=side, seed=seed)
                for arm_name in arms:
                    results.append(
                        solve_cover(
                            get_arm(arm_name), instance, steps=steps, warm_start=warm,
                            seed=seed, device=device, log=log,
                        )
                    )
    if out_dir is not None:
        write_results(results, out_dir, name="warm_start_variants")
    return results


def run_shape_variants(
    shapes: tuple[str, ...] = SHAPES,
    arms: tuple[str, ...] = ("adam", "trust_region", "jet_lbfgs"),
    seeds: tuple[int, ...] = (0, 1, 2),
    kinds: tuple[str, ...] = ("square", "disk"),
    *,
    size: int = 24,
    side: int | None = None,
    steps: int = 150,
    device: str = "cpu",
    out_dir: str | None = None,
    log: bool = False,
) -> list[CoverResult]:
    """The P4 shape/curvature variant study: the same solve under each ``shape_kind`` surrogate.

    Runs autodiff-compatible arms (the closed-form arm is box-only) with the box vs disk soft
    occupancy, tagging each :class:`CoverResult` with its ``shape_kind`` so the final square
    counts can be compared across surrogate shapes.
    """
    results: list[CoverResult] = []
    for kind in kinds:
        for shape in shapes:
            for seed in seeds:
                instance = make_instance(shape, size=size, side=side, seed=seed)
                for arm_name in arms:
                    results.append(
                        solve_cover(
                            get_arm(arm_name), instance, steps=steps, shape_kind=kind,
                            seed=seed, device=device, log=log,
                        )
                    )
    if out_dir is not None:
        write_results(results, out_dir, name="shape_variants")
    return results


def run_anneal_variants(
    shapes: tuple[str, ...] = SHAPES,
    arms: tuple[str, ...] = ("adam", "cubic_gauss_newton"),
    seeds: tuple[int, ...] = (0, 1, 2),
    betas: tuple[float, ...] = (1.0, 4.0, 8.0),
    *,
    beta0: float = 1.0,
    beta1: float = 8.0,
    size: int = 24,
    side: int | None = None,
    steps: int = 150,
    device: str = "cpu",
    out_dir: str | None = None,
    log: bool = False,
) -> list[CoverResult]:
    """H3 ablation: the annealed ``beta`` homotopy vs each fixed-``beta`` sharpness.

    Runs the same solve once with ``beta`` annealing ``beta0 -> beta1`` (``beta_schedule ==
    "anneal"``) and once per constant ``beta`` in ``betas`` (``"fixed@<beta>"``). H3 predicts the
    annealed schedule yields the lowest final square count of the group; each
    :class:`CoverResult` is tagged with its ``beta_schedule`` so the two can be compared on the
    same instances.
    """
    schedules: list[float | None] = [None, *betas]
    results: list[CoverResult] = []
    for fixed in schedules:
        for shape in shapes:
            for seed in seeds:
                instance = make_instance(shape, size=size, side=side, seed=seed)
                for arm_name in arms:
                    results.append(
                        solve_cover(
                            get_arm(arm_name), instance, steps=steps, beta0=beta0, beta1=beta1,
                            fixed_beta=fixed, seed=seed, device=device, log=log,
                        )
                    )
    if out_dir is not None:
        write_results(results, out_dir, name="anneal_variants")
    return results


def write_results(results: list[CoverResult], out_dir: str, *, name: str = "results") -> tuple[Path, Path]:
    """Write the per-run JSON log and a flat CSV; return both paths.

    The dense per-step ``history`` (only used in-memory to draw the energy-trajectory figure and
    never read back) is dropped from the JSON so the committed artifact stays small; the placed
    ``squares`` are kept for reproducible cover overlays.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{name}.json"
    csv_path = out / f"{name}.csv"
    records = [{k: v for k, v in asdict(r).items() if k != "history"} for r in results]
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    fields = [
        "shape", "arm", "shape_kind", "warm_start", "beta_schedule", "seed", "side", "n_ones",
        "n_greedy", "n_active", "n_final", "n_completion", "feasible_before_completion",
        "lower_bound", "ratio_final", "ratio_greedy", "init_energy", "final_energy",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            writer.writerow({k: row[k] for k in fields})
    return json_path, csv_path


def write_summary(results: list[CoverResult], out_dir: str, *, name: str = "summary") -> tuple[Path, Path]:
    """Write the aggregated ``(shape, arm)`` summary as JSON + CSV; return both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    rows = [
        {"shape": shape, "arm": arm, **stats}
        for (shape, arm), stats in sorted(summary.items())
    ]
    json_path = out / f"{name}.json"
    csv_path = out / f"{name}.csv"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    fields = ["shape", "arm", "final_mean", "final_std", "greedy_mean", "ratio_mean",
              "lower_bound", "beats_greedy_rate", "n"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
    return json_path, csv_path


def summarize(results: list[CoverResult]) -> dict[tuple[str, str], dict[str, float]]:
    """Aggregate to mean +/- std final square count (and mean ratio) per ``(shape, arm)``."""
    grouped: dict[tuple[str, str], list[CoverResult]] = {}
    for r in results:
        grouped.setdefault((r.shape, r.arm), []).append(r)
    summary: dict[tuple[str, str], dict[str, float]] = {}
    for key, runs in grouped.items():
        finals = [float(r.n_final) for r in runs]
        summary[key] = {
            "final_mean": statistics.fmean(finals),
            "final_std": statistics.pstdev(finals) if len(finals) > 1 else 0.0,
            "greedy_mean": statistics.fmean([float(r.n_greedy) for r in runs]),
            "ratio_mean": statistics.fmean([r.ratio_final for r in runs]),
            "lower_bound": statistics.fmean([float(r.lower_bound) for r in runs]),
            "beats_greedy_rate": statistics.fmean(
                [1.0 if r.n_final < r.n_greedy else 0.0 for r in runs]
            ),
            "n": float(len(runs)),
        }
    return summary


def format_table(results: list[CoverResult]) -> str:
    """Render a ``shape x arm`` grid of mean +/- std final square count (lower is better)."""
    summary = summarize(results)
    shapes = sorted({s for s, _ in summary})
    arms = [a for a in ARMS if any((s, a) in summary for s in shapes)]
    width = max((len(a) for a in arms), default=6) + 2
    header = "shape".ljust(12) + "greedy".rjust(10) + "  lb".rjust(6)
    header += "".join(a.rjust(width + 6) for a in arms)
    lines = [header, "-" * len(header)]
    for shape in shapes:
        any_key = next((s for s in summary if s[0] == shape), None)
        greedy = summary[any_key]["greedy_mean"] if any_key else float("nan")
        lb = summary[any_key]["lower_bound"] if any_key else float("nan")
        row = shape.ljust(12) + f"{greedy:8.1f}".rjust(10) + f"{lb:4.0f}".rjust(6)
        for arm in arms:
            stats = summary.get((shape, arm))
            cell = "n/a" if stats is None else f"{stats['final_mean']:4.1f}+/-{stats['final_std']:3.1f}"
            row += cell.rjust(width + 6)
        lines.append(row)
    lines.append("")
    lines.append("Cells are mean +/- std final square count over seeds (lower is better);")
    lines.append("'greedy' is the heuristic baseline, 'lb' the certified area lower bound.")
    return "\n".join(lines)


def _variant_table(
    results: list[CoverResult],
    key: Callable[[CoverResult], str],
    row_label: str,
    caption: str,
) -> str:
    """Render a ``key x arm`` grid of mean +/- std final square count (lower is better)."""
    grouped: dict[tuple[str, str], list[CoverResult]] = {}
    for r in results:
        grouped.setdefault((key(r), r.arm), []).append(r)
    row_keys = sorted({k for k, _ in grouped})
    arms = [a for a in ARMS if any((k, a) in grouped for k in row_keys)]
    label_w = max(len(row_label), max((len(k) for k in row_keys), default=0)) + 2
    cell_w = max((len(a) for a in arms), default=6) + 8
    header = row_label.ljust(label_w) + "".join(a.rjust(cell_w) for a in arms)
    lines = [header, "-" * len(header)]
    for rk in row_keys:
        row = rk.ljust(label_w)
        for arm in arms:
            runs = grouped.get((rk, arm))
            if not runs:
                cell = "n/a"
            else:
                finals = [float(r.n_final) for r in runs]
                mean = statistics.fmean(finals)
                std = statistics.pstdev(finals) if len(finals) > 1 else 0.0
                cell = f"{mean:4.1f}+/-{std:3.1f}"
            row += cell.rjust(cell_w)
        lines.append(row)
    lines.append("")
    lines.append(caption)
    return "\n".join(lines)


def format_shape_variant_table(results: list[CoverResult]) -> str:
    """Render a ``shape_kind x arm`` grid of mean +/- std final square count (P4 variant study)."""
    return _variant_table(
        results, lambda r: r.shape_kind, "shape_kind",
        "Final square count (lower is better) by soft-occupancy surrogate shape.",
    )


def format_warm_start_table(results: list[CoverResult]) -> str:
    """Render a ``warm_start x arm`` grid of mean +/- std final square count (greedy vs LP)."""
    return _variant_table(
        results, lambda r: r.warm_start, "warm_start",
        "Final square count (lower is better) by warm start (greedy corners vs LP register).",
    )


def format_anneal_table(results: list[CoverResult]) -> str:
    """Render a ``beta_schedule x arm`` grid of mean +/- std final square count (H3 ablation)."""
    return _variant_table(
        results, lambda r: r.beta_schedule, "beta_sched",
        "Final square count (lower is better) by beta schedule (annealed vs fixed sharpness).",
    )


__all__ = [
    "format_anneal_table",
    "format_shape_variant_table",
    "format_table",
    "format_warm_start_table",
    "run_anneal_variants",
    "run_instance",
    "run_shape_variants",
    "run_sweep",
    "run_sweep_sides",
    "run_warm_start_variants",
    "summarize",
    "write_results",
    "write_summary",
]
