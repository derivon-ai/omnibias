# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""CLI for the min-square-cover study.

Examples::

    # Offline smoke run (small images, few steps, CPU):
    python -m examples.min_square_cover.run_demo --shapes blob scatter --steps 40 --seeds 0

    # Full sweep with the certified LP lower bound:
    python -m examples.min_square_cover.run_demo --seeds 0 1 2 --steps 200 --certify

The table reports the mean final square count per (shape, arm); lower is better. The
certified block prints, for one instance, the exact feasibility, the area / LP lower bounds,
and the robustness margin -- the ``ceil(lower_bound) <= optimum <= K`` sandwich.
"""

from __future__ import annotations

import argparse

from examples.min_square_cover.arms import ARMS, get_arm
from examples.min_square_cover.certify import certify_cover, lp_rounded_cover
from examples.min_square_cover.coverage import SHAPE_KINDS
from examples.min_square_cover.data import SHAPES, Instance, load_instance, make_instance
from examples.min_square_cover.experiment import (
    format_anneal_table,
    format_shape_variant_table,
    format_table,
    format_warm_start_table,
    run_anneal_variants,
    run_instance,
    run_shape_variants,
    run_sweep,
    run_sweep_sides,
    run_warm_start_variants,
    write_summary,
)
from examples.min_square_cover.train import solve_cover
from examples.min_square_cover.verify_cert import certify_cover_robustness


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Min-square-cover as geometric optimisation.")
    p.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--size", type=int, default=24, help="image side length (pixels)")
    p.add_argument("--side", type=int, default=None, help="square side (default: per-shape)")
    p.add_argument(
        "--sides", nargs="+", type=int, default=None,
        help="run the full shape x side sweep over these square sides (overrides --side)",
    )
    p.add_argument("--steps", type=int, default=150)
    p.add_argument(
        "--shape-kind", default="square", choices=list(SHAPE_KINDS),
        help="soft-occupancy surrogate shape used in the continuous solve",
    )
    p.add_argument(
        "--warm-start", default="greedy", choices=["greedy", "lp"],
        help="initial centers/gates: greedy corners or the LP-relaxation register",
    )
    p.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    p.add_argument(
        "--image", default=None,
        help="solve a real image file instead of the synthetic shapes (needs Pillow/scikit-image)",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="grayscale foreground threshold for --image (pixels >= threshold are 1-pixels)",
    )
    p.add_argument(
        "--invert", action="store_true",
        help="treat dark pixels as foreground for --image (light background)",
    )
    p.add_argument(
        "--max-size", type=int, default=None,
        help="downscale the longest side of --image to at most this many pixels",
    )
    p.add_argument("--certify", action="store_true", help="also print an LP-bounded certificate")
    p.add_argument(
        "--figures", action="store_true",
        help="write results/figures/*.png (cover counts, energy trajectory, gap, overlay)",
    )
    p.add_argument(
        "--shape-variants", action="store_true",
        help="run the P4 square-vs-disk surrogate study instead of the main sweep",
    )
    p.add_argument(
        "--warm-start-variants", action="store_true",
        help="run the greedy-vs-LP warm-start study instead of the main sweep",
    )
    p.add_argument(
        "--anneal-variants", action="store_true",
        help="run the H3 annealed-vs-fixed-beta study instead of the main sweep",
    )
    p.add_argument("--out-dir", default="examples/min_square_cover/results")
    return p.parse_args(argv)


def _certify_header() -> list[str]:
    lines = ["Certified sandwich (ceil(lower_bound) <= optimum <= K), arm=cubic_newton:"]
    header = f"{'shape':<12}{'feasible':>9}{'area_lb':>9}{'lp_lb':>8}{'K':>4}{'K_lp':>6}{'ratio':>7}"
    header += f"{'robust':>8}{'cov@0':>8}{'rob@.25':>9}"
    lines.append(header)
    return lines


def _certify_row(instance: Instance, steps: int) -> str:
    """Solve one instance with the best second-order arm and certify it as a table row.

    Reports both registers side by side: the continuous solve's cover ``K``, the LP register's
    certified lower bound ``lp_lb`` and its own rounded cover ``K_lp`` (``ceil(lp_lb) == K_lp``
    proves optimality), and the ``omnibias-verify`` continuous coverage certificate (``cov@0`` =
    rigorous inf soft-OR coverage at the pixels; ``rob@.25`` = is that coverage provably above 0.5
    over every +/-0.25px cell).
    """
    arm = get_arm("cubic_newton")
    result = solve_cover(arm, instance, steps=steps, seed=0)
    cert = certify_cover(instance.image, result.squares, instance.side, with_lp=True)
    lp = f"{cert.lp_lower_bound:.2f}" if cert.lp_lower_bound is not None else "n/a"
    lp_sq = lp_rounded_cover(instance.image, instance.side)
    klp = str(len(lp_sq)) if lp_sq is not None else "n/a"
    rob = certify_cover_robustness(
        instance.image, result.squares, instance.side, beta=4.0, threshold=0.5, delta=0.25
    )
    cov0 = "n/a"
    robust = "n/a"
    if rob is not None:
        rob0 = certify_cover_robustness(
            instance.image, result.squares, instance.side, beta=4.0, threshold=0.5, delta=0.0
        )
        cov0 = f"{rob0.certified_min_coverage:.3f}" if rob0 is not None else "n/a"
        robust = str(rob.robust)
    return (
        f"{instance.name[:11]:<12}{str(cert.feasible):>9}{cert.area_lower_bound:>9}{lp:>8}"
        f"{cert.n_used:>4}{klp:>6}{cert.optimality_ratio:>7.2f}{cert.robustness_margin:>8}"
        f"{cov0:>8}{robust:>9}"
    )


def _certify_block(shapes: tuple[str, ...], size: int, side: int | None, steps: int) -> str:
    """Solve one instance per shape with the best second-order arm and certify each."""
    lines = _certify_header()
    for shape in shapes:
        instance = make_instance(shape, size=size, side=side, seed=0)
        lines.append(_certify_row(instance, steps))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.image is not None:
        if args.side is None:
            raise SystemExit("--image requires --side (no per-shape default for a real image)")
        instance = load_instance(
            args.image, args.side, threshold=args.threshold,
            invert=args.invert, max_size=args.max_size,
        )
        print(
            f"Min-square-cover on image {args.image!r}: name={instance.name} "
            f"shape={instance.shape} n_ones={instance.n_ones} side={instance.side} "
            f"arms={args.arms} seeds={args.seeds} steps={args.steps} "
            f"shape_kind={args.shape_kind} warm_start={args.warm_start} device={args.device}"
        )
        results = run_instance(
            instance,
            arms=tuple(args.arms),
            seeds=tuple(args.seeds),
            steps=args.steps,
            shape_kind=args.shape_kind,
            warm_start=args.warm_start,
            device=args.device,
            out_dir=args.out_dir,
            log=False,
        )
        print("\n" + format_table(results))
        if args.certify:
            print("\n" + "\n".join([*_certify_header(), _certify_row(instance, args.steps)]))
        if args.figures:
            from examples.min_square_cover.plots import (
                plot_certified_gap,
                plot_cover_counts,
                plot_cover_overlay,
                plot_energy_trajectory,
            )

            fdir = f"{args.out_dir}/figures"
            plot_cover_counts(results, f"{fdir}/cover_counts.png")
            plot_certified_gap(results, f"{fdir}/certified_gap.png")
            if any(r.history for r in results):
                plot_energy_trajectory(results, f"{fdir}/energy_trajectory.png")
            best = min(results, key=lambda r: r.n_final)
            plot_cover_overlay(instance, best.squares, instance.side, f"{fdir}/cover_overlay.png")
            print(f"\nWrote figures to {fdir}/")
        print(f"\nWrote results to {args.out_dir}/image_results.json and image_results.csv")
        return
    if args.shape_variants:
        print(
            f"Min-square-cover shape-variant study: shapes={args.shapes} "
            f"kinds={list(SHAPE_KINDS)} seeds={args.seeds} size={args.size} "
            f"side={args.side} steps={args.steps} device={args.device}"
        )
        variant_arms = tuple(a for a in args.arms if a != "closed_form_newton")
        results = run_shape_variants(
            shapes=tuple(args.shapes),
            arms=variant_arms,
            seeds=tuple(args.seeds),
            kinds=SHAPE_KINDS,
            size=args.size,
            side=args.side,
            steps=args.steps,
            device=args.device,
            out_dir=args.out_dir,
            log=False,
        )
        print("\n" + format_shape_variant_table(results))
        print(f"\nWrote results to {args.out_dir}/shape_variants.json and shape_variants.csv")
        return
    if args.warm_start_variants:
        print(
            f"Min-square-cover warm-start study: shapes={args.shapes} "
            f"warm_starts=['greedy', 'lp'] seeds={args.seeds} size={args.size} "
            f"side={args.side} steps={args.steps} device={args.device}"
        )
        results = run_warm_start_variants(
            shapes=tuple(args.shapes),
            arms=tuple(args.arms),
            seeds=tuple(args.seeds),
            warm_starts=("greedy", "lp"),
            size=args.size,
            side=args.side,
            steps=args.steps,
            device=args.device,
            out_dir=args.out_dir,
            log=False,
        )
        print("\n" + format_warm_start_table(results))
        print(f"\nWrote results to {args.out_dir}/warm_start_variants.json and .csv")
        return
    if args.anneal_variants:
        print(
            f"Min-square-cover annealing (H3) study: shapes={args.shapes} "
            f"beta_schedules=['anneal', 'fixed@1', 'fixed@4', 'fixed@8'] seeds={args.seeds} "
            f"size={args.size} side={args.side} steps={args.steps} device={args.device}"
        )
        anneal_arms = tuple(a for a in args.arms if a != "closed_form_newton")
        results = run_anneal_variants(
            shapes=tuple(args.shapes),
            arms=anneal_arms,
            seeds=tuple(args.seeds),
            betas=(1.0, 4.0, 8.0),
            size=args.size,
            side=args.side,
            steps=args.steps,
            device=args.device,
            out_dir=args.out_dir,
            log=False,
        )
        print("\n" + format_anneal_table(results))
        if args.figures:
            from examples.min_square_cover.plots import plot_anneal_ablation

            fig = plot_anneal_ablation(results, f"{args.out_dir}/figures/anneal_ablation.png")
            print(f"\nWrote figure to {fig}")
        print(f"\nWrote results to {args.out_dir}/anneal_variants.json and .csv")
        return
    sides_desc = args.sides if args.sides is not None else args.side
    print(
        f"Min-square-cover sweep: shapes={args.shapes} arms={args.arms} "
        f"seeds={args.seeds} size={args.size} side(s)={sides_desc} steps={args.steps} "
        f"shape_kind={args.shape_kind} warm_start={args.warm_start} device={args.device}"
    )
    if args.sides is not None:
        results = run_sweep_sides(
            shapes=tuple(args.shapes),
            arms=tuple(args.arms),
            seeds=tuple(args.seeds),
            sides=tuple(args.sides),
            size=args.size,
            steps=args.steps,
            shape_kind=args.shape_kind,
            warm_start=args.warm_start,
            device=args.device,
            out_dir=args.out_dir,
            log=False,
        )
    else:
        results = run_sweep(
            shapes=tuple(args.shapes),
            arms=tuple(args.arms),
            seeds=tuple(args.seeds),
            size=args.size,
            side=args.side,
            steps=args.steps,
            shape_kind=args.shape_kind,
            warm_start=args.warm_start,
            device=args.device,
            out_dir=args.out_dir,
            log=False,
        )
    write_summary(results, args.out_dir)
    print("\n" + format_table(results))
    if args.certify:
        print("\n" + _certify_block(tuple(args.shapes), args.size, args.side, args.steps))
    if args.figures:
        from examples.min_square_cover.plots import make_all_figures

        paths = make_all_figures(results, f"{args.out_dir}/figures", size=args.size)
        print("\nWrote figures: " + ", ".join(p.name for p in paths))
    print(
        f"\nWrote results to {args.out_dir}/results.json and results.csv "
        f"(+ summary.json / summary.csv)"
    )


if __name__ == "__main__":
    main()
