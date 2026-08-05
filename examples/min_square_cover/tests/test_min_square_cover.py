# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Offline CPU smoke tests for the min-square-cover example (small images, few steps)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omnibias.shape.torch.ops")

from examples.min_square_cover.arms import ARMS, get_arm  # noqa: E402
from examples.min_square_cover.certify import (  # noqa: E402
    area_lower_bound,
    certify_cover,
    lp_fractional_cover,
    lp_lower_bound,
    lp_rounded_cover,
    verify_cover,
)
from examples.min_square_cover.coverage import SHAPE_KINDS, complete_and_prune  # noqa: E402
from examples.min_square_cover.data import (  # noqa: E402
    SHAPES,
    coverage_fraction,
    greedy_cover,
    is_feasible,
    load_binary_image,
    load_instance,
    make_instance,
)
from examples.min_square_cover.experiment import (  # noqa: E402
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
    summarize,
    write_results,
    write_summary,
)
from examples.min_square_cover.shapes import lp_init_centers  # noqa: E402
from examples.min_square_cover.train import CoverResult, solve_cover  # noqa: E402

_SOLVE_ARMS = ("adam", "cubic_newton", "cubic_gauss_newton")


@pytest.mark.parametrize("shape", SHAPES)
def test_images_are_binary_and_greedy_is_feasible(shape: str):
    inst = make_instance(shape, size=14, seed=0)
    vals = set(inst.image.unique().tolist())
    assert vals <= {0.0, 1.0}
    assert inst.n_ones > 0
    squares = greedy_cover(inst.image, inst.side)
    assert is_feasible(inst.image, squares, inst.side)
    assert coverage_fraction(inst.image, squares, inst.side) == 1.0


@pytest.mark.parametrize("arm_name", _SOLVE_ARMS)
def test_solve_produces_feasible_irredundant_cover(arm_name: str):
    inst = make_instance("blob", size=12, side=5, seed=0)
    result = solve_cover(get_arm(arm_name), inst, steps=25, seed=0, k_slack=2)
    # Feasible and consistent with the reported squares.
    assert verify_cover(inst.image, result.squares, inst.side)
    assert result.n_final == len(result.squares)
    assert coverage_fraction(inst.image, result.squares, inst.side) == 1.0
    # Certified sandwich: lower bound <= produced count.
    assert result.lower_bound <= result.n_final
    assert result.n_final <= result.n_greedy + 2
    # Irredundant: removing any square breaks feasibility.
    for sq in result.squares:
        rest = [s for s in result.squares if s != sq]
        assert not is_feasible(inst.image, rest, inst.side)


def test_certificate_sandwich_and_robustness():
    inst = make_instance("l_shape", size=14, seed=0)
    squares = greedy_cover(inst.image, inst.side)
    cert = certify_cover(inst.image, squares, inst.side, with_lp=False)
    assert cert.feasible
    assert cert.area_lower_bound <= cert.n_used
    assert cert.area_lower_bound == area_lower_bound(inst.image, inst.side)
    assert cert.robustness_margin >= 0
    assert cert.optimality_ratio >= 1.0


def test_lp_lower_bound_when_convex_available():
    inst = make_instance("scatter", size=12, side=5, seed=0)
    lp = lp_lower_bound(inst.image, inst.side)
    if lp is None:
        pytest.skip("omnibias-convex not installed")
    greedy = len(greedy_cover(inst.image, inst.side))
    # LP relaxation optimum lower-bounds the integer optimum, which the greedy count upper-bounds.
    assert 0.0 < lp <= greedy + 1e-6
    assert lp >= area_lower_bound(inst.image, inst.side) - 1e-6


def test_arms_registry():
    assert "adam" in ARMS and "cubic_newton" in ARMS
    assert get_arm("cubic_newton").kind == "scalar"
    assert get_arm("cubic_gauss_newton").kind == "residual"
    assert get_arm("adam").kind == "first_order"
    # second-order menu additions
    assert get_arm("jet_lbfgs").kind == "scalar"
    assert get_arm("jet_subspace").kind == "scalar"
    assert get_arm("diagonal_curvature").kind == "residual"
    assert get_arm("gauss_newton").kind == "functional_gn"
    assert get_arm("closed_form_newton").kind == "closed_form"
    with pytest.raises(ValueError):
        get_arm("nope")


@pytest.mark.parametrize(
    "arm_name",
    ["jet_lbfgs", "jet_subspace", "diagonal_curvature", "gauss_newton", "closed_form_newton"],
)
def test_new_arms_produce_feasible_cover(arm_name: str):
    inst = make_instance("blob", size=12, side=5, seed=0)
    result = solve_cover(get_arm(arm_name), inst, steps=15, seed=0, k_slack=2)
    assert verify_cover(inst.image, result.squares, inst.side)
    assert result.n_final == len(result.squares)
    assert result.lower_bound <= result.n_final


def test_closed_form_hessian_matches_matrix_free_hvp():
    """H6: the closed-form dense coverage Hessian equals the matrix-free autodiff HVP."""
    from omnibias.shape.torch import ops as shape

    from examples.min_square_cover.coverage import scalar_energy
    from examples.min_square_cover.shapes import grid_axes

    torch.set_default_dtype(torch.float64)
    inst = make_instance("blob", size=12, side=5, seed=0)
    rows, cols = grid_axes(inst.shape)
    axes = (rows.double(), cols.double())
    k, side, beta = 4, 5.0, 8.0
    torch.manual_seed(0)
    centers = torch.rand(k, 2, dtype=torch.float64) * 11.0
    gate_logits = torch.randn(k, dtype=torch.float64)
    gates = torch.sigmoid(gate_logits)
    image = inst.image.double()

    h = shape.coverage_energy_hessian(
        axes, centers, side, beta, gates, image, loss="softplus", kappa=4.0, lam=0.1, wrt="all"
    )
    params = torch.cat([centers.reshape(-1), gate_logits]).clone().requires_grad_(True)

    def energy(p: torch.Tensor) -> torch.Tensor:
        c = p[: 2 * k].reshape(k, 2)
        gl = p[2 * k :]
        return scalar_energy(axes, c, gl, image, side, beta, loss="softplus", kappa=4.0, lam=0.1)

    v = torch.randn(3 * k, dtype=torch.float64)
    hvp = torch.autograd.functional.hvp(energy, params, v)[1]
    assert torch.allclose(h @ v, hvp, atol=1e-7)


@pytest.mark.parametrize("shape_kind", SHAPE_KINDS)
@pytest.mark.parametrize("arm_name", ["adam", "cubic_newton", "cubic_gauss_newton"])
def test_shape_kind_produces_feasible_cover(shape_kind: str, arm_name: str):
    """Both the box and disk soft-occupancy surrogates yield a feasible, irredundant cover."""
    inst = make_instance("blob", size=12, side=5, seed=0)
    result = solve_cover(get_arm(arm_name), inst, steps=20, shape_kind=shape_kind, seed=0, k_slack=2)
    assert result.shape_kind == shape_kind
    assert verify_cover(inst.image, result.squares, inst.side)
    assert result.n_final == len(result.squares)
    assert result.lower_bound <= result.n_final
    for sq in result.squares:
        rest = [s for s in result.squares if s != sq]
        assert not is_feasible(inst.image, rest, inst.side)


def test_closed_form_arm_requires_square_shape_kind():
    """The closed_form arm uses the box-specific closed-form Hessian; disk must be rejected."""
    inst = make_instance("blob", size=12, side=5, seed=0)
    with pytest.raises(ValueError, match="shape_kind must be 'square'"):
        solve_cover(get_arm("closed_form_newton"), inst, steps=5, shape_kind="disk", seed=0)


def test_invalid_shape_kind_raises():
    inst = make_instance("blob", size=12, side=5, seed=0)
    with pytest.raises(ValueError, match="shape_kind must be one of"):
        solve_cover(get_arm("adam"), inst, steps=3, shape_kind="triangle", seed=0)


def test_run_shape_variants_and_table(tmp_path):
    results = run_shape_variants(
        shapes=("blob",),
        arms=("adam", "cubic_gauss_newton"),
        seeds=(0,),
        kinds=("square", "disk"),
        size=12,
        side=5,
        steps=15,
        out_dir=str(tmp_path),
    )
    assert len(results) == 4  # 2 kinds x 1 shape x 1 seed x 2 arms
    assert {r.shape_kind for r in results} == {"square", "disk"}
    assert all(verify_cover(make_instance("blob", size=12, side=5).image, r.squares, 5) for r in results)
    table = format_shape_variant_table(results)
    assert "square" in table and "disk" in table and "adam" in table
    assert (tmp_path / "shape_variants.json").exists()
    assert (tmp_path / "shape_variants.csv").exists()


def test_lp_fractional_cover_structure():
    inst = make_instance("blob", size=14, side=5, seed=0)
    lp = lp_fractional_cover(inst.image, inst.side)
    if lp is None:
        pytest.skip("omnibias-convex not installed")
    assert len(lp.positions) == len(lp.weights)
    assert all(0.0 <= w <= 1.0 + 1e-9 for w in lp.weights)
    # objective is the sum of the fractional weights and equals the scalar lower-bound wrapper.
    assert abs(lp.objective - sum(lp.weights)) < 1e-3
    assert abs(lp.objective - lp_lower_bound(inst.image, inst.side)) < 1e-9


def test_complete_and_prune_makes_feasible_irredundant_cover():
    inst = make_instance("blob", size=12, side=5, seed=0)
    # Start from a single (insufficient) square: completion must add squares to reach feasibility.
    squares, n_completion, feasible_before = complete_and_prune(inst.image, inst.side, [(0, 0)])
    assert not feasible_before
    assert n_completion >= 1
    assert verify_cover(inst.image, squares, inst.side)
    # Irredundant: dropping any square must break feasibility.
    for sq in squares:
        assert not verify_cover(inst.image, [s for s in squares if s != sq], inst.side)


@pytest.mark.parametrize("shape", SHAPES)
def test_lp_rounded_cover_is_feasible_and_at_most_greedy(shape: str):
    inst = make_instance(shape, size=14, side=5, seed=0)
    squares = lp_rounded_cover(inst.image, inst.side)
    if squares is None:
        pytest.skip("omnibias-convex not installed")
    assert verify_cover(inst.image, squares, inst.side)
    # The LP register's rounded cover never needs more squares than the greedy baseline.
    assert len(squares) <= len(greedy_cover(inst.image, inst.side))


def test_lp_rounded_cover_matches_lower_bound_proves_optimality():
    # On blob the LP relaxation is integral: the rounded cover hits ceil(lp_lb) == area_lb,
    # so the two registers *prove* the discrete optimum (K_lp == certified lower bound).
    inst = make_instance("blob", size=16, side=5, seed=0)
    squares = lp_rounded_cover(inst.image, inst.side)
    lb = lp_lower_bound(inst.image, inst.side)
    if squares is None or lb is None:
        pytest.skip("omnibias-convex not installed")
    import math

    assert len(squares) == math.ceil(lb - 1e-6) == area_lower_bound(inst.image, inst.side)
    assert verify_cover(inst.image, squares, inst.side)


def test_lp_init_centers_prioritises_high_weight_positions():
    inst = make_instance("blob", size=14, side=5, seed=0)
    lp = lp_fractional_cover(inst.image, inst.side)
    if lp is None:
        pytest.skip("omnibias-convex not installed")
    k = 6
    centers, gate_logits = lp_init_centers(inst, lp.positions, lp.weights, k, init_gate=2.0, seed=0)
    assert centers.shape == (k, 2)
    assert gate_logits.shape == (k,)
    # Gates start on (the count penalty prunes later); centers are the top-weight positions.
    assert torch.allclose(gate_logits, torch.full((k,), 2.0))
    top = max(range(len(lp.weights)), key=lambda i: lp.weights[i])
    r, c = lp.positions[top]
    assert float(centers[0, 0]) == r + inst.side / 2.0
    assert float(centers[0, 1]) == c + inst.side / 2.0


@pytest.mark.parametrize("arm_name", ["adam", "cubic_newton"])
def test_lp_warm_start_produces_feasible_cover(arm_name: str):
    inst = make_instance("blob", size=14, side=5, seed=0)
    result = solve_cover(get_arm(arm_name), inst, steps=25, warm_start="lp", seed=0)
    # omnibias-convex is a hard dep of this example, so the LP warm start is actually used.
    assert result.warm_start == "lp"
    assert verify_cover(inst.image, result.squares, inst.side)
    assert result.n_final == len(result.squares)
    assert result.lower_bound <= result.n_final


def test_invalid_warm_start_raises():
    inst = make_instance("blob", size=12, side=5, seed=0)
    with pytest.raises(ValueError, match="warm_start must be"):
        solve_cover(get_arm("adam"), inst, steps=3, warm_start="magic", seed=0)


def test_lp_warm_start_falls_back_to_greedy(monkeypatch):
    """When the LP register is unavailable, warm_start='lp' degrades to greedy (recorded honestly)."""
    import examples.min_square_cover.certify as certify_mod

    # solve_cover imports lp_fractional_cover lazily from certify, so patch it at the source.
    monkeypatch.setattr(certify_mod, "lp_fractional_cover", lambda *a, **k: None)
    inst = make_instance("blob", size=12, side=5, seed=0)
    result = solve_cover(get_arm("cubic_newton"), inst, steps=15, warm_start="lp", seed=0)
    assert result.warm_start == "greedy"
    assert verify_cover(inst.image, result.squares, inst.side)


def test_run_warm_start_variants_and_table(tmp_path):
    results = run_warm_start_variants(
        shapes=("blob",),
        arms=("adam", "cubic_newton"),
        seeds=(0,),
        warm_starts=("greedy", "lp"),
        size=14,
        side=5,
        steps=15,
        out_dir=str(tmp_path),
    )
    assert len(results) == 4  # 2 warm starts x 1 shape x 1 seed x 2 arms
    assert {r.warm_start for r in results} == {"greedy", "lp"}
    assert all(verify_cover(make_instance("blob", size=14, side=5).image, r.squares, 5) for r in results)
    table = format_warm_start_table(results)
    assert "greedy" in table and "lp" in table and "adam" in table
    assert (tmp_path / "warm_start_variants.json").exists()
    assert (tmp_path / "warm_start_variants.csv").exists()


def test_solve_records_beta_schedule():
    inst = make_instance("blob", size=12, side=5, seed=0)
    annealed = solve_cover(get_arm("adam"), inst, steps=8, seed=0)
    assert annealed.beta_schedule == "anneal"
    fixed = solve_cover(get_arm("adam"), inst, steps=8, fixed_beta=4.0, seed=0)
    assert fixed.beta_schedule == "fixed@4"
    # A fixed-sharpness solve still rounds+completes to a feasible cover.
    assert verify_cover(inst.image, fixed.squares, inst.side)
    assert fixed.n_final == len(fixed.squares)


def test_run_anneal_variants_and_table(tmp_path):
    results = run_anneal_variants(
        shapes=("blob",),
        arms=("adam", "cubic_gauss_newton"),
        seeds=(0,),
        betas=(1.0, 8.0),
        size=12,
        side=5,
        steps=15,
        out_dir=str(tmp_path),
    )
    # (1 annealed + 2 fixed) x 1 shape x 1 seed x 2 arms
    assert len(results) == 6
    assert {r.beta_schedule for r in results} == {"anneal", "fixed@1", "fixed@8"}
    assert all(
        verify_cover(make_instance("blob", size=12, side=5).image, r.squares, 5) for r in results
    )
    table = format_anneal_table(results)
    assert "anneal" in table and "fixed@1" in table and "adam" in table
    assert (tmp_path / "anneal_variants.json").exists()
    csv_path = tmp_path / "anneal_variants.csv"
    assert csv_path.exists()
    assert "beta_schedule" in csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")


def test_verify_cert_interval_encloses_float_coverage():
    """Soundness: the interval soft-OR coverage encloses the float64 coverage at every point."""
    pytest.importorskip("omnibias.core.verified.transcend")
    from omnibias.core.verified.interval import Interval
    from omnibias.shape.torch import ops as shape

    from examples.min_square_cover.verify_cert import _coverage_interval_fn

    torch.set_default_dtype(torch.float64)
    side, beta = 5.0, 4.0
    centers = [(3.2, 4.7), (8.1, 2.9), (6.0, 9.3)]
    rows, cols = torch.arange(12.0), torch.arange(12.0)
    occ = shape.soft_box((rows, cols), torch.tensor(centers), side, beta)
    c_float, _ = shape.soft_or_coverage(occ, torch.ones(len(centers)))
    cov = _coverage_interval_fn(centers, side, beta)
    for i in range(12):
        for j in range(12):
            iv = cov((Interval.point(float(i)), Interval.point(float(j))))
            assert iv.lo <= float(c_float[i, j]) <= iv.hi


def test_certify_cover_robustness_valid_cover():
    verify = pytest.importorskip("omnibias.verify")  # noqa: F841
    from examples.min_square_cover.verify_cert import certify_cover_robustness

    inst = make_instance("blob", size=10, side=4, seed=0)
    squares = greedy_cover(inst.image, inst.side)
    cert = certify_cover_robustness(inst.image, squares, inst.side, beta=4.0, threshold=0.5, delta=0.0)
    assert cert is not None
    assert cert.robust  # a feasible cover certifiably exceeds the coverage threshold
    assert cert.certified_min_coverage >= cert.threshold
    assert cert.all_converged
    assert cert.n_pixels == int(inst.image.sum())


def test_certify_cover_robustness_detects_hole():
    verify = pytest.importorskip("omnibias.verify")  # noqa: F841
    from examples.min_square_cover.verify_cert import certify_cover_robustness

    inst = make_instance("blob", size=10, side=4, seed=0)
    squares = greedy_cover(inst.image, inst.side)
    # Dropping a square opens a hole; the soft-OR coverage there is provably ~0 < threshold.
    cert = certify_cover_robustness(
        inst.image, squares[:-1], inst.side, beta=4.0, threshold=0.5, delta=0.0
    )
    assert cert is not None
    assert not cert.robust
    assert cert.certified_min_coverage < cert.threshold


def test_certify_cover_robustness_delta_shrinks_coverage():
    verify = pytest.importorskip("omnibias.verify")  # noqa: F841
    from examples.min_square_cover.verify_cert import certify_cover_robustness

    inst = make_instance("blob", size=10, side=4, seed=0)
    squares = greedy_cover(inst.image, inst.side)
    c0 = certify_cover_robustness(inst.image, squares, inst.side, beta=4.0, delta=0.0)
    c1 = certify_cover_robustness(inst.image, squares, inst.side, beta=4.0, delta=0.3)
    assert c0 is not None and c1 is not None
    # Enlarging the cell can only lower the rigorous worst-case coverage.
    assert c1.certified_min_coverage <= c0.certified_min_coverage + 1e-9


def _write_png(tmp_path, arr):
    """Save a uint8 HxW array as a grayscale PNG and return the path (skips without Pillow)."""
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "img.png"
    Image.fromarray(arr).save(path)
    return str(path)


def test_load_binary_image_white_foreground(tmp_path):
    import numpy as np

    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[2:6, 3:7] = 255  # white block -> 1-pixels to cover
    img = load_binary_image(_write_png(tmp_path, arr))
    assert tuple(img.shape) == (10, 10)
    assert set(img.unique().tolist()) <= {0.0, 1.0}
    assert bool((img[2:6, 3:7] == 1.0).all())
    assert float(img.sum()) == 16.0


def test_load_binary_image_invert_dark_foreground(tmp_path):
    import numpy as np

    arr = np.full((10, 10), 255, dtype=np.uint8)
    arr[2:6, 3:7] = 0  # dark block on light background
    img = load_binary_image(_write_png(tmp_path, arr), invert=True)
    assert bool((img[2:6, 3:7] == 1.0).all())
    assert float(img.sum()) == 16.0


def test_load_binary_image_threshold_controls_foreground(tmp_path):
    import numpy as np

    arr = np.full((8, 8), 100, dtype=np.uint8)  # uniform mid-gray ~0.392
    path = _write_png(tmp_path, arr)
    # threshold below the gray level -> everything foreground; above -> nothing.
    assert float(load_binary_image(path, threshold=0.3).sum()) == 64.0
    assert float(load_binary_image(path, threshold=0.5).sum()) == 0.0


def test_load_binary_image_max_size_downscales(tmp_path):
    import numpy as np

    arr = np.zeros((40, 40), dtype=np.uint8)
    arr[10:30, 10:30] = 255
    img = load_binary_image(_write_png(tmp_path, arr), max_size=20)
    assert max(img.shape) == 20
    assert set(img.unique().tolist()) <= {0.0, 1.0}


def test_load_instance_wraps_binary_image(tmp_path):
    import numpy as np

    arr = np.zeros((12, 12), dtype=np.uint8)
    arr[3:9, 3:9] = 255
    path = _write_png(tmp_path, arr)
    inst = load_instance(path, side=4)
    assert inst.name == "img"  # falls back to the file stem
    assert inst.side == 4
    assert inst.shape == (12, 12)
    assert set(inst.image.unique().tolist()) <= {0.0, 1.0}
    # A real loaded instance is solvable end-to-end by the greedy baseline.
    squares = greedy_cover(inst.image, inst.side)
    assert is_feasible(inst.image, squares, inst.side)
    # Explicit name overrides the stem.
    assert load_instance(path, side=4, name="logo").name == "logo"


def test_run_instance_on_loaded_image(tmp_path):
    import numpy as np

    arr = np.zeros((14, 14), dtype=np.uint8)
    arr[2:9, 2:11] = 255
    inst = load_instance(_write_png(tmp_path, arr), side=5)
    results = run_instance(
        inst, arms=("adam", "cubic_newton"), seeds=(0,), steps=15, out_dir=str(tmp_path)
    )
    assert len(results) == 2  # 1 seed x 2 arms
    assert all(r.shape == "img" for r in results)  # labelled by the file stem
    assert all(verify_cover(inst.image, r.squares, inst.side) for r in results)
    assert (tmp_path / "image_results.json").exists()
    assert (tmp_path / "image_results.csv").exists()


def _fake_result(arm: str, shape: str, n_final: int, seed: int) -> CoverResult:
    inst = make_instance(shape, size=16, side=5, seed=seed)
    squares = greedy_cover(inst.image, inst.side)[:n_final]
    history = [
        {"step": float(s), "beta": 1.0 + s, "lam": 0.01 * s, "energy": 100.0 / (s + 1)}
        for s in range(5)
    ]
    return CoverResult(
        arm=arm, shape=shape, shape_kind="square", warm_start="greedy", beta_schedule="anneal",
        seed=seed, side=5, n_ones=inst.n_ones, n_greedy=len(greedy_cover(inst.image, inst.side)),
        n_active=n_final, n_final=n_final, n_completion=0, feasible_before_completion=True,
        lower_bound=2, ratio_final=n_final / 2.0, ratio_greedy=3.0, init_energy=100.0,
        final_energy=20.0, steps=5, squares=squares, history=history,
    )


def test_plots_write_all_figures(tmp_path):
    pytest.importorskip("matplotlib")
    from examples.min_square_cover.plots import make_all_figures

    results = [
        _fake_result(arm, shape, n, seed)
        for shape in ("blob", "l_shape")
        for arm, n in (("adam", 6), ("cubic_newton", 5))
        for seed in (0, 1)
    ]
    paths = make_all_figures(results, tmp_path, size=16)
    names = {p.name for p in paths}
    assert {"cover_counts.png", "certified_gap.png", "energy_trajectory.png", "cover_overlay.png"} <= names
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_plot_anneal_ablation_writes_figure(tmp_path):
    pytest.importorskip("matplotlib")
    import dataclasses

    from examples.min_square_cover.plots import plot_anneal_ablation

    base = [
        _fake_result(arm, "blob", n, seed)
        for arm in ("adam", "cubic_gauss_newton")
        for n, seed in ((6, 0), (7, 1))
    ]
    results = [
        dataclasses.replace(r, beta_schedule=sched)
        for sched in ("anneal", "fixed@1", "fixed@8")
        for r in base
    ]
    out = plot_anneal_ablation(results, tmp_path / "anneal_ablation.png")
    assert out.exists() and out.stat().st_size > 0


def test_run_sweep_sides_and_summary(tmp_path):
    results = run_sweep_sides(
        shapes=("blob",), arms=("adam", "cubic_newton"), seeds=(0,),
        sides=(4, 6), size=14, steps=12, out_dir=str(tmp_path),
    )
    assert len(results) == 4  # 2 sides x 1 shape x 1 seed x 2 arms
    assert {r.side for r in results} == {4, 6}
    assert all(
        verify_cover(make_instance("blob", size=14, side=r.side).image, r.squares, r.side)
        for r in results
    )
    assert (tmp_path / "results.json").exists()
    jp, cp = write_summary(results, str(tmp_path))
    assert jp.exists() and cp.exists()
    header = cp.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "final_mean" in header and "shape" in header and "beats_greedy_rate" in header


def test_run_sweep_and_reporting(tmp_path):
    results = run_sweep(
        shapes=("blob",),
        arms=("adam", "cubic_gauss_newton"),
        seeds=(0,),
        size=12,
        side=5,
        steps=20,
        out_dir=str(tmp_path),
    )
    assert len(results) == 2
    assert all(verify_cover(make_instance("blob", size=12, side=5).image, r.squares, 5) for r in results)
    summary = summarize(results)
    assert ("blob", "adam") in summary
    table = format_table(results)
    assert "blob" in table and "greedy" in table
    write_results(results, str(tmp_path))
    assert (tmp_path / "results.json").exists()
    csv_path = tmp_path / "results.csv"
    assert csv_path.exists()
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "shape_kind" in header.split(",")
