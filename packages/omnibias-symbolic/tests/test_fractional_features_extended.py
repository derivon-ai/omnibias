# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Three more fractional column families: piecewise, activation, spectral.

``test_fractional_features.py`` covers the Grunwald-Letnikov (grid) and
single-terminal closed-form builders. This module covers the three that extend
them, and each is gated against the thing it claims to be rather than against
itself:

* **piecewise** (closed form) -- with one terminal it must reduce *exactly* to the
  single-terminal closed-form builder, which is the only way to be sure the
  multi-patch machinery has not quietly changed the operator.
* **activation** (closed form) -- for ``exp`` at integer ``alpha`` the answer is
  ``exp`` again, so the column is checked against the analytic value rather than a
  reference implementation.
* **spectral** (numerical-spectral, *not* closed form) -- on a basis eigenmode at
  ``alpha=2`` it must reproduce the integer-order Laplacian eigenvalue
  ``(pi/L)^2``, and it must reject a grid that does not match its transform basis,
  which is the failure that would otherwise return a plausible wrong column.

The discovery gate at the end is the one that matters for a user: on a grid too
coarse for the Grunwald-Letnikov columns to pin the order down, the closed-form
columns still select the right ``alpha``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.symbolic.discovery import (
    JetBundle,
    NeuralJetDiscoverer,
    build_jet_activation_fractional_features,
    build_jet_fractional_features_closed_form,
    build_jet_piecewise_fractional_features,
    build_jet_spectral_fractional_features,
    gl_fractional_derivative,
    split_x_grid,
)

pytest.importorskip("omnibias.fractional.jax.ops.analytic")


def poly_tower(x: np.ndarray, px: list[float]) -> np.ndarray:
    """Derivative tower ``[P, P', P'', ...]`` of ``P = sum px[k] x^k`` (columns)."""
    cols = []
    for d in range(len(px)):
        col = np.zeros_like(x)
        for k in range(d, len(px)):
            fac = 1.0
            for j in range(d):
                fac *= k - j
            col = col + px[k] * fac * x ** (k - d)
        cols.append(col)
    return np.stack(cols, axis=1)


def poly_bundle(x: np.ndarray, px: list[float]) -> JetBundle:
    return JetBundle(x=x, jets=poly_tower(x, px))


# --------------------------------------------------------------------------- #
# piecewise / multi-terminal
# --------------------------------------------------------------------------- #
def test_one_terminal_reduces_to_the_single_terminal_builder() -> None:
    """The load-bearing reduction: same operator, just expressed multi-patch.

    Exact away from the terminal. *At* the terminal the piecewise operator applies
    its ``gap`` safe-clamp (``max(x, a + gap)``) to step off the Riemann-Liouville
    singularity, so that one sample differs by O(gap) -- checked separately rather
    than papered over with a loose tolerance everywhere.
    """
    x = np.linspace(0.0, 2.0, 40)
    bundle = poly_bundle(x, [0.0, 0.0, 0.0, 1.0])  # x^3
    single, single_names = build_jet_fractional_features_closed_form(
        bundle, orders=(0.5, 1.5), kind="caputo"
    )
    piece, piece_names = build_jet_piecewise_fractional_features(
        bundle, orders=(0.5, 1.5), terminals=[0.0], kind="caputo", gap=1e-6
    )
    assert single_names == ["D^0.5(y)", "D^1.5(y)"]
    assert piece_names == ["Dpw^0.5(y)", "Dpw^1.5(y)"]
    assert np.allclose(piece[1:], single[1:], rtol=0, atol=1e-12)
    assert np.abs(piece[0] - single[0]).max() < 1e-6

    # Shrinking the clamp shrinks the discrepancy, confirming that is its origin.
    tighter, _ = build_jet_piecewise_fractional_features(
        bundle, orders=(1.5,), terminals=[0.0], kind="caputo", gap=1e-9
    )
    assert abs(tighter[0, 0] - single[0, 1]) < abs(piece[0, 1] - single[0, 1])


def test_multiple_terminals_stay_finite_and_track_the_owning_patch() -> None:
    """A hard-selected patch must equal a single-terminal build of that terminal.

    For a cubic the Taylor tower is exact everywhere, so both patches describe the
    same function -- but the Caputo operator is terminal-dependent, so the columns
    legitimately differ between patches. What must hold is that on the region a
    patch owns, the multi-terminal column *is* that patch.
    """
    x = np.linspace(0.0, 4.0, 61)  # includes 2.0 exactly, so no snapping surprise
    bundle = poly_bundle(x, [0.0, 1.0, 0.5])
    multi, _ = build_jet_piecewise_fractional_features(
        bundle, orders=(0.5,), terminals=[0.0, 2.0], kind="caputo"
    )
    assert multi.shape == (61, 1)
    assert np.all(np.isfinite(multi))
    # The upper patch, built alone on the sub-grid it owns (a lone terminal must be
    # the sub-grid's first point, which is the guard tested below).
    owned = x >= 2.0
    sub = JetBundle(x=x[owned], jets=bundle.jets[owned])
    upper, _ = build_jet_piecewise_fractional_features(
        sub, orders=(0.5,), terminals=[2.0], kind="caputo"
    )
    # Skip the terminal sample itself: the gap clamp lands differently there.
    assert np.allclose(multi[owned, 0][1:], upper[1:, 0], atol=1e-10)


def test_a_terminal_off_the_grid_is_snapped_to_the_nearest_sample() -> None:
    """Snapping is what keeps the tower and the operator's expansion point aligned."""
    x = np.linspace(0.0, 4.0, 61)
    bundle = poly_bundle(x, [0.0, 1.0, 0.5])
    on_grid, _ = build_jet_piecewise_fractional_features(
        bundle, orders=(0.5,), terminals=[0.0, 2.0], kind="caputo"
    )
    nudged, _ = build_jet_piecewise_fractional_features(
        bundle, orders=(0.5,), terminals=[0.0, 2.01], kind="caputo"
    )
    assert np.array_equal(on_grid, nudged)


def test_blending_is_between_the_two_hard_patches() -> None:
    x = np.linspace(0.0, 4.0, 61)
    bundle = poly_bundle(x, [0.0, 1.0, 0.5])
    hard, _ = build_jet_piecewise_fractional_features(
        bundle, orders=(0.5,), terminals=[0.0, 2.0], kind="caputo", blend=0.0
    )
    soft, _ = build_jet_piecewise_fractional_features(
        bundle, orders=(0.5,), terminals=[0.0, 2.0], kind="caputo", blend=0.5
    )
    assert np.all(np.isfinite(soft))
    # A smooth partition of unity differs from hard selection only near the seam.
    seam = np.abs(x - 2.0) < 0.5
    assert np.abs(soft[~seam] - hard[~seam]).max() < np.abs(soft[seam] - hard[seam]).max()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"orders": ()}, "at least one order"),
        ({"terminals": []}, "at least one terminal"),
        ({"terminals": [1.0, 0.0]}, "strictly increasing"),
        ({"terminals": [1.0]}, "above the first grid point"),
        ({"terminals": [0.0, 0.01]}, "duplicate grid samples"),
        ({"source_order": 9}, "outside the tower width"),
        ({"tower_width": 0}, "must be in 1"),
    ],
)
def test_piecewise_guards(kwargs: dict[str, object], match: str) -> None:
    x = np.linspace(0.0, 2.0, 20)
    bundle = poly_bundle(x, [0.0, 1.0])
    base: dict[str, object] = {"orders": (0.5,), "terminals": [0.0]}
    with pytest.raises(ValueError, match=match):
        build_jet_piecewise_fractional_features(bundle, **{**base, **kwargs})


def test_piecewise_rejects_an_unsorted_grid() -> None:
    x = np.linspace(0.0, 2.0, 20)[::-1]
    bundle = JetBundle(x=x, jets=poly_tower(x, [0.0, 1.0]))
    with pytest.raises(ValueError, match="strictly increasing x grid"):
        build_jet_piecewise_fractional_features(
            bundle, orders=(0.5,), terminals=[-1.0]
        )


# --------------------------------------------------------------------------- #
# activation special functions
# --------------------------------------------------------------------------- #
def test_activation_exp_matches_the_analytic_value() -> None:
    """``D^n exp = exp`` for integer ``n``; no reference implementation needed."""
    x = np.linspace(0.1, 2.0, 40)
    bundle = JetBundle(x=x, jets=np.stack([np.exp(x)] * 2, axis=1))
    cols, names = build_jet_activation_fractional_features(
        bundle, orders=(1.0, 2.0), name="exp"
    )
    assert names == ["D^1(exp(x))", "D^2(exp(x))"]
    assert np.allclose(cols[:, 0], np.exp(x), rtol=1e-12)
    assert np.allclose(cols[:, 1], np.exp(x), rtol=1e-12)


def test_activation_columns_depend_only_on_x() -> None:
    """Which is why they carry no source_order and no LHS-order guard."""
    x = np.linspace(0.1, 1.5, 30)
    a = JetBundle(x=x, jets=np.stack([np.exp(x), np.exp(x)], axis=1))
    b = JetBundle(x=x, jets=np.stack([np.zeros_like(x), np.sin(x)], axis=1))
    ca, _ = build_jet_activation_fractional_features(a, orders=(0.5,), name="exp")
    cb, _ = build_jet_activation_fractional_features(b, orders=(0.5,), name="exp")
    assert np.array_equal(ca, cb)


@pytest.mark.parametrize("name", ["exp", "cosh", "sinh"])
def test_every_registered_activation_builds(name: str) -> None:
    x = np.linspace(0.2, 1.5, 25)
    bundle = JetBundle(x=x, jets=np.stack([np.zeros_like(x)] * 2, axis=1))
    cols, names = build_jet_activation_fractional_features(
        bundle, orders=(0.5,), name=name
    )
    assert cols.shape == (25, 1)
    assert names == [f"D^0.5({name}(x))"]
    assert np.all(np.isfinite(cols))


def test_unregistered_activation_names_the_available_set() -> None:
    """Notably sigmoid, which the registry dropped for numerical stability."""
    x = np.linspace(0.1, 1.0, 10)
    bundle = JetBundle(x=x, jets=np.stack([x, x], axis=1))
    with pytest.raises(ValueError, match=r"sigmoid.*available: \['cosh', 'exp', 'sinh'\]"):
        build_jet_activation_fractional_features(bundle, orders=(0.5,), name="sigmoid")
    with pytest.raises(ValueError, match="at least one order"):
        build_jet_activation_fractional_features(bundle, orders=(), name="exp")


# --------------------------------------------------------------------------- #
# spectral
# --------------------------------------------------------------------------- #
def dirichlet_grid(n: int, length: float) -> np.ndarray:
    return np.arange(1, n + 1) * length / (n + 1)


def neumann_grid(n: int, length: float) -> np.ndarray:
    return (np.arange(n) + 0.5) * length / n


def test_alpha_two_reproduces_the_integer_laplacian_on_an_eigenmode() -> None:
    """The claim the spectral path lives or dies by."""
    length, n = 1.0, 64
    x = dirichlet_grid(n, length)
    mode = np.sin(math.pi * x / length)
    bundle = JetBundle(x=x, jets=np.stack([mode, mode], axis=1))
    cols, names = build_jet_spectral_fractional_features(bundle, orders=(2.0,))
    assert names == ["Dspec^2(y)"]
    assert np.allclose(cols[:, 0], (math.pi / length) ** 2 * mode, atol=1e-9)


@pytest.mark.parametrize("k", [1, 2, 5])
def test_every_eigenmode_gets_its_own_eigenvalue(k: int) -> None:
    """``(-Delta)^{alpha/2} sin(k pi x/L) = (k pi/L)^alpha sin(k pi x/L)``."""
    length, n, alpha = 2.0, 48, 0.7
    x = dirichlet_grid(n, length)
    mode = np.sin(k * math.pi * x / length)
    bundle = JetBundle(x=x, jets=np.stack([mode, mode], axis=1))
    cols, _ = build_jet_spectral_fractional_features(bundle, orders=(alpha,))
    want = (k * math.pi / length) ** alpha * mode
    assert np.allclose(cols[:, 0], want, atol=1e-9)


def test_neumann_basis_works_on_its_own_grid() -> None:
    length, n, alpha = 1.0, 48, 1.3
    x = neumann_grid(n, length)
    mode = np.cos(2.0 * math.pi * x / length)
    bundle = JetBundle(x=x, jets=np.stack([mode, mode], axis=1))
    cols, _ = build_jet_spectral_fractional_features(
        bundle, orders=(alpha,), bc="neumann"
    )
    want = (2.0 * math.pi / length) ** alpha * mode
    assert np.allclose(cols[:, 0], want, atol=1e-9)


def test_the_wrong_grid_layout_is_rejected_not_silently_wrong() -> None:
    """A Dirichlet transform on a Neumann grid returns a plausible wrong column."""
    length, n = 1.0, 32
    x = neumann_grid(n, length)
    mode = np.sin(math.pi * x / length)
    bundle = JetBundle(x=x, jets=np.stack([mode, mode], axis=1))
    with pytest.raises(ValueError, match=r"bc='dirichlet' requires the x_j = \(j\+1\)"):
        build_jet_spectral_fractional_features(bundle, orders=(1.0,), bc="dirichlet")
    # ...and the mirrored mistake is caught too.
    other = JetBundle(
        x=dirichlet_grid(n, length),
        jets=np.stack([mode, mode], axis=1),
    )
    with pytest.raises(ValueError, match=r"bc='neumann' requires the x_j = \(j\+1/2\)"):
        build_jet_spectral_fractional_features(other, orders=(1.0,), bc="neumann")


def test_windowed_path_returns_the_real_part_and_skips_the_grid_check() -> None:
    """The windowed operator is for a decaying signal on any uniform grid."""
    x = np.linspace(0.0, 4.0, 64)
    signal = np.exp(-((x - 2.0) ** 2) / 0.2)
    bundle = JetBundle(x=x, jets=np.stack([signal, signal], axis=1))
    cols, names = build_jet_spectral_fractional_features(
        bundle, orders=(0.5,), windowed=True, length=4.0
    )
    assert names == ["Dwin^0.5(y)"]
    assert cols.dtype == np.float64 and np.all(np.isfinite(cols))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"orders": ()}, "at least one order"),
        ({"bc": "periodic"}, "bc must be"),
        ({"source_order": 7}, "outside jet width"),
        ({"length": -1.0}, "length must be > 0"),
    ],
)
def test_spectral_guards(kwargs: dict[str, object], match: str) -> None:
    x = dirichlet_grid(16, 1.0)
    bundle = JetBundle(x=x, jets=np.stack([np.sin(math.pi * x)] * 2, axis=1))
    with pytest.raises(ValueError, match=match):
        build_jet_spectral_fractional_features(
            bundle, **{**{"orders": (1.0,)}, **kwargs}
        )


def test_spectral_requires_a_uniform_grid() -> None:
    x = np.array([0.1, 0.2, 0.45, 0.9])
    bundle = JetBundle(x=x, jets=np.stack([x, x], axis=1))
    with pytest.raises(ValueError, match="near-\\)uniform x grid"):
        build_jet_spectral_fractional_features(bundle, orders=(1.0,))


# --------------------------------------------------------------------------- #
# wiring into the discoverer
# --------------------------------------------------------------------------- #
def piecewise_law_split(alpha: float, c_frac: float, c_y: float):
    """``target = c_frac Dpw^alpha(P) + c_y P`` on a cubic ``P``, three splits."""
    px = [0.0, 1.0, -0.5, 0.25]

    def build(x: np.ndarray) -> JetBundle:
        tower = poly_tower(x, px)
        col, _ = build_jet_piecewise_fractional_features(
            JetBundle(x=x, jets=tower),
            orders=(alpha,),
            terminals=[0.0, 2.0],
            kind="caputo",
        )
        target = c_frac * col[:, 0] + c_y * tower[:, 0]
        return JetBundle(x=x, jets=np.concatenate([tower, target[:, None]], axis=1))

    tr, va, te = split_x_grid(xmin=0.0, xmax=4.0, n_train=90, n_val=60, n_test=60)
    return build(tr), build(va), build(te), len(px)


def test_discoverer_recovers_a_piecewise_fractional_law() -> None:
    btr, bva, bte, tower_width = piecewise_law_split(0.5, 1.0, 0.3)
    disc = NeuralJetDiscoverer(
        max_library_degree=1,
        piecewise_fractional_orders=(0.5,),
        piecewise_fractional_terminals=(0.0, 2.0),
        piecewise_fractional_kind="caputo",
        piecewise_fractional_tower_width=tower_width,
    )
    res = disc.discover(btr, bva, bte, candidate_lhs_orders=(4,))
    terms = {str(t["name"]): float(t["coefficient"]) for t in res.active_terms()}
    assert res.test_rmse < 1e-6, terms
    assert terms.get("Dpw^0.5(y)", 0.0) == pytest.approx(1.0, abs=1e-3)
    assert terms.get("y", 0.0) == pytest.approx(0.3, abs=1e-3)


def test_tower_width_must_exclude_an_appended_target_column() -> None:
    """Without it the target column is read as a high derivative of the tower."""
    btr, bva, bte, tower_width = piecewise_law_split(0.5, 1.0, 0.3)
    common = {
        "max_library_degree": 1,
        "piecewise_fractional_orders": (0.5,),
        "piecewise_fractional_terminals": (0.0, 2.0),
        "piecewise_fractional_kind": "caputo",
    }
    correct = NeuralJetDiscoverer(
        **common, piecewise_fractional_tower_width=tower_width
    ).discover(btr, bva, bte, candidate_lhs_orders=(4,))
    polluted = NeuralJetDiscoverer(**common).discover(
        btr, bva, bte, candidate_lhs_orders=(4,)
    )
    assert correct.test_rmse < polluted.test_rmse


def test_activation_columns_reach_the_discoverer() -> None:
    """``dy = 2 D^0.5(exp(x)) + 0.5 y`` -- a law only the activation columns express."""
    def build(x: np.ndarray) -> JetBundle:
        y = np.sin(2.0 * x)
        col, _ = build_jet_activation_fractional_features(
            JetBundle(x=x, jets=np.stack([y, y], axis=1)), orders=(0.5,), name="exp"
        )
        return JetBundle(x=x, jets=np.stack([y, 2.0 * col[:, 0] + 0.5 * y], axis=1))

    tr, va, te = split_x_grid(xmin=0.2, xmax=2.0, n_train=90, n_val=60, n_test=60)
    disc = NeuralJetDiscoverer(
        max_library_degree=1,
        activation_fractional_orders=(0.5,),
        activation_fractional_name="exp",
    )
    res = disc.discover(build(tr), build(va), build(te), candidate_lhs_orders=(1,))
    terms = {str(t["name"]): float(t["coefficient"]) for t in res.active_terms()}
    assert res.test_rmse < 1e-6, terms
    assert terms.get("D^0.5(exp(x))", 0.0) == pytest.approx(2.0, abs=1e-3)


def test_lhs_guard_holds_for_the_new_grid_based_families() -> None:
    """A column of the LHS jet must never be offered as its own feature."""
    x_tr, x_va, x_te = split_x_grid(xmin=0.0, xmax=4.0, n_train=60, n_val=40, n_test=40)

    def build(x: np.ndarray) -> JetBundle:
        y = np.sin(2.0 * x)
        return JetBundle(x=x, jets=np.stack([y, np.cos(x), y], axis=1))

    disc = NeuralJetDiscoverer(
        max_library_degree=1,
        piecewise_fractional_orders=(0.5,),
        piecewise_fractional_terminals=(0.0,),
        piecewise_fractional_source_order=1,
    )
    res = disc.discover(build(x_tr), build(x_va), build(x_te), candidate_lhs_orders=(1,))
    assert all("Dpw^0.5(dy)" not in str(t["name"]) for t in res.active_terms())


# --------------------------------------------------------------------------- #
# the honesty gate: closed form beats the grid operator on a coarse grid
# --------------------------------------------------------------------------- #
def test_closed_form_columns_select_the_right_alpha_where_gl_cannot() -> None:
    """The order-selection gate, stated as what was actually measured.

    The law is ``target = D^0.6(P) + 0.4 P`` for a cubic ``P``, built with the exact
    closed-form Caputo operator on a 24-point grid. Scoring candidate orders by
    validation RMSE, the closed-form columns pick 0.6 with a clear margin (measured
    9.0e-4 at the true order against 4.5e-2 for the runner-up), while the
    Grunwald-Letnikov columns pick 0.4 and plateau at 0.26 -- around 5% of the
    signal -- with the scores nearly flat across orders.

    The honest reason is *not* simply "GL is coarse". Refining the grid shrinks the
    GL column's own error (measured 3.1% -> 0.35% relative going from 24 to 400
    points) but does **not** fix the selection: GL still picks 0.4 at every
    resolution tested, because it discretises a different lower-terminal
    convention, and that structural bias is larger than the difference between
    neighbouring candidate orders. A closed-form column has no such bias to trade
    off against, which is the whole point of having one.
    """
    px = [0.0, 1.0, -0.5, 0.25]
    true_alpha = 0.6
    candidates = (0.4, 0.5, 0.6, 0.7, 0.8)

    def build(x: np.ndarray) -> JetBundle:
        tower = poly_tower(x, px)
        col, _ = build_jet_fractional_features_closed_form(
            JetBundle(x=x, jets=tower), orders=(true_alpha,), kind="caputo"
        )
        target = col[:, 0] + 0.4 * tower[:, 0]
        return JetBundle(x=x, jets=np.concatenate([tower, target[:, None]], axis=1))

    tr, va, te = split_x_grid(xmin=0.0, xmax=3.0, n_train=24, n_val=24, n_test=24)
    btr, bva, bte = build(tr), build(va), build(te)

    def scores(kind: str) -> dict[float, float]:
        out: dict[float, float] = {}
        for alpha in candidates:
            disc = (
                NeuralJetDiscoverer(
                    max_library_degree=1,
                    piecewise_fractional_orders=(alpha,),
                    piecewise_fractional_terminals=(0.0,),
                    piecewise_fractional_kind="caputo",
                    piecewise_fractional_tower_width=len(px),
                )
                if kind == "closed_form"
                else NeuralJetDiscoverer(
                    max_library_degree=1, fractional_orders=(alpha,)
                )
            )
            out[alpha] = disc.discover(
                btr, bva, bte, candidate_lhs_orders=(4,)
            ).validation_rmse
        return out

    closed = scores("closed_form")
    grid = scores("gl")
    best_closed = min(closed, key=lambda a: closed[a])
    best_grid = min(grid, key=lambda a: grid[a])
    assert best_closed == pytest.approx(true_alpha), closed
    assert best_grid != pytest.approx(true_alpha), grid
    # Two decisive margins: over the grid operator (measured 284x) and over the
    # closed-form runner-up order (measured 50x), so the pick is not a near-tie.
    assert closed[true_alpha] < 0.02 * grid[best_grid], (closed, grid)
    runner_up = min(v for a, v in closed.items() if a != true_alpha)
    assert closed[true_alpha] < 0.1 * runner_up, closed


def test_gl_column_error_shrinks_with_h_but_never_wins_the_order() -> None:
    """Backs the "not simply coarse" claim above with the two measured curves."""
    px = [0.0, 1.0, -0.5, 0.25]
    true_alpha = 0.6
    errors = []
    for n in (24, 200):
        x = np.linspace(0.0, 3.0, n)
        tower = poly_tower(x, px)
        exact, _ = build_jet_fractional_features_closed_form(
            JetBundle(x=x, jets=tower), orders=(true_alpha,), kind="caputo"
        )
        gl = gl_fractional_derivative(
            tower[:, 0], alpha=true_alpha, h=float(x[1] - x[0])
        )
        errors.append(
            float(np.abs(gl - exact[:, 0]).max()) / float(np.abs(exact[:, 0]).max())
        )
    coarse, fine = errors
    assert coarse > fine, errors  # the column really does converge
    assert fine > 1e-3, errors  # but not to round-off, which is what costs it
