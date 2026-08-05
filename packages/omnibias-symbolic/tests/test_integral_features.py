# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Integral (non-local) columns: the first library here that is not local.

Every other feature family in ``discovery.py`` determines a row from the signal
near that point. These do not, which is what lets a discovery run express an
integro-differential law -- and which is also where the new failure modes live.
The module is organised around those failure modes rather than around the code:

* **against analytic values**, not a reference implementation -- the three
  families each have a closed-form answer for a chosen integrand;
* **the terminal is per-bundle.** An indefinite integral is defined only up to a
  constant fixed by where the grid starts, so two splits that begin at different
  ``x`` carry genuinely different columns. This is the one thing that will
  silently ruin a discovery run, so it is tested from both sides: that it really
  does break, and that ``origin`` fixes it;
* **identifiability.** A rank-1 separable kernel produces a column proportional
  to ``x``, which no fit can distinguish from the ``x`` column. The condition
  diagnostic must say so rather than the run quietly returning a wrong
  coefficient;
* **the LHS guard runs the other way round** from the derivative families:
  integrating the jet one order *above* the left-hand side reproduces it, while a
  global Fredholm column never does -- and must not be dropped, since putting the
  unknown on both sides is exactly what a Fredholm equation is.

The two end-to-end gates recover ``y' = -int_0^x y`` and a genuine Fredholm
equation of the second kind with their coefficients, and the field-side gate
recovers a nonlocal PDE through the ``extra_columns_fn`` hook.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.symbolic.diagnostics import design_conditioning_report
from omnibias.symbolic.discovery import (
    JetBundle,
    NeuralJetDiscoverer,
    build_jet_integral_features,
    split_x_grid,
)
from omnibias.symbolic.field_discovery import (
    FieldJet,
    FieldLawDiscoverer,
    measure_integral_columns,
)

Measure = pytest.importorskip("omnibias.measure").Measure

PI = math.pi


def trapezoid_measure(x: np.ndarray) -> object:
    """The composite trapezoid rule on ``x``, as a Measure."""
    w = np.empty_like(x)
    h = np.diff(x)
    w[0] = 0.5 * h[0]
    w[-1] = 0.5 * h[-1]
    w[1:-1] = 0.5 * (h[:-1] + h[1:])
    return Measure(nodes=x.reshape(-1, 1), weights=w, name="trapezoid")


def gauss_measure(a: float, b: float, n: int) -> tuple[np.ndarray, object]:
    """A sorted Gauss-Legendre rule on ``[a, b]`` and the grid it lives on."""
    from omnibias.measure._core.measure import lebesgue

    mu = lebesgue([(a, b)], n)
    order = np.argsort(mu.nodes[:, 0])
    x = mu.nodes[order, 0]
    return x, Measure(nodes=x.reshape(-1, 1), weights=mu.weights[order], name="gl")


def sine_bundle(x: np.ndarray) -> JetBundle:
    return JetBundle(x=x, jets=np.stack([np.sin(x), np.cos(x)], axis=1))


# --------------------------------------------------------------------------- #
# each family against its analytic value
# --------------------------------------------------------------------------- #
def test_running_integral_matches_the_antiderivative() -> None:
    """``int_0^x sin = 1 - cos x``."""
    x = np.linspace(0.0, 2.0, 401)
    cols, names = build_jet_integral_features(sine_bundle(x))
    assert names == ["I(y)"]
    assert np.abs(cols[:, 0] - (1.0 - np.cos(x))).max() < 1e-5


def test_fredholm_column_matches_the_definite_integral() -> None:
    """With ``K == 1`` every row is the same number: ``int_0^2 sin = 1 - cos 2``."""
    x = np.linspace(0.0, 2.0, 401)
    cols, names = build_jet_integral_features(
        sine_bundle(x),
        measure=trapezoid_measure(x),
        kernels={"one": lambda xi, tj: np.ones_like(xi)},
        running=False,
    )
    assert names == ["F[one](y)"]
    assert np.abs(cols[:, 0] - (1.0 - math.cos(2.0))).max() < 1e-5


def test_volterra_column_matches_its_analytic_value() -> None:
    """``int_0^x (x - t) sin t dt = x - sin x``."""
    x = np.linspace(0.0, 2.0, 401)
    cols, names = build_jet_integral_features(
        sine_bundle(x),
        volterra_kernels={"xt": lambda xi, tj: xi - tj},
        running=False,
    )
    assert names == ["V[xt](y)"]
    assert np.abs(cols[:, 0] - (x - np.sin(x))).max() < 1e-4


def test_a_high_order_rule_is_worth_having() -> None:
    """The payoff of taking the weights from a Measure instead of the grid.

    24 Gauss-Legendre nodes evaluate a smooth Fredholm column to round-off, where
    401 trapezoid points reach only ``1e-6``. This is why ``measure`` exists at all
    rather than the builder deriving trapezoid weights for every family.
    """
    x, mu = gauss_measure(0.0, 2.0, 24)
    gl, _ = build_jet_integral_features(
        sine_bundle(x),
        measure=mu,
        kernels={"sep": lambda xi, tj: np.exp(xi) * tj},
        running=False,
    )
    want_gl = np.exp(x) * (math.sin(2.0) - 2.0 * math.cos(2.0))
    assert np.abs(gl[:, 0] - want_gl).max() < 1e-13

    xt = np.linspace(0.0, 2.0, 401)
    trap, _ = build_jet_integral_features(
        sine_bundle(xt),
        measure=trapezoid_measure(xt),
        kernels={"sep": lambda xi, tj: np.exp(xi) * tj},
        running=False,
    )
    want_trap = np.exp(xt) * (math.sin(2.0) - 2.0 * math.cos(2.0))
    assert np.abs(trap[:, 0] - want_trap).max() > 1e-9


def test_the_measure_may_be_a_factory() -> None:
    """Which is how splits of different resolution each get their own rule."""
    x = np.linspace(0.0, 2.0, 201)
    kw = {"kernels": {"one": lambda xi, tj: np.ones_like(xi)}, "running": False}
    direct, _ = build_jet_integral_features(
        sine_bundle(x), measure=trapezoid_measure(x), **kw
    )
    made, _ = build_jet_integral_features(
        sine_bundle(x), measure=trapezoid_measure, **kw
    )
    assert np.array_equal(direct, made)


# --------------------------------------------------------------------------- #
# the terminal is per-bundle: the failure mode, and the fix
# --------------------------------------------------------------------------- #
def test_columns_on_grids_with_different_starts_really_do_disagree() -> None:
    """Documented rather than surprising: the whole reason ``origin`` exists.

    ``split_x_grid`` interleaves, so its three splits begin at different ``x``.
    Left alone, each running-integral column integrates from its own first sample,
    and a law fitted on one split does not describe another.
    """
    train, val, _test = split_x_grid(
        xmin=0.0, xmax=6.0, n_train=240, n_val=160, n_test=160
    )
    assert train[0] != val[0], "the fixture no longer exercises the hazard"
    on_val, _ = build_jet_integral_features(sine_bundle(val))
    # int_0^x sin = 1 - cos x; from val[0] the column is short by exactly that
    # much evaluated at val[0], and by nothing else.
    expected_gap = 1.0 - math.cos(val[0])
    assert np.allclose(
        on_val[:, 0], (1.0 - np.cos(val)) - expected_gap, atol=2e-4
    )
    # Small in absolute terms and still fatal: it is ~9% of the column's range,
    # where the end-to-end gates below fit to 1e-3.
    assert expected_gap / np.ptp(on_val[:, 0]) > 0.05


def test_origin_rebases_the_causal_columns() -> None:
    x = np.linspace(0.0, 4.0, 801)
    base, _ = build_jet_integral_features(sine_bundle(x))
    shifted, _ = build_jet_integral_features(sine_bundle(x), origin=1.0)
    # int_1^x = int_0^x - int_0^1, a constant shift, negative below the origin.
    assert np.allclose(shifted[:, 0], base[:, 0] - (1.0 - math.cos(1.0)), atol=1e-6)
    assert shifted[x < 1.0, 0].max() < 0.0
    # origin at the first sample is the default, exactly.
    same, _ = build_jet_integral_features(sine_bundle(x), origin=0.0)
    assert np.array_equal(same, base)


def test_origin_is_snapped_to_a_sample() -> None:
    """Same convention as the piecewise fractional terminals, for the same reason."""
    x = np.linspace(0.0, 4.0, 401)  # spacing 0.01, so 1.0 is a sample
    on_grid, _ = build_jet_integral_features(sine_bundle(x), origin=1.0)
    nudged, _ = build_jet_integral_features(sine_bundle(x), origin=1.002)
    assert np.array_equal(on_grid, nudged)


def test_origin_also_rebases_a_volterra_column() -> None:
    x = np.linspace(0.0, 3.0, 601)
    kw = {"volterra_kernels": {"xt": lambda xi, tj: xi - tj}, "running": False}
    base, _ = build_jet_integral_features(sine_bundle(x), **kw)
    shifted, _ = build_jet_integral_features(sine_bundle(x), origin=1.0, **kw)
    assert not np.allclose(base, shifted)
    # int_1^x (x-t) sin t dt, by parts: (x - 1) cos 1 - (sin x - sin 1)
    want = (x - 1.0) * math.cos(1.0) - (np.sin(x) - math.sin(1.0))
    assert np.abs(shifted[x >= 1.0, 0] - want[x >= 1.0]).max() < 1e-4


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_a_measure_on_other_nodes_is_refused_rather_than_interpolated() -> None:
    x = np.linspace(0.0, 2.0, 64)
    other = np.linspace(0.0, 2.0, 64) + 0.001
    with pytest.raises(ValueError, match="must be the bundle's 64-point grid"):
        build_jet_integral_features(
            sine_bundle(x),
            measure=trapezoid_measure(other),
            kernels={"one": lambda xi, tj: np.ones_like(xi)},
            running=False,
        )


def test_fredholm_without_a_measure_says_where_the_weights_come_from() -> None:
    x = np.linspace(0.0, 2.0, 32)
    with pytest.raises(ValueError, match="needs a measure to build Fredholm columns"):
        build_jet_integral_features(
            sine_bundle(x), kernels={"one": lambda xi, tj: np.ones_like(xi)}
        )


def test_a_multidimensional_measure_is_refused() -> None:
    x = np.linspace(0.0, 2.0, 16)
    flat = trapezoid_measure(x)
    with pytest.raises(ValueError, match="integral features are 1-D"):
        build_jet_integral_features(
            sine_bundle(x),
            measure=flat.product(flat),
            kernels={"one": lambda xi, tj: np.ones_like(xi)},
            running=False,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"running": False}, "needs at least one column"),
        ({"source_order": 5}, "outside jet width"),
        ({"origin": 9.0}, "outside the grid"),
        (
            {
                "kernels": {"k": lambda xi, tj: np.ones_like(xi)},
                "volterra_kernels": {"k": lambda xi, tj: np.ones_like(xi)},
                "measure": trapezoid_measure,
            },
            "appear in both",
        ),
        (
            {"volterra_kernels": {"bad": lambda xi, tj: np.ones(3)}},
            "returned shape",
        ),
    ],
)
def test_builder_guards(kwargs: dict[str, object], match: str) -> None:
    x = np.linspace(0.0, 2.0, 24)
    with pytest.raises(ValueError, match=match):
        build_jet_integral_features(sine_bundle(x), **kwargs)


def test_an_unsorted_or_tiny_grid_is_refused() -> None:
    x = np.linspace(0.0, 2.0, 24)[::-1]
    with pytest.raises(ValueError, match="strictly increasing x grid"):
        build_jet_integral_features(JetBundle(x=x, jets=np.zeros((24, 2))))
    with pytest.raises(ValueError, match="at least two grid points"):
        build_jet_integral_features(JetBundle(x=np.zeros(1), jets=np.zeros((1, 2))))


# --------------------------------------------------------------------------- #
# conditioning: the diagnostic that makes an unidentifiable kernel visible
# --------------------------------------------------------------------------- #
def test_the_report_separates_bad_scaling_from_collinearity() -> None:
    """The two numbers answer different questions, so they are reported separately."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 3))
    scaled = a * np.array([1.0, 1e6, 1e-6])
    report = design_conditioning_report(scaled)
    assert report["max_column_scale_ratio"] > 1e10
    assert report["design_condition_number"] > 1e10
    # ...but standardizing reveals independent columns, which is the truth.
    assert report["standardized_condition_number"] < 10.0

    collinear = np.column_stack([a[:, 0], a[:, 1], a[:, 0] + 1e-12 * a[:, 2]])
    both = design_conditioning_report(collinear)
    assert both["max_column_scale_ratio"] < 10.0
    assert both["standardized_condition_number"] > 1e8


def test_a_rank_one_kernel_is_reported_as_unidentifiable() -> None:
    """``K(x,t) = x t`` makes ``F[K](y)`` a multiple of ``x``, so no fit can separate them.

    The coefficient a run would return in that case is arbitrary, which is exactly
    the sort of quiet wrong answer the diagnostic exists to expose. Contrast with a
    kernel whose ``x``-dependence lies outside the polynomial library, where the
    standardized condition number is 1.
    """

    def bundle(n: int) -> JetBundle:
        x = np.linspace(0.0, 1.0, n)
        return JetBundle(x=x, jets=np.stack([x + 1.0, np.ones(n)], axis=1))

    def conditioning(kernel) -> float:
        disc = NeuralJetDiscoverer(
            max_library_degree=1,
            integral_kernels={"k": kernel},
            integral_measure=trapezoid_measure,
        )
        result = disc.discover(
            bundle(401), bundle(257), bundle(263), candidate_lhs_orders=(0,)
        )
        return float(result.diagnostics["standardized_condition_number"])

    degenerate = conditioning(lambda xi, tj: xi * tj)
    identifiable = conditioning(lambda xi, tj: np.sin(PI * xi) * tj)
    assert degenerate > 1e8, degenerate
    assert identifiable < 10.0, identifiable


# --------------------------------------------------------------------------- #
# the LHS guard, which runs the other way round here
# --------------------------------------------------------------------------- #
def test_integrating_one_order_above_the_lhs_is_dropped() -> None:
    """``I(dy) = y`` up to a constant, so offering it would hand over the answer."""

    def bundle(n: int) -> JetBundle:
        x = np.linspace(0.0, 3.0, n)
        return JetBundle(x=x, jets=np.stack([np.sin(x), np.cos(x)], axis=1))

    disc = NeuralJetDiscoverer(
        max_library_degree=1, integral_running=True, integral_source_order=1
    )
    result = disc.discover(
        bundle(301), bundle(211), bundle(217), candidate_lhs_orders=(0,)
    )
    assert all("I(" not in str(t["name"]) for t in result.active_terms())


def test_a_fredholm_column_is_kept_even_when_it_reads_the_lhs_jet() -> None:
    """The Fredholm family must survive ``source_order == lhs_order``.

    Dropping it would remove the headline capability: an integral equation of the
    second kind is *defined* by the unknown appearing under the integral as well as
    on the left.
    """

    def bundle(n: int) -> JetBundle:
        x = np.linspace(0.0, 1.0, n)
        return JetBundle(x=x, jets=np.stack([x + 1.0, np.ones(n)], axis=1))

    disc = NeuralJetDiscoverer(
        max_library_degree=1,
        integral_kernels={"k": lambda xi, tj: np.sin(PI * xi) * tj},
        integral_measure=trapezoid_measure,
    )
    result = disc.discover(
        bundle(401), bundle(257), bundle(263), candidate_lhs_orders=(0,)
    )
    assert "F[k](y)" in result.equation.term_names


# --------------------------------------------------------------------------- #
# end-to-end: two laws no local library can express
# --------------------------------------------------------------------------- #
def test_recovers_an_integro_differential_law() -> None:
    """``y'(x) = -int_0^x y`` with ``y = cos x``, so ``I(y) = sin x`` and ``y' = -sin x``.

    The splits differ in resolution rather than in start, so they share the
    terminal ``origin=0``; that is the discipline the per-bundle terminal imposes.
    """

    def bundle(n: int) -> JetBundle:
        x = np.linspace(0.0, 6.0, n)
        return JetBundle(x=x, jets=np.stack([np.cos(x), -np.sin(x)], axis=1))

    disc = NeuralJetDiscoverer(
        max_library_degree=1, integral_running=True, integral_origin=0.0
    )
    result = disc.discover(
        bundle(241), bundle(161), bundle(163), candidate_lhs_orders=(1,)
    )
    terms = {str(t["name"]): float(t["coefficient"]) for t in result.active_terms()}
    assert list(terms) == ["I(y)"], terms
    assert terms["I(y)"] == pytest.approx(-1.0, abs=1e-3)
    assert result.test_rmse < 1e-3


def test_recovers_a_fredholm_equation_of_the_second_kind() -> None:
    """``y(x) = 1 + x + int_0^1 sin(pi x) t y(t) dt``, exact solution ``1 + x + M sin(pi x)``.

    ``M = int_0^1 t y(t) dt`` closes the equation: substituting the ansatz gives
    ``M (1 - 1/pi) = 5/6``. The kernel's ``x``-factor is ``sin(pi x)``, deliberately
    outside the polynomial library, which is what makes the integral term
    identifiable at all -- see the rank-1 test above for what happens otherwise.
    """
    m = (5.0 / 6.0) / (1.0 - 1.0 / PI)

    def kernel(xi: np.ndarray, tj: np.ndarray) -> np.ndarray:
        return np.sin(PI * xi) * tj

    def bundle(n: int) -> JetBundle:
        x = np.linspace(0.0, 1.0, n)
        y = 1.0 + x + m * np.sin(PI * x)
        dy = 1.0 + m * PI * np.cos(PI * x)
        return JetBundle(x=x, jets=np.stack([y, dy], axis=1))

    # the equation really is satisfied by that solution, at quadrature accuracy
    check = bundle(801)
    col, _ = build_jet_integral_features(
        check, measure=trapezoid_measure(check.x), kernels={"k": kernel}, running=False
    )
    assert np.abs(check.jets[:, 0] - (1.0 + check.x + col[:, 0])).max() < 1e-6

    disc = NeuralJetDiscoverer(
        max_library_degree=1,
        integral_kernels={"k": kernel},
        integral_measure=trapezoid_measure,
    )
    result = disc.discover(
        bundle(401), bundle(257), bundle(263), candidate_lhs_orders=(0,)
    )
    terms = {str(t["name"]): float(t["coefficient"]) for t in result.active_terms()}
    assert set(terms) == {"F[k](y)", "x"}, terms
    assert terms["F[k](y)"] == pytest.approx(1.0, abs=1e-3)
    assert terms["x"] == pytest.approx(1.0, abs=1e-3)
    assert result.equation.intercept == pytest.approx(1.0, abs=1e-3)
    assert result.diagnostics["standardized_condition_number"] < 10.0


# --------------------------------------------------------------------------- #
# the field side: a nonlocal PDE through the extra_columns_fn hook
# --------------------------------------------------------------------------- #
C1, BETA, GAMMA, T_MAX = 0.1, 0.3, 0.45, 1.0
DECAY = C1 * PI**2


def field_grid(nx: int, nt: int) -> np.ndarray:
    x = (np.arange(nx) + 0.5) / nx
    t = (np.arange(nt) + 0.5) / nt * T_MAX
    mesh = np.meshgrid(x, t, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


def field_jet(points: np.ndarray) -> FieldJet:
    r"""Three modes of ``u_t = C1 u_xx + c2 int cos(3 pi x) u``, built analytically.

    ``sin(pi x) e^{-a t}`` and ``gamma sin(2 pi x) e^{-4 a t}`` are heat modes; the
    steady ``beta cos(3 pi x)`` is what the nonlocal term balances. Three modes,
    not two: with only two the columns ``u``, ``u_xx`` and ``F`` span the same
    plane, every pair of them reproduces ``u_t``, and there is no unique sparsest
    law to recover.
    """
    x, t = points[:, 0], points[:, 1]
    e1, e2 = np.exp(-DECAY * t), np.exp(-4.0 * DECAY * t)
    s1, s2, c3 = np.sin(PI * x), np.sin(2.0 * PI * x), np.cos(3.0 * PI * x)
    return FieldJet(
        X=points,
        order=2,
        var_names=("x", "t"),
        partials={
            (0, 0): s1 * e1 + BETA * c3 + GAMMA * s2 * e2,
            (1, 0): (
                PI * np.cos(PI * x) * e1
                - 3.0 * PI * BETA * np.sin(3.0 * PI * x)
                + 2.0 * PI * GAMMA * np.cos(2.0 * PI * x) * e2
            ),
            (0, 1): -DECAY * s1 * e1 - 4.0 * DECAY * GAMMA * s2 * e2,
            (2, 0): (
                -(PI**2) * s1 * e1
                - 9.0 * PI**2 * BETA * c3
                - 4.0 * PI**2 * GAMMA * s2 * e2
            ),
            (1, 1): (
                -DECAY * PI * np.cos(PI * x) * e1
                - 8.0 * DECAY * PI * GAMMA * np.cos(2.0 * PI * x) * e2
            ),
            (0, 2): DECAY**2 * s1 * e1 + 16.0 * DECAY**2 * GAMMA * s2 * e2,
        },
    )


def uniform_measure(points: np.ndarray) -> object:
    n = points.shape[0]
    return Measure(nodes=points, weights=np.full(n, T_MAX / n), name="uniform")


def cos3_kernel(query: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    return np.cos(3.0 * PI * query[:, None, 0]) * np.ones((1, nodes.shape[0]))


def test_field_columns_recover_a_nonlocal_pde() -> None:
    """``u_t = 0.1 u_xx + c2 int_Omega cos(3 pi x) u``, with ``c2`` read off analytically.

    ``c2 = 9 C1 pi^2 beta / S`` is what makes the steady ``cos(3 pi x)`` mode
    balance, where ``S = int u dmu``. ``S`` is a quadrature, so it drifts slightly
    between splits of different resolution -- which is why the residual gate is
    ``1e-3`` relative rather than round-off, and the honest statement of what these
    columns cost.
    """
    columns = measure_integral_columns(uniform_measure, {"c3": cos3_kernel})
    train = field_jet(field_grid(40, 30))
    integral = columns(train)["F[c3](u)"]
    s = float(integral[0] / math.cos(3.0 * PI * train.X[0, 0]))
    c2 = 9.0 * C1 * PI**2 * BETA / s
    residual = train.partial((0, 1)) - (C1 * train.partial((2, 0)) + c2 * integral)
    assert np.abs(residual).max() < 1e-12, "the constructed law is not exact"

    result = FieldLawDiscoverer(max_degree=1, time_axis=1).discover(
        train,
        field_jet(field_grid(34, 26)),
        field_jet(field_grid(37, 28)),
        lhs_index=(0, 1),
        extra_columns_fn=columns,
    )
    terms = {str(t["name"]): float(t["coefficient"]) for t in result.active_terms()}
    assert set(terms) == {"F[c3](u)", "u_xx"}, terms
    assert terms["u_xx"] == pytest.approx(C1, rel=1e-3)
    assert terms["F[c3](u)"] == pytest.approx(c2, rel=1e-3)
    assert result.test_rmse / result.target_scale < 1e-3


def test_field_columns_refuse_a_measure_on_other_points() -> None:
    columns = measure_integral_columns(
        uniform_measure(field_grid(10, 8)), {"c3": cos3_kernel}
    )
    with pytest.raises(ValueError, match="must be the jet's sample points"):
        columns(field_jet(field_grid(12, 9)))


def test_field_column_guards() -> None:
    with pytest.raises(ValueError, match="at least one kernel"):
        measure_integral_columns(uniform_measure, {})
    bad = measure_integral_columns(
        uniform_measure, {"bad": lambda query, nodes: np.ones(3)}
    )
    with pytest.raises(ValueError, match="returned shape"):
        bad(field_jet(field_grid(6, 5)))
    not_a_measure = measure_integral_columns(lambda points: object(), {"c3": cos3_kernel})
    with pytest.raises(TypeError, match="must be an omnibias.measure.Measure"):
        not_a_measure(field_jet(field_grid(6, 5)))
