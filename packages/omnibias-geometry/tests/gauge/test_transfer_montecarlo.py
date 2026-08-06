# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The validation headline: a certified gap against a Monte Carlo of the same matrix.

The comparison is only meaningful because both sides describe *one fixed matrix*.
The certificate bounds ``-ln(lambda_1 / lambda_0)`` by interval arithmetic; the
Monte Carlo estimates the same number from the decay of a sampled autocorrelation
of the path measure ``prod_t T_{x_t, x_{t+1}}`` that this very matrix defines.
No ensemble-versus-matrix correspondence is assumed anywhere.
"""

from __future__ import annotations

import math

import pytest
from omnibias.geometry.gauge.transfer import (
    certified_gap_versus_monte_carlo,
    certified_transfer_matrix_gap,
    sample_transfer_path_ensemble,
    su2_class_angle_transfer,
    su2_heat_kernel_transfer,
    u1_heat_kernel_transfer,
)
from omnibias.geometry.gauge.transfer.montecarlo import (
    _connected_correlator,
    _default_operators,
    _jackknife_error,
    _supported_start,
)

SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- #
# The class-angle matrix: dense and positive, yet isospectral to the character one
# --------------------------------------------------------------------------- #
def test_the_class_angle_matrix_is_isospectral_to_the_character_basis() -> None:
    """Same operator, different basis: the closed-form spectrum must be untouched."""
    character = su2_heat_kernel_transfer(0.8, max_dynkin=3)
    angle = su2_class_angle_transfer(0.8, max_dynkin=3)

    assert character.exact_eigenvalues is not None
    assert angle.exact_eigenvalues is not None
    assert len(angle.exact_eigenvalues) == len(character.exact_eigenvalues)
    for got, want in zip(angle.exact_eigenvalues, character.exact_eigenvalues, strict=True):
        assert got.lo == pytest.approx(want.lo, rel=1e-15)
        assert got.hi == pytest.approx(want.hi, rel=1e-15)


def test_the_class_angle_matrix_is_dense_positive_and_symmetric() -> None:
    """Everything the path measure and Birkhoff-Hopf need, unlike the diagonal twin."""
    angle = su2_class_angle_transfer(0.8, max_dynkin=3)
    assert angle.entrywise_positive
    assert angle.symmetric

    off_diagonal = [
        cell
        for i, row in enumerate(angle.entries)
        for j, cell in enumerate(row)
        if i != j
    ]
    assert all(cell.lo > 0.0 for cell in off_diagonal)

    for i, row in enumerate(angle.entries):
        for j, cell in enumerate(row):
            mirror = angle.entries[j][i]
            assert cell.lo == pytest.approx(mirror.lo, abs=1e-15)
            assert cell.hi == pytest.approx(mirror.hi, abs=1e-15)


def test_the_character_basis_cannot_be_sampled_which_is_why_the_angle_basis_exists() -> None:
    r"""A diagonal transfer matrix's path measure lives entirely on *constant* paths.

    ``prod_t T_{x_t, x_{t+1}}`` vanishes unless every ``x_t`` agrees, so the chain
    freezes, the correlator is flat and no mass can be measured.  That is the whole
    reason :func:`su2_class_angle_transfer` exists.
    """
    character = su2_heat_kernel_transfer(0.8, max_dynkin=3)
    ensemble = sample_transfer_path_ensemble(
        character, chain_length=8, n_samples=4, n_sweeps=4, n_thermalize=200, seed=0
    )
    for sample in ensemble.observables:
        for row in sample:
            assert len(set(row)) == 1, "a diagonal transfer matrix must freeze the chain"

    curves = _connected_correlator(ensemble)
    mean = [sum(c[t] for c in curves) / len(curves) for t in range(len(curves[0]))]
    assert all(value == pytest.approx(mean[0], abs=1e-12) for value in mean)


def test_a_matrix_with_a_negative_entry_is_refused_rather_than_sampled() -> None:
    """A path measure needs non-negative weights; truncation can break that at small t."""
    truncated = su2_class_angle_transfer(0.05, max_dynkin=5)
    assert not truncated.entrywise_positive
    with pytest.raises(ValueError, match="non-negative weights"):
        sample_transfer_path_ensemble(truncated, chain_length=8, n_samples=4, seed=0)


# --------------------------------------------------------------------------- #
# The headline
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
def test_the_certified_bound_sits_below_the_monte_carlo_estimate(seed: int) -> None:
    """The plan's headline, on one small explicit-transfer-matrix su(2) model."""
    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    check = certified_gap_versus_monte_carlo(transfer, seed=seed)

    assert check.consistent, check.detail
    assert check.certified_gap_lower > 0.0
    assert math.isfinite(check.monte_carlo_mass)
    assert math.isfinite(check.monte_carlo_error)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_monte_carlo_brackets_the_exact_gap_the_two_sided_test(seed: int) -> None:
    """The check with teeth: the estimate must agree with closed-form truth *both* ways.

    ``consistent`` alone is one-sided and the estimator's residual bias helps it
    pass.  This one fails if the sampler, the matrix, or the identification of
    "gap" is wrong in either direction.
    """
    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    check = certified_gap_versus_monte_carlo(transfer, seed=seed)

    assert check.exact_gap is not None
    assert check.agrees_with_exact, (
        f"MC {check.monte_carlo_mass:.4f} +/- {check.monte_carlo_error:.4f} "
        f"vs exact {check.exact_gap:.4f}"
    )


def test_the_certified_bound_is_exactly_tight_on_this_model() -> None:
    """With the exact eigenvectors available, the partner chain loses nothing."""
    coupling = 0.8
    transfer = su2_class_angle_transfer(coupling, max_dynkin=3)
    result = certified_transfer_matrix_gap(transfer)

    # su(2) gap: C2(1) - C2(0) = 3/4, so m a = 3t/4 exactly.
    assert result.spectral_gap_lower == pytest.approx(0.75 * coupling, rel=1e-9)


@pytest.mark.parametrize(
    ("build", "exact"),
    [
        (lambda: su2_class_angle_transfer(0.6, max_dynkin=3), 0.45),
        (lambda: su2_class_angle_transfer(1.0, max_dynkin=3), 0.75),
        (lambda: su2_class_angle_transfer(0.8, max_dynkin=5), 0.60),
        (lambda: u1_heat_kernel_transfer(0.5, n_max=3, basis="angle"), 0.50),
    ],
)
def test_the_cross_check_holds_across_models_and_couplings(build, exact: float) -> None:
    """Not a single tuned point: the same statement across models, couplings, sizes."""
    transfer = build()
    for seed in SEEDS:
        check = certified_gap_versus_monte_carlo(transfer, seed=seed)
        assert check.exact_gap == pytest.approx(exact, rel=1e-9)
        assert check.consistent, check.detail
        assert check.agrees_with_exact, check.detail


def test_the_estimate_is_reproducible_from_its_seed() -> None:
    """A seeded harness that drifted would make every number above unfalsifiable."""
    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    first = certified_gap_versus_monte_carlo(transfer, seed=7)
    second = certified_gap_versus_monte_carlo(transfer, seed=7)
    assert first.monte_carlo_mass == second.monte_carlo_mass
    assert first.monte_carlo_error == second.monte_carlo_error


# --------------------------------------------------------------------------- #
# The pieces that make the headline trustworthy
# --------------------------------------------------------------------------- #
def test_the_default_operator_is_the_ratio_that_kills_higher_mode_contamination() -> None:
    r"""``v_1 / v_0``, which for the class-angle matrix is the fundamental character."""
    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    (operator,) = _default_operators(transfer)

    expected = tuple(
        v1 / v0
        for v1, v0 in zip(
            transfer.subdominant_vectors[0], transfer.perron_vector, strict=True
        )
    )
    assert operator == pytest.approx(expected, rel=1e-12)

    # sin(2 theta) / sin(theta) = 2 cos(theta): the su(2) fundamental character.
    n_points = transfer.dimension
    characters = tuple(
        2.0 * math.cos(math.pi * j / (n_points + 1)) for j in range(1, n_points + 1)
    )
    assert operator == pytest.approx(characters, rel=1e-12)


def test_the_ratio_operator_has_exactly_zero_overlap_with_every_higher_mode() -> None:
    r"""Why ``v_1 / v_0`` and not ``v_1``: an exact statement, not a statistical one.

    Under the path measure an operator overlaps mode ``k`` by
    ``sum_x v_0[x] O[x] v_k[x]``.  For ``O = v_1 / v_0`` that collapses to the
    orthogonality relation ``sum_x v_1[x] v_k[x] = delta_{1k}``, so contamination
    from higher modes is *zero*, not merely small.  The naive ``O = v_1`` keeps a
    sizeable overlap with mode 3 here.
    """
    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    modes = (transfer.perron_vector, *transfer.subdominant_vectors)
    perron = transfer.perron_vector

    def overlaps(operator: tuple[float, ...]) -> list[float]:
        raw = [
            sum(perron[x] * operator[x] * mode[x] for x in range(transfer.dimension))
            for mode in modes
        ]
        scale = max(abs(v) for v in raw)
        return [v / scale for v in raw]

    (ratio,) = _default_operators(transfer)
    naive = tuple(float(v) for v in transfer.subdominant_vectors[0])

    ratio_overlaps = overlaps(ratio)
    assert ratio_overlaps[1] == pytest.approx(1.0, abs=1e-12)
    for k, value in enumerate(ratio_overlaps):
        if k != 1:
            assert value == pytest.approx(0.0, abs=1e-12)

    naive_overlaps = overlaps(naive)
    assert naive_overlaps[1] == pytest.approx(1.0, abs=1e-12)
    assert max(abs(v) for k, v in enumerate(naive_overlaps) if k != 1) > 0.1


def test_the_jackknife_error_tracks_the_honest_seed_to_seed_scatter() -> None:
    """The error bar has to be believable, or ``agrees_with_exact`` proves nothing.

    Guards the specific bug this estimator had: feeding pre-formed replicas to a
    raw-sample jackknife helper understated the error by about ``sqrt(n)``, which
    made a wrong estimate look like a precise one.
    """
    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    checks = [certified_gap_versus_monte_carlo(transfer, seed=s) for s in range(8)]
    masses = [c.monte_carlo_mass for c in checks]
    quoted = sum(c.monte_carlo_error for c in checks) / len(checks)

    mean = sum(masses) / len(masses)
    scatter = math.sqrt(sum((m - mean) ** 2 for m in masses) / (len(masses) - 1))
    assert 0.2 < quoted / scatter < 5.0, f"quoted {quoted:.4f} vs scatter {scatter:.4f}"


def test_the_jackknife_helper_uses_the_replica_formula() -> None:
    """Checked against a closed-form case: for the mean, jackknife == standard error."""
    curves = [[float(i), 0.0] for i in range(10)]
    error = _jackknife_error(curves, lambda rows: sum(r[0] for r in rows) / len(rows))

    values = [c[0] for c in curves]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    assert error == pytest.approx(math.sqrt(variance / len(values)), rel=1e-12)


def test_the_chain_starts_inside_the_support_of_the_path_measure() -> None:
    r"""Guards a real sampler bug: a random start can have probability *zero*.

    Heat-bath updates cannot leave a zero-weight configuration -- a site flanked by
    two states it has no weight to bridge never moves again -- so seeding at random
    silently produced configurations outside the measure and biased the mass.
    """
    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    weights = [[0.5 * (c.lo + c.hi) for c in row] for row in transfer.entries]
    start = _supported_start(weights)
    assert weights[start][start] > 0.0
    assert weights[start][start] == max(weights[i][i] for i in range(len(weights)))


def test_the_sampler_reads_only_matrix_entries_not_the_spectrum() -> None:
    """The oracle's independence: strip the spectrum and the estimate is unchanged."""
    from dataclasses import replace

    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    blinded = replace(transfer, exact_eigenvalues=None)

    full = sample_transfer_path_ensemble(transfer, seed=3)
    stripped = sample_transfer_path_ensemble(blinded, seed=3)
    assert full.observables == stripped.observables


def test_a_wrong_matrix_is_caught_which_is_what_makes_this_falsifiable() -> None:
    """Pair a certified bound with a Monte Carlo of a *different* matrix: it must fail.

    Without this, ``consistent`` passing everywhere would be uninformative.
    """
    honest = su2_class_angle_transfer(0.4, max_dynkin=3)
    stiff = certified_transfer_matrix_gap(su2_class_angle_transfer(1.6, max_dynkin=3))

    check = certified_gap_versus_monte_carlo(honest, seed=0, gap=stiff)
    assert not check.consistent
    assert "EXCEEDS" in check.detail


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"chain_length": 2}, "chain_length"),
        ({"n_samples": 1}, "n_samples"),
    ],
)
def test_the_sampler_refuses_settings_that_cannot_produce_an_error_bar(
    kwargs: dict[str, int], match: str
) -> None:
    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    with pytest.raises(ValueError, match=match):
        sample_transfer_path_ensemble(transfer, **kwargs)


def test_an_operator_of_the_wrong_length_is_refused() -> None:
    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    with pytest.raises(ValueError, match="one value per state"):
        sample_transfer_path_ensemble(
            transfer, chain_length=8, n_samples=4, operators=((1.0, 2.0),)
        )
