# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Randomized brute-force soundness sweep for the substrate gap certificate.

The claim a gap certificate makes is ``lower_bound <= OPT <= energy``. Only the
first half can be got wrong quietly: ``energy`` is the value of an assignment we
actually hold, so it is an upper bound by construction, whereas ``lower_bound``
comes from a relaxation and is a genuine assertion about a quantity nobody
evaluated.

So the sweep compares ``lower_bound`` to an exhaustive ``2^n`` oracle with **no
tolerance at all**. A lower bound that exceeds the true minimum is false however
narrowly it does so, and tolerance-carrying assertions are exactly what let two
other unsound certificate paths in this repo stay green.
"""

from __future__ import annotations

import numpy as np
import pytest
from _enclosure import assert_lower_bound
from omnibias.discrete import brute_force_min, certify_gap, decode


def _instance(make_toy, seed: int, *, n_max: int = 6):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, n_max + 1))
    m = rng.standard_normal((n, n))
    return make_toy(m + m.T, c=rng.standard_normal(n), const=0.3)


def test_negative_coeff_lower_bound_never_exceeds_the_true_minimum(make_toy) -> None:  # type: ignore[no-untyped-def]
    """The always-available back-stop bound, swept without SOS in the loop."""
    for seed in range(120):
        problem = _instance(make_toy, seed)
        _, e_min = brute_force_min(problem)
        assignment, _ = decode(problem, seed=seed, n_starts=16)
        cert = certify_gap(problem, assignment, level=0)
        assert_lower_bound(cert.lower_bound, e_min, what=f"minimum energy (seed={seed})")


@pytest.mark.slow
def test_sos_lower_bound_never_exceeds_the_true_minimum(make_toy) -> None:  # type: ignore[no-untyped-def]
    """The level-1 Lasserre / SOS bound. Slow (a bisection per instance)."""
    pytest.importorskip("omnibias.sos")
    for seed in range(60):
        problem = _instance(make_toy, seed)
        _, e_min = brute_force_min(problem)
        assignment, _ = decode(problem, seed=seed, n_starts=16)
        cert = certify_gap(problem, assignment, level=1, bisection_steps=20)
        assert_lower_bound(cert.lower_bound, e_min, what=f"minimum energy (seed={seed})")


def test_sos_lower_bound_smoke(make_toy) -> None:  # type: ignore[no-untyped-def]
    """A cheap always-on slice of the sweep above, so the SOS path is never unwatched."""
    pytest.importorskip("omnibias.sos")
    for seed in range(6):
        problem = _instance(make_toy, seed, n_max=4)
        _, e_min = brute_force_min(problem)
        assignment, _ = decode(problem, seed=seed, n_starts=8)
        cert = certify_gap(problem, assignment, level=1, bisection_steps=12)
        assert_lower_bound(cert.lower_bound, e_min, what=f"minimum energy (seed={seed})")


def test_decoded_energy_is_attained_and_matches_the_oracle(make_toy) -> None:  # type: ignore[no-untyped-def]
    """``energy`` must be a real assignment's value, hence never below ``OPT``.

    The comparison needs a relative tolerance for a reason unrelated to soundness:
    ``brute_force_min`` evaluates all ``2^n`` points in one batched matmul, whose
    reduction order differs from the single-vector path, so the same assignment can
    come back a few ulp apart. That is a floating-point reproducibility allowance on
    two computations of *the same* quantity -- not slack on a bound, which is why it
    is scaled to the magnitude involved instead of being a fixed absolute constant.
    """
    for seed in range(120):
        problem = _instance(make_toy, seed)
        _, e_min = brute_force_min(problem)
        assignment, _ = decode(problem, seed=seed, n_starts=16)
        cert = certify_gap(problem, assignment, level=0)
        ulp = 64.0 * np.finfo(float).eps * max(abs(e_min), 1.0)
        assert cert.energy >= e_min - ulp, (
            f"seed={seed}: decoded energy {cert.energy!r} is below the exhaustive "
            f"minimum {e_min!r} by more than {ulp!r} -- the oracle missed the optimum"
        )
        assert cert.energy == pytest.approx(
            float(problem.energy(np.asarray(assignment, dtype=float)))
        )
