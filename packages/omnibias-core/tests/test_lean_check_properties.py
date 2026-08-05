# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Property-based sweeps for the Lean-kernel bridge obligation generator.

Isolated in its own module so the (dev-only) ``hypothesis`` dependency is skipped
gracefully in the clean per-package CI venvs, while the deterministic bridge
tests in ``test_lean_check.py`` always run.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from omnibias.core.proof import generate_obligation  # noqa: E402
from omnibias.core.proof.certificate import (  # noqa: E402
    interval_certificate,
    positive_definite_certificate,
)
from omnibias.core.proof.lean_check import _scaled_pair  # noqa: E402
from omnibias.core.verified.interval import Interval  # noqa: E402


@given(st.floats(min_value=0.0, max_value=1.0, exclude_max=True, allow_nan=False))
def test_spectral_gap_obligation_is_always_true(ratio: float) -> None:
    src = generate_obligation({"subdominant_ratio_upper": ratio})
    assert src is not None
    frac = Fraction(ratio)
    # The emitted obligation `0 < gapNumerator rn rd` is true exactly because
    # rn < rd (and rd > 0): the kernel will accept it.
    assert frac.numerator < frac.denominator
    assert frac.denominator > 0
    assert f"gapNumerator {frac.numerator} {frac.denominator}" in src


@given(
    st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_positive_interval_scaling_preserves_sign(lo: float, width: float) -> None:
    hi = lo + width
    ilo, ihi = _scaled_pair(lo, hi)
    # Scaling by a common positive denominator preserves order and sign.
    assert ilo > 0
    assert ilo <= ihi
    src = generate_obligation(interval_certificate("q", Interval(lo, hi)))
    assert src is not None
    assert "enclosed_quantity_pos" in src


@given(
    st.floats(min_value=-1e6, max_value=-1e-6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_negative_interval_scaling_preserves_sign(hi: float, width: float) -> None:
    lo = hi - width
    ilo, ihi = _scaled_pair(lo, hi)
    assert ihi < 0
    assert ilo <= ihi
    src = generate_obligation(interval_certificate("q", Interval(lo, hi)))
    assert src is not None
    assert "enclosed_quantity_neg" in src


@given(
    st.lists(
        st.tuples(
            st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=6,
    )
)
def test_positive_definite_pivots_yield_allpivotspos(pivots: list[tuple[float, float]]) -> None:
    # Every pivot has a strictly positive lower endpoint -> a well-formed allPivotsPos obligation
    # whose per-pivot scaled lower endpoints are all positive (kernel `decide` will accept it).
    intervals = [Interval(lo, lo + width) for lo, width in pivots]
    for iv in intervals:
        ilo, _ihi = _scaled_pair(iv.lo, iv.hi)
        assert ilo > 0
    src = generate_obligation(positive_definite_certificate("pd", intervals))
    assert src is not None
    assert "allPivotsPos" in src
    assert src.count("⟨") == len(intervals)
