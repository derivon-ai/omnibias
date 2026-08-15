# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for the Mathlib-backed bridge (``omnibias.formal.mathlib_check``).

These run with or without a Lean toolchain: obligation generation, the tamper
path, and graceful degradation are always exercised; the actual ``lake`` build
(and ``verified=True``) is asserted only when ``lake`` + the analytic checkout
are present (the dedicated ``lean-analytic`` CI job).
"""

from __future__ import annotations

from omnibias.core.proof.certificate import (
    interval_certificate,
    make_certificate,
    positive_definite_certificate,
    taylor_model_certificate,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import radii_polynomial_certificate
from omnibias.core.verified.taylor_model import TaylorModel
from omnibias.formal import (
    MATHLIB_CLAIM_KEY,
    analytic_root,
    check_certificate,
    enclosure_trace_certificate,
    generate_obligation,
    mathlib_check_available,
    nk_existence_certificate,
    tower_coeffs_certificate,
)


def test_claim_key_is_distinct_from_theorem_prover_verified() -> None:
    # The whole point of the tier: it must never be conflated with the
    # minimal-kernel flag.
    assert MATHLIB_CLAIM_KEY == "mathlib_verified"
    assert MATHLIB_CLAIM_KEY != "theorem_prover_verified"


def test_analytic_root_is_discoverable() -> None:
    root = analytic_root()
    assert root is not None
    assert root.name == "omnibias-analytic"
    assert (root / "lakefile.lean").is_file()
    assert (root / "OmnibiasAnalytic" / "Check" / "EnclosedSign.lean").is_file()


def test_generate_positive_interval_obligation() -> None:
    cert = interval_certificate("q", Interval(0.5, 2.0))
    src = generate_obligation(cert)
    assert src is not None
    assert "OmnibiasAnalytic.Check.enclosed_pos" in src
    assert ": 0 < x" in src
    assert "by norm_num" in src
    assert "namespace OmnibiasAnalytic.Generated" in src
    # 0.5 is emitted as the exact rational 1/2, not a float.
    assert "(1 : \u211a) / 2" in src
    assert "0.5" not in src


def test_generate_negative_interval_obligation() -> None:
    cert = interval_certificate("q", Interval(-2.0, -0.5))
    src = generate_obligation(cert)
    assert src is not None
    assert "OmnibiasAnalytic.Check.enclosed_neg" in src
    assert ": x < 0" in src
    assert "by norm_num" in src


def test_generate_integer_endpoint_obligation() -> None:
    cert = interval_certificate("q", Interval(2.0, 4.0))
    src = generate_obligation(cert)
    assert src is not None
    # An integer lower bound renders without a denominator.
    assert "(2 : \u211a)" in src
    assert "enclosed_pos" in src


def test_generate_returns_none_for_straddling_interval() -> None:
    # An interval that straddles zero carries no sign obligation.
    assert generate_obligation(interval_certificate("q", Interval(-1.0, 1.0))) is None


def test_generate_returns_none_for_unsupported() -> None:
    assert generate_obligation({"foo": "bar"}) is None


# --------------------------------------------------------------------------- #
# Positive-definite (ℚ pivots).
# --------------------------------------------------------------------------- #
def test_generate_positive_definite_obligation() -> None:
    cert = positive_definite_certificate("pd", [Interval(1.0, 1.5), Interval(0.5, 0.75)])
    src = generate_obligation(cert)
    assert src is not None
    assert "0 < (1 : \u211a)" in src  # first pivot lower endpoint
    assert "0 < ((1 : \u211a) / 2)" in src  # second pivot lower endpoint (0.5 == 1/2)
    assert "refine" in src and "norm_num" in src
    assert "positive-definite" in src.lower()


def test_positive_definite_with_nonpositive_pivot_yields_none() -> None:
    # A pivot whose lower endpoint is <= 0 cannot certify positive-definiteness.
    cert = positive_definite_certificate("pd", [Interval(1.0, 1.5), Interval(-0.1, 0.2)])
    assert generate_obligation(cert) is None


# --------------------------------------------------------------------------- #
# Newton-Kantorovich radii polynomial.
# --------------------------------------------------------------------------- #
def test_generate_radii_polynomial_obligation() -> None:
    rc = radii_polynomial_certificate(0.001, 0.05, 0.0, 0.05)
    assert rc is not None  # these bounds admit a contracting radius
    src = generate_obligation(rc.certificate)
    assert src is not None
    assert "^ 2" in src  # the r^2 term of the radii polynomial
    assert "< 0" in src  # p(r) < 0
    assert "< 1" in src  # kappa < 1
    assert "ExistsUnique" not in src and "∃!" not in src
    assert "refine" in src and "norm_num" in src
    assert "0x" not in src and "e-" not in src  # exact rationals, never a float literal


def test_radii_polynomial_that_does_not_contract_yields_none() -> None:
    # A hand-forged payload whose bounds do not actually contract must be refused
    # (the bridge re-derives p(r) and kappa exactly and declines to emit a false fact).
    cert = make_certificate(
        claim="bogus",
        payload={
            "type": "radii_polynomial",
            "radius": 1.0,
            "kappa": 0.5,
            "p_value": -1.0,
            "Y0": 5.0,  # p(1) = 5 + ... - 1 > 0: not negative
            "Z0": 0.1,
            "Z1": 0.0,
            "Z2": 0.1,
        },
    )
    assert generate_obligation(cert) is None


# --------------------------------------------------------------------------- #
# Krawczyk unique-zero box.
# --------------------------------------------------------------------------- #
def _krawczyk_cert(kappa: float = 0.25) -> dict:
    return make_certificate(
        claim="unique zero of F in the Krawczyk box",
        payload={
            "type": "krawczyk",
            "radius": 0.5,
            "kappa": kappa,
            "center": [1.0, 2.0],
            "enclosure": [[0.9, 1.1], [1.8, 2.2]],
        },
    )


def test_generate_krawczyk_obligation() -> None:
    src = generate_obligation(_krawczyk_cert())
    assert src is not None
    assert "< 1" in src  # kappa < 1
    assert "-" in src  # center - r < lo terms
    assert "refine" in src and "norm_num" in src
    assert "0.5" not in src  # radius 0.5 emitted as the exact rational 1/2
    assert "ExistsUnique" not in src and "∃!" not in src


def test_krawczyk_not_contracting_yields_none() -> None:
    assert generate_obligation(_krawczyk_cert(kappa=1.5)) is None


def test_krawczyk_image_not_strictly_inside_yields_none() -> None:
    cert = make_certificate(
        claim="bogus",
        payload={
            "type": "krawczyk",
            "radius": 0.5,
            "kappa": 0.25,
            "center": [1.0],
            "enclosure": [[0.4, 1.1]],  # lo=0.4 is NOT > center - r = 0.5
        },
    )
    assert generate_obligation(cert) is None


# --------------------------------------------------------------------------- #
# Taylor-model centre-value sign.
# --------------------------------------------------------------------------- #
def test_generate_taylor_model_positive_centre() -> None:
    tm = TaylorModel(0.0, 1.0, [Interval(2.0, 2.5), Interval(-1.0, 1.0)], Interval(-0.1, 0.1))
    src = generate_obligation(taylor_model_certificate("tm", tm))
    assert src is not None
    # centre value in coeffs[0] + remainder = [1.9, 2.6] > 0.
    assert "OmnibiasAnalytic.Check.enclosed_pos" in src


def test_generate_taylor_model_negative_centre() -> None:
    tm = TaylorModel(0.0, 1.0, [Interval(-2.5, -2.0), Interval(-1.0, 1.0)], Interval(-0.1, 0.1))
    src = generate_obligation(taylor_model_certificate("tm", tm))
    assert src is not None
    assert "OmnibiasAnalytic.Check.enclosed_neg" in src


def test_taylor_model_straddling_centre_yields_none() -> None:
    tm = TaylorModel(0.0, 1.0, [Interval(-0.5, 0.5)], Interval(-0.1, 0.1))
    assert generate_obligation(taylor_model_certificate("tm", tm)) is None


# --------------------------------------------------------------------------- #
# PINN a-posteriori finite margin.
# --------------------------------------------------------------------------- #
def test_generate_pinn_margin_obligation() -> None:
    cert = make_certificate(
        claim="a-posteriori sup-norm error bound",
        payload={
            "type": "pinn_aposteriori_error",
            "error_bound": 0.01,
            "finite_obligation": {
                "type": "error_bound_le_threshold",
                "threshold": 0.05,
                "margin": [0.04, 0.04],  # threshold - error_bound > 0
            },
        },
    )
    src = generate_obligation(cert)
    assert src is not None
    # A positive margin (error below threshold) is an enclosed-positive obligation.
    assert "OmnibiasAnalytic.Check.enclosed_pos" in src


# --------------------------------------------------------------------------- #
# Tower coefficients (exact integer recurrence).
# --------------------------------------------------------------------------- #
def test_generate_tower_coeffs_obligation() -> None:
    cert = tower_coeffs_certificate("sigmoid", 2)
    src = generate_obligation(cert)
    assert src is not None
    assert "import OmnibiasAnalytic.Tower" in src
    assert "sigmoidCoeffList 2 = [0, 1, -3, 2]" in src
    assert "native_decide" in src
    assert "collapse" not in src.lower() or "not a collapse" in src


def test_tower_coeffs_mismatch_yields_none() -> None:
    cert = make_certificate(
        claim="bogus",
        payload={
            "type": "tower_coeffs",
            "family": "sigmoid",
            "n": 2,
            "coeffs": [0, 1, 0, 0],
        },
    )
    assert generate_obligation(cert) is None


def test_tower_coeffs_unknown_family_yields_none() -> None:
    cert = make_certificate(
        claim="bogus",
        payload={"type": "tower_coeffs", "family": "mish", "n": 0, "coeffs": [1]},
    )
    assert generate_obligation(cert) is None


def test_generate_nk_existence_obligation() -> None:
    cert = nk_existence_certificate("radii")
    src = generate_obligation(cert)
    assert src is not None
    assert "import OmnibiasAnalytic.Check.Kantorovich.Plant" in src
    assert "quadratic_plant_radii_unique_zero" in src
    assert "quadraticPlant" in src
    assert "Icc (5 / 4) (7 / 4)" in src


def test_generate_nk_existence_krawczyk_route() -> None:
    cert = nk_existence_certificate("krawczyk")
    src = generate_obligation(cert)
    assert src is not None
    assert "quadratic_plant_krawczyk_unique_zero" in src


def test_nk_existence_mismatch_yields_none() -> None:
    cert = make_certificate(
        claim="bogus",
        payload={
            "type": "nk_existence",
            "family": "quadratic",
            "route": "radii",
            "center": [3, 2],
            "radius": [1, 4],
            "A": [1, 3],
            "Y0": [1, 11],
            "Z0": [0, 1],
            "Z1": [0, 1],
            "Z2": [2, 3],
        },
    )
    assert generate_obligation(cert) is None


def test_nk_existence_unknown_route_yields_none() -> None:
    cert = make_certificate(
        claim="bogus",
        payload={
            "type": "nk_existence",
            "family": "quadratic",
            "route": "dottie",
            "center": [3, 2],
            "radius": [1, 4],
            "A": [1, 3],
            "Y0": [1, 12],
            "Z0": [0, 1],
            "Z1": [0, 1],
            "Z2": [2, 3],
        },
    )
    assert generate_obligation(cert) is None


def test_nk_existence_unknown_family_yields_none() -> None:
    cert = make_certificate(
        claim="bogus",
        payload={
            "type": "nk_existence",
            "family": "dottie",
            "route": "radii",
            "center": [3, 2],
            "radius": [1, 4],
            "A": [1, 3],
            "Y0": [1, 12],
            "Z0": [0, 1],
            "Z1": [0, 1],
            "Z2": [2, 3],
        },
    )
    assert generate_obligation(cert) is None


def test_generate_enclosure_trace_obligation() -> None:
    cert = enclosure_trace_certificate("nk")
    src = generate_obligation(cert)
    assert src is not None
    assert "import OmnibiasAnalytic.Check.Enclosure.Plant" in src
    assert "open OmnibiasAnalytic.Check QInterval Set" in src
    assert "nk_trace_unique_zero" in src
    assert "evalTrace nkBoundOps" in src
    assert "quadraticPlant" in src


def test_generate_enclosure_trace_other_families() -> None:
    assert "tower_horner_result" in (generate_obligation(enclosure_trace_certificate("tower")) or "")
    assert "bernoulli_b2_zetaNeg1" in (
        generate_obligation(enclosure_trace_certificate("bernoulli")) or ""
    )
    assert "ldlt_plant_pivots_pos" in (
        generate_obligation(enclosure_trace_certificate("ldlt")) or ""
    )


def test_enclosure_trace_mismatch_yields_none() -> None:
    cert = enclosure_trace_certificate("tower")
    payload = dict(cert["payload"])
    payload["result"] = {"lo": [1, 1], "hi": [1, 1]}
    bogus = make_certificate(claim="bogus", payload=payload)
    assert generate_obligation(bogus) is None


def test_enclosure_trace_unknown_family_yields_none() -> None:
    cert = make_certificate(
        claim="bogus",
        payload={
            "type": "enclosure_trace",
            "family": "dottie",
            "ops": [],
            "result": {"lo": [0, 1], "hi": [0, 1]},
        },
    )
    assert generate_obligation(cert) is None


def test_check_certificate_enclosure_trace_nk() -> None:
    cert = enclosure_trace_certificate("nk")
    result = check_certificate(cert)
    assert "nk_trace_unique_zero" in result.obligation
    if not mathlib_check_available():
        assert result.available is False
        assert result.verified is False
    else:  # pragma: no cover - only on a machine with Lean + Mathlib
        assert result.available is True
        assert result.verified is True


def test_check_certificate_nk_existence_radii() -> None:
    cert = nk_existence_certificate("radii")
    result = check_certificate(cert)
    assert "quadratic_plant_radii_unique_zero" in result.obligation
    if not mathlib_check_available():
        assert result.available is False
        assert result.verified is False
    else:  # pragma: no cover - only on a machine with Lean + Mathlib
        assert result.available is True
        assert result.verified is True


def test_check_certificate_tower_sigmoid_two() -> None:
    cert = tower_coeffs_certificate("sigmoid", 2)
    result = check_certificate(cert)
    assert "sigmoidCoeffList 2" in result.obligation
    if not mathlib_check_available():
        assert result.available is False
        assert result.verified is False
    else:  # pragma: no cover - only on a machine with Lean + Mathlib
        assert result.available is True
        assert result.verified is True


def test_check_certificate_generates_radii_obligation_without_toolchain() -> None:
    rc = radii_polynomial_certificate(0.001, 0.05, 0.0, 0.05)
    assert rc is not None
    result = check_certificate(rc.certificate)
    assert "norm_num" in result.obligation
    if not mathlib_check_available():
        assert result.available is False
        assert result.verified is False


def test_tampered_certificate_is_rejected_before_lean() -> None:
    cert = interval_certificate("q", Interval(0.5, 2.0))
    tampered = dict(cert)
    tampered["payload"] = {**cert["payload"], "tampered": True}  # body changed, digest stale
    result = check_certificate(tampered)
    assert result.verified is False
    assert "digest" in result.detail


def test_check_certificate_graceful_without_toolchain() -> None:
    cert = interval_certificate("q", Interval(0.5, 2.0))
    result = check_certificate(cert)
    # Whatever the environment, the obligation is generated and no exception is raised.
    assert "enclosed_pos" in result.obligation
    if not mathlib_check_available():
        assert result.available is False
        assert result.verified is False
    else:  # pragma: no cover - only on a machine with Lean + Mathlib
        assert result.available is True
        assert result.verified is True
