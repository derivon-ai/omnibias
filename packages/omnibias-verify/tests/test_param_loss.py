# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Proof-carrying training: certify a trained ``theta*`` is a strict local minimum.

Coverage:

* **enclosure soundness** (repo rule) -- the interval parameter-space gradient / Hessian must
  contain the true values on a dense deterministic grid *and* random samples inside the box; the
  reference is an *independent* plain-float finite-difference implementation (no intervals, no
  hyper-duals), so this validates the derivative formulae as well as the enclosure;
* **positive** -- a realizable teacher fit is certified a locally-unique, strict (PD-Hessian) local
  minimum, the sealed certificate's digest verifies, and ``eig_min.lo > 0``;
* **negative controls** -- a non-stationary point and a too-wide box are both correctly refused;
* **tamper evidence** -- forging a sealed bound breaks the digest;
* **Lean obligation** -- the certificate exposes a well-formed ``eig_min > 0`` obligation, and the
  optional kernel gate degrades gracefully when no Lean toolchain is present.
"""

from __future__ import annotations

import copy
import math
import random

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.proof.lean_check import generate_obligation, lean_check_available
from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    MLPArchitecture,
    ParamSpaceLoss,
    TrainingCertificate,
    certify_trained_min,
    flat_params_from_layers,
)

# --------------------------------------------------------------------------- #
# Independent plain-float reference (no intervals) for the soundness checks.
# --------------------------------------------------------------------------- #
Data = list[tuple[tuple[float, ...], tuple[float, ...]]]


def _net_scalar(theta: list[float], x: float) -> float:
    """The 1-1-1 tanh net ``v * tanh(w x + b) + c`` in the flat layout ``[w, b, v, c]``."""
    w, b, v, c = theta
    return v * math.tanh(w * x + b) + c


def _loss_float(theta: list[float], data: Data) -> float:
    return sum((_net_scalar(theta, x[0]) - y[0]) ** 2 for x, y in data) / len(data)


def _fd_grad(theta: list[float], data: Data, h: float = 1e-5) -> list[float]:
    g = []
    for i in range(len(theta)):
        tp, tm = list(theta), list(theta)
        tp[i] += h
        tm[i] -= h
        g.append((_loss_float(tp, data) - _loss_float(tm, data)) / (2.0 * h))
    return g


def _fd_hess(theta: list[float], data: Data, h: float = 1e-3) -> list[list[float]]:
    n = len(theta)
    f0 = _loss_float(theta, data)
    hess = [[0.0] * n for _ in range(n)]
    for i in range(n):
        tp, tm = list(theta), list(theta)
        tp[i] += h
        tm[i] -= h
        hess[i][i] = (_loss_float(tp, data) - 2.0 * f0 + _loss_float(tm, data)) / (h * h)
    for i in range(n):
        for j in range(i + 1, n):
            def _shift(si: float, sj: float, i: int = i, j: int = j) -> float:
                t = list(theta)
                t[i] += si
                t[j] += sj
                return _loss_float(t, data)

            val = (_shift(h, h) - _shift(h, -h) - _shift(-h, h) + _shift(-h, -h)) / (4.0 * h * h)
            hess[i][j] = hess[j][i] = val
    return hess


# A realizable, reasonably well-conditioned teacher fit on a fixed data grid: theta* = teacher is a
# global minimum (loss 0) with an isolated, positive-definite Hessian (no continuous symmetry for a
# single tanh unit), so it is a genuine strict local minimum.
ARCH = MLPArchitecture(dims=(1, 1, 1), activation="tanh")
TEACHER = [-1.3, 0.25, 1.2, 0.1]
GRID = [(-2.0 + 4.0 * i / 11.0) for i in range(12)]
DATA: Data = [((x,), (_net_scalar(TEACHER, x),)) for x in GRID]


# --------------------------------------------------------------------------- #
# architecture + layout helpers
# --------------------------------------------------------------------------- #
def test_architecture_shapes_and_param_count() -> None:
    arch = MLPArchitecture(dims=(2, 3, 1), activation="tanh")
    assert arch.layer_shapes == [(3, 2), (1, 3)]
    assert arch.n_params == (3 * 2 + 3) + (1 * 3 + 1)  # 9 + 4 = 13
    assert arch.out_dim == 1


def test_architecture_rejects_bad_spec() -> None:
    with pytest.raises(ValueError):
        MLPArchitecture(dims=(1,))
    with pytest.raises(ValueError):
        MLPArchitecture(dims=(1, 1), activation="relu")


def test_flat_params_from_layers_matches_layout() -> None:
    flat = flat_params_from_layers([([[-1.3]], [0.25]), ([[1.2]], [0.1])])
    assert flat == TEACHER


# --------------------------------------------------------------------------- #
# enclosure soundness (dense grid + random samples inside the box)
# --------------------------------------------------------------------------- #
def test_interval_grad_hess_enclose_true_values() -> None:
    problem = ParamSpaceLoss(ARCH, DATA)
    radius = 5e-3
    box = [Interval(t - radius, t + radius) for t in TEACHER]
    grad_box = problem.grad(box)
    hess_box = problem.hessian(box)

    # deterministic grid of the 3^4 corner/centre combinations + uniform random samples
    offsets = [-radius, 0.0, radius]
    grid = [
        [TEACHER[k] + off[k] for k in range(4)]
        for off in _product(offsets, 4)
    ]
    rng = random.Random(1234)
    samples = grid + [
        [rng.uniform(t - radius, t + radius) for t in TEACHER] for _ in range(40)
    ]

    for p in samples:
        g = _fd_grad(p, DATA)
        for i in range(4):
            assert grad_box[i].lo - 1e-6 <= g[i] <= grad_box[i].hi + 1e-6
        hh = _fd_hess(p, DATA)
        for i in range(4):
            for j in range(4):
                assert hess_box[i][j].lo - 1e-3 <= hh[i][j] <= hess_box[i][j].hi + 1e-3


def test_value_encloses_true_loss() -> None:
    problem = ParamSpaceLoss(ARCH, DATA)
    radius = 5e-3
    box = [Interval(t - radius, t + radius) for t in TEACHER]
    enc = problem.value(box)
    rng = random.Random(7)
    for _ in range(50):
        p = [rng.uniform(t - radius, t + radius) for t in TEACHER]
        assert enc.lo - 1e-12 <= _loss_float(p, DATA) <= enc.hi + 1e-12


# --------------------------------------------------------------------------- #
# positive certificate
# --------------------------------------------------------------------------- #
def test_certify_strict_local_min_positive() -> None:
    cert = certify_trained_min(ARCH, DATA, TEACHER, radius=1e-3)
    assert isinstance(cert, TrainingCertificate)
    assert cert.unique_stationary is True
    assert cert.strict_local_min is True
    assert cert.certified is True
    assert cert.verified is True  # sealed digest matches body
    assert cert.flatness.eig_min.lo > 0.0
    assert cert.loss_enclosure.lo <= 0.0 <= cert.loss_enclosure.hi
    assert cert.loss_enclosure.hi < 1e-4  # realizable fit -> ~zero loss over the small box

    meta = cert.certificate["meta"]
    assert meta["kind"] == "strict_local_min"
    assert meta["unique_stationary"] is True
    assert meta["strict_local_min"] is True
    assert cert.certificate["honesty"]["unproven_claim"] is False
    assert cert.certificate["honesty"]["strict_local_min"] is True


def test_certify_provenance_recorded() -> None:
    cert = certify_trained_min(
        ARCH, DATA, TEACHER, radius=1e-3, provenance={"run": "unit-test", "seed": 0}
    )
    assert cert.certificate["meta"]["provenance"] == {"run": "unit-test", "seed": 0}
    assert cert.certificate["meta"]["dims"] == [1, 1, 1]
    assert cert.certificate["meta"]["activation"] == "tanh"


# --------------------------------------------------------------------------- #
# negative controls
# --------------------------------------------------------------------------- #
def test_certify_refuses_non_stationary_point() -> None:
    # shift the output bias away from the fit -> gradient no longer vanishes
    bad = list(TEACHER)
    bad[3] += 0.3
    cert = certify_trained_min(ARCH, DATA, bad, radius=1e-3)
    assert cert.unique_stationary is False
    assert cert.certified is False


def test_certify_refuses_too_wide_box() -> None:
    # the certificate is local: a large ball can no longer certify PD / uniqueness
    cert = certify_trained_min(ARCH, DATA, TEACHER, radius=1.0)
    assert cert.certified is False
    assert cert.flatness.eig_min.lo <= 0.0


def test_certify_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        certify_trained_min(ARCH, DATA, TEACHER[:3], radius=1e-3)
    with pytest.raises(ValueError):
        certify_trained_min(ARCH, DATA, TEACHER, radius=0.0)


# --------------------------------------------------------------------------- #
# tamper evidence + Lean obligation
# --------------------------------------------------------------------------- #
def test_certificate_tamper_is_detected() -> None:
    cert = certify_trained_min(ARCH, DATA, TEACHER, radius=1e-3)
    assert cert.verified
    tampered = copy.deepcopy(cert.certificate)
    tampered["payload"]["interval"]["lo"] = "0.0"  # forge a weaker eig_min lower bound
    assert not verify_certificate_digest(tampered)


def test_lean_obligation_is_well_formed() -> None:
    cert = certify_trained_min(ARCH, DATA, TEACHER, radius=1e-3)
    obligation = generate_obligation(cert.certificate)
    assert obligation is not None
    assert "enclosed_quantity_pos" in obligation  # eig_min.lo > 0 sign obligation


def test_lean_gate_runs_or_degrades_gracefully() -> None:
    cert = certify_trained_min(ARCH, DATA, TEACHER, radius=1e-3, lean=True)
    assert cert.lean is not None
    if lean_check_available():
        assert cert.lean.verified is True
        assert cert.theorem_prover_verified is True
    else:
        assert cert.lean.available is False
        assert cert.theorem_prover_verified is False


# --------------------------------------------------------------------------- #
# kernel-verified matrix positive-definiteness (full LDL^T inertia vector)
# --------------------------------------------------------------------------- #
def test_strict_min_seals_matrix_pd_pivot_vector() -> None:
    cert = certify_trained_min(ARCH, DATA, TEACHER, radius=1e-3)
    assert cert.strict_local_min is True
    assert cert.positive_definite is True
    assert cert.pd_certificate is not None
    assert cert.verified is True  # both the eig_min and the PD pivot certificates seal cleanly
    payload = cert.pd_certificate["payload"]
    assert payload["type"] == "positive_definite"
    assert payload["n"] == ARCH.n_params
    assert len(payload["pivots"]) == ARCH.n_params
    assert cert.certificate["meta"]["positive_definite"] is True


def test_pd_obligation_is_the_full_inertia_vector_not_the_scalar() -> None:
    cert = certify_trained_min(ARCH, DATA, TEACHER, radius=1e-3)
    assert cert.pd_certificate is not None
    obligation = generate_obligation(cert.pd_certificate)
    assert obligation is not None
    # The matrix-PD obligation is the whole pivot vector, not a single enclosed scalar.
    assert "allPivotsPos" in obligation
    assert "import Omnibias.LDLT" in obligation
    assert "enclosed_quantity_pos" not in obligation


def test_non_strict_certificate_has_no_pd_payload() -> None:
    # a too-wide ball no longer certifies PD, so no pivot vector is sealed
    cert = certify_trained_min(ARCH, DATA, TEACHER, radius=1.0)
    assert cert.certified is False
    assert cert.positive_definite is False
    assert cert.pd_certificate is None


def test_pd_certificate_tamper_breaks_verified() -> None:
    cert = certify_trained_min(ARCH, DATA, TEACHER, radius=1e-3)
    assert cert.pd_certificate is not None
    tampered = copy.deepcopy(cert)
    # forge the first pivot's lower endpoint to a stronger (larger) value
    tampered.pd_certificate["payload"]["pivots"][0]["lo"] = (1e9).hex()
    assert not verify_certificate_digest(tampered.pd_certificate)


# --------------------------------------------------------------------------- #
# L2-regularised certification: reach past the single-unit ceiling
# --------------------------------------------------------------------------- #
# A 2-hidden-unit tanh net (P=7) fit to realisable data is over-parametrised: its bare-loss
# minimum has a near-flat direction (the true smallest Hessian eigenvalue is ~1e-7), so it is NOT
# a strict local minimum and cannot honestly be certified. Training with weight decay changes the
# objective to J(theta) = L(theta) + l2*||theta||^2, whose Hessian Hess(L) + 2*l2*I lifts every
# eigenvalue by 2*l2 -- turning it into a genuinely strict, certifiable minimum. THETA_REG2 is the
# regularised minimiser argmin J for LAM2 (Newton to |grad J| ~ 1e-15); as a bit-reproducible fixed
# vector it certifies deterministically, and the test self-checks that it is really stationary.
ARCH2 = MLPArchitecture(dims=(1, 2, 1), activation="tanh")
TEACHER2 = [1.5, -1.1, -0.4, 0.3, 0.8, -0.6, 0.15]  # flat layout [w1, w2, b1, b2, v1, v2, c]
GRID2 = [(-2.0 + 4.0 * i / 13.0) for i in range(14)]


def _net2(theta: list[float], x: float) -> float:
    w1, w2, b1, b2, v1, v2, c = theta
    return v1 * math.tanh(w1 * x + b1) + v2 * math.tanh(w2 * x + b2) + c


DATA2: Data = [((x,), (_net2(TEACHER2, x),)) for x in GRID2]
LAM2 = 0.2
THETA_REG2 = [
    0.6058923674625403,
    -0.6058923674625402,
    -0.06111551848384052,
    0.06111551848384049,
    0.7257819538408141,
    -0.7257819538408139,
    0.029476461542900508,
]


def test_l2_regularizer_folds_in_exactly() -> None:
    # Hess(L + l2||theta||^2) = Hess(L) + 2*l2*I, added analytically -> exact, no extra widening.
    box = [Interval(t - 1e-3, t + 1e-3) for t in THETA_REG2]
    bare = ParamSpaceLoss(ARCH2, DATA2, l2=0.0).hessian(box)
    reg = ParamSpaceLoss(ARCH2, DATA2, l2=LAM2).hessian(box)
    n = ARCH2.n_params
    for i in range(n):
        for j in range(n):
            shift = 2.0 * LAM2 if i == j else 0.0
            assert abs((reg[i][j].lo - bare[i][j].lo) - shift) < 1e-9
            assert abs((reg[i][j].hi - bare[i][j].hi) - shift) < 1e-9


def test_certify_bigger_net_via_l2_regularization() -> None:
    assert ARCH2.n_params == 7  # bigger than the single-unit baseline (P=4)
    # self-check: THETA_REG2 really is the regularised minimiser (grad J ~ 0 at the fixed vector)
    pt = [Interval.point(t) for t in THETA_REG2]
    grad_j = ParamSpaceLoss(ARCH2, DATA2, l2=LAM2).grad(pt)
    assert max(abs(gi.mid) for gi in grad_j) < 1e-6

    cert = certify_trained_min(ARCH2, DATA2, THETA_REG2, radius=1e-3, l2=LAM2)
    assert cert.certified is True
    assert cert.unique_stationary is True
    assert cert.strict_local_min is True
    assert cert.verified is True
    assert cert.flatness.eig_min.lo > 0.0
    assert cert.certificate["meta"]["l2"] == LAM2
    assert cert.certificate["meta"]["n_params"] == 7


def test_bare_loss_min_is_not_certifiable_without_l2() -> None:
    # The same point with l2=0 is not certifiable (it is the regularised minimiser, not a critical
    # point of the bare loss) -> refused. The regulariser is load-bearing.
    cert0 = certify_trained_min(ARCH2, DATA2, THETA_REG2, radius=1e-3, l2=0.0)
    assert cert0.certified is False


def test_certify_rejects_negative_l2() -> None:
    with pytest.raises(ValueError):
        certify_trained_min(ARCH, DATA, TEACHER, radius=1e-3, l2=-0.1)
    with pytest.raises(ValueError):
        ParamSpaceLoss(ARCH, DATA, l2=-1.0)


def test_default_l2_is_zero_in_meta() -> None:
    cert = certify_trained_min(ARCH, DATA, TEACHER, radius=1e-3)
    assert cert.certificate["meta"]["l2"] == 0.0


def _product(values: list[float], repeat: int) -> list[list[float]]:
    out: list[list[float]] = [[]]
    for _ in range(repeat):
        out = [prefix + [v] for prefix in out for v in values]
    return out
