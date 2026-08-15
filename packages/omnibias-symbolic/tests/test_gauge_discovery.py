# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gauge-covariant singlet discoverer: positive/negative controls and API guards."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

geometry = pytest.importorskip("omnibias.geometry")
from omnibias.geometry.gauge._core.covariant_jet import (  # noqa: E402
    SELF_DUAL_ACTION_OVER_TOPOLOGICAL,
    SINGLET_TR_F2,
    SINGLET_TR_F_FTILDE,
)
from omnibias.symbolic.field_discovery import FieldJet  # noqa: E402
from omnibias.symbolic.gauge_discovery import (  # noqa: E402
    _GAUGE_EXTRA_HINT,
    GaugeLawDiscoverer,
    discover_yang_mills_singlet_law,
    make_yang_mills_bpst_split,
    make_yang_mills_polynomial_split,
)

BPST_RMSE_FLOOR = 1e-6
GENERIC_RMSE_FLOOR = 1e-3


def test_gauge_extra_hint_names_optional_extra() -> None:
    assert "omnibias-symbolic[gauge]" in _GAUGE_EXTRA_HINT


def test_bpst_recovers_self_dual_singlet_law() -> None:
    train, val, test, conns, _pts = make_yang_mills_bpst_split(seed=0)
    out = discover_yang_mills_singlet_law(train, val, test, conns, random_state=0)
    assert out["diagnostics"]["gauge_equivariance"]["passed"] is True
    assert float(out["test_rmse"]) < BPST_RMSE_FLOOR
    terms = out["selected_terms"]
    assert len(terms) == 1
    assert terms[0]["name"] == SINGLET_TR_F_FTILDE
    coef = float(terms[0]["coefficient"])
    assert coef == pytest.approx(SELF_DUAL_ACTION_OVER_TOPOLOGICAL, rel=1e-3)


def test_generic_connection_is_negative_control() -> None:
    train, val, test, conns, _pts = make_yang_mills_polynomial_split(seed=2)
    out = discover_yang_mills_singlet_law(train, val, test, conns, random_state=2)
    terms = {row["name"]: float(row["coefficient"]) for row in out["selected_terms"]}
    one_term_self_dual = (
        set(terms) == {SINGLET_TR_F_FTILDE}
        and abs(terms[SINGLET_TR_F_FTILDE] - SELF_DUAL_ACTION_OVER_TOPOLOGICAL)
        / SELF_DUAL_ACTION_OVER_TOPOLOGICAL
        < 1e-2
        and float(out["test_rmse"]) < BPST_RMSE_FLOOR
    )
    assert not one_term_self_dual


def test_discoverer_is_deterministic() -> None:
    train, val, test, conns, _pts = make_yang_mills_bpst_split(seed=3)
    a = discover_yang_mills_singlet_law(train, val, test, conns, random_state=3)
    b = discover_yang_mills_singlet_law(train, val, test, conns, random_state=3)
    assert a["equation"] == b["equation"]
    assert a["test_rmse"] == b["test_rmse"]


def test_split_points_are_disjoint() -> None:
    _tr, _va, _te, _conns, (x_tr, x_va, x_te) = make_yang_mills_bpst_split(seed=4)
    def _rows(x: np.ndarray) -> set[tuple[float, ...]]:
        return {tuple(np.round(row, 12)) for row in x}

    assert _rows(x_tr).isdisjoint(_rows(x_va))
    assert _rows(x_tr).isdisjoint(_rows(x_te))
    assert _rows(x_va).isdisjoint(_rows(x_te))


def test_rejects_field_jet() -> None:
    train, val, test, conns, _pts = make_yang_mills_bpst_split(seed=5, counts=(8, 4, 4))
    dummy = FieldJet(
        X=np.zeros((8, 4)),
        order=1,
        partials={(0, 0, 0, 0): np.zeros(8), (1, 0, 0, 0): np.zeros(8)},
        var_names=("t", "x", "y", "z"),
    )
    disc = GaugeLawDiscoverer()
    with pytest.raises(TypeError, match="GaugeCovariantJet"):
        disc.discover(dummy, val, test, connections=conns)


@pytest.mark.parametrize("illegal", ["|A|^2", "F_01_2", "u_x"])
def test_illegal_extra_raises_before_stlsq(illegal: str) -> None:
    train, val, test, conns, _pts = make_yang_mills_bpst_split(seed=6, counts=(8, 4, 4))

    def extra(jet):
        return {illegal: np.ones(jet.batch)}

    called = {"stlsq": False}

    def _boom(*_args, **_kwargs):
        called["stlsq"] = True
        raise AssertionError("fit_sparse_equation must not run")

    disc = GaugeLawDiscoverer()
    with patch("omnibias.symbolic.gauge_discovery.fit_sparse_equation", _boom):
        with pytest.raises(ValueError, match="allowlisted"):
            disc.discover(
                train, val, test, connections=conns, extra_columns_fn=extra
            )
    assert called["stlsq"] is False


def test_rejects_adjoint_singlet_mix() -> None:
    train, val, test, conns, _pts = make_yang_mills_bpst_split(seed=7, counts=(8, 4, 4))

    def extra(jet):
        adj = jet.adjoint_1forms()
        return {name: np.mean(val, axis=(1, 2)) for name, val in adj.items()}

    disc = GaugeLawDiscoverer()
    with pytest.raises(TypeError, match="adjoint"):
        disc.discover(train, val, test, connections=conns, extra_columns_fn=extra)


def test_missing_connections_does_not_pass_gate() -> None:
    train, val, test, _conns, _pts = make_yang_mills_bpst_split(seed=9, counts=(8, 4, 4))
    out = GaugeLawDiscoverer().discover(train, val, test)
    assert out.diagnostics["gauge_equivariance"]["passed"] is False
    assert out.diagnostics["gauge_equivariance"]["yang_mills_claim"] is False


def test_gate_failure_is_closed() -> None:
    train, val, test, conns, _pts = make_yang_mills_bpst_split(seed=8, counts=(8, 4, 4))
    disc = GaugeLawDiscoverer()
    with patch(
        "omnibias.symbolic.gauge_discovery.evaluate_gauge_law_gate",
        return_value={"passed": False, "residual_defect": 1.0},
    ):
        with pytest.raises(ValueError, match="equivariance"):
            disc.discover(train, val, test, connections=conns)


def test_lazy_export_from_package_root() -> None:
    from omnibias.symbolic import GaugeLawDiscoverer as exported

    assert exported is GaugeLawDiscoverer
