# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The demand vocabulary: orders, specs, partial requests and engine refusals (backend-free)."""

from __future__ import annotations

import pytest
from omnibias.core.demand import (
    DERIVATIVE_ORDER,
    QUANTITIES,
    Demand,
    Quantities,
    QuantitySpec,
    Residual,
    block_order,
    partial_requests,
)


def test_vocabulary_is_ordered_and_every_quantity_has_an_order() -> None:
    assert set(QUANTITIES) == set(DERIVATIVE_ORDER)
    assert DERIVATIVE_ORDER["hess_form"] == 2 and DERIVATIVE_ORDER["mixed"] == 2
    assert DERIVATIVE_ORDER["third"] == 3 and DERIVATIVE_ORDER["lap_grad"] == 3
    assert DERIVATIVE_ORDER["fourth"] == 4 and DERIVATIVE_ORDER["bilap"] == 4
    # the established names keep their relative order, so stored configs sort as before
    old = ["value", "grad", "grad_sq", "directional", "lap", "bilap"]
    assert [q for q in QUANTITIES if q in old] == old
    assert Demand.of("fourth", "lap", "hess_form").to_config() == [
        "value",
        "lap",
        "hess_form",
        "fourth",
    ]


def test_quantities_tuple_hides_everything_undeclared() -> None:
    names = [n for n in Quantities._fields if n not in ("u", "x")]
    q = Quantities(**{"u": 1, "x": 2, **dict.fromkeys(names, 3)})
    res = Residual(lambda t: t, ["hess_form", "third"])
    masked = res.masked(q)
    assert masked.hess_form == 3 and masked.third == 3
    assert all(getattr(masked, n) is None for n in names if n not in ("hess_form", "third"))


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="total order 2"):
        QuantitySpec(mixed=((3, 0),))
    with pytest.raises(ValueError, match="total order 4"):
        QuantitySpec(fourth=((2, 1),))
    with pytest.raises(ValueError, match="total order 3"):
        QuantitySpec(third=((4, -1),))
    spec = QuantitySpec(third=((3, 0), (1, 2)), hess_form=[[1.0, 0.0], [0.0, -1.0]])
    spec.check(Demand.of("third", "hess_form"), 2)
    with pytest.raises(ValueError, match="must define"):
        QuantitySpec(third=((3, 0),)).check(Demand.of("mixed"))
    with pytest.raises(ValueError, match="does not read"):
        spec.check(Demand.of("third"))
    with pytest.raises(ValueError, match="length 3"):
        spec.check(Demand.of("third", "hess_form"), 3)


def test_partial_requests_share_one_list() -> None:
    spec = QuantitySpec(third=((3, 0), (1, 2)), mixed=((1, 1),))
    alphas, where, weights = partial_requests(
        Demand.of("mixed", "third", "lap_grad", "bilap"), spec, 2
    )
    assert alphas[: where["mixed"][1]] == [(1, 1)]
    assert alphas[slice(*where["third"])] == [(3, 0), (1, 2)]
    # lap_grad: d_j^2 d_i for i, j in 0..D-1, row major in (i, j)
    assert alphas[slice(*where["lap_grad"])] == [(3, 0), (1, 2), (2, 1), (0, 3)]
    # bilap: u_iiii for each coordinate and 2 u_iijj for each pair
    assert alphas[slice(*where["bilap"])] == [(4, 0), (0, 4), (2, 2)]
    assert weights == (1.0, 1.0, 2.0)
    only = partial_requests(Demand.of("bilap"), QuantitySpec(), 3, coordinates=[0, 2])
    assert only[0] == [(4, 0, 0), (0, 0, 4), (2, 0, 2)] and only[2] == (1.0, 1.0, 2.0)
    assert partial_requests(Demand.of("lap"), QuantitySpec(), 2) == ([], {}, ())


def test_each_engine_serves_its_own_names_and_points_to_the_stream_engine() -> None:
    third = Residual(lambda q: q.u, ["third"])
    assert block_order(third, None, "streams") == 3
    with pytest.raises(ValueError, match="stacked stream engine provides third"):
        block_order(third, None, "projected")
    with pytest.raises(ValueError, match="stacked stream engine provides third"):
        block_order(third, None, "full")
    grad = Residual(lambda q: q.u, ["grad"])
    with pytest.raises(ValueError, match="full gradient is not formed"):
        block_order(grad, None, "projected")
    assert block_order(grad, None, "streams") == 1
    bilap = Residual(lambda q: q.u, ["bilap"])
    with pytest.raises(ValueError, match="first-layer projected engine"):
        block_order(bilap, None, "full")
    with pytest.raises(ValueError, match="engine must be"):
        block_order(third, None, "nonsense")
    assert Demand.of("lap", "bilap", "directional").projectable
    assert not Demand.of("grad").projectable and not Demand.of("lap_grad").projectable
