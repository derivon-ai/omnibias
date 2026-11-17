# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Planner: descriptors, refusals with reasons, ranking, find mode, plan cache, format versioning."""

from __future__ import annotations

import json

import pytest
from omnibias.core.chain import ChainDescriptor
from omnibias.core.demand import Demand
from omnibias.core.planner import (
    AUTO_FIND_MIN_USES,
    AUTO_TRAINING_USES,
    FIND_CUT_FACTOR,
    FIND_TOP_K,
    PLAN_FORMAT,
    CostModel,
    Descriptor,
    FindRefusal,
    OperatorRequest,
    Plan,
    PlanCache,
    Work,
    algorithms_for,
    default_plan_cache_path,
    estimator_probes,
    host_memory_limit,
    resolve_planning,
    select,
)


def chain(D=1000, H=64, depth=3, act="tanh", out=1, dtype="float64"):
    return ChainDescriptor.from_widths(
        [D] + [H] * depth + [out], [act] * depth + [None], dtype=dtype
    )


def lap(power=1, **kw):
    return OperatorRequest("laplacian", power=power, **kw)


# -- requests and descriptors --------------------------------------------------------------------------------


def test_request_aliases_and_validation():
    assert OperatorRequest("bilaplacian") == OperatorRequest("laplacian", power=2)
    assert OperatorRequest("laplacian", power=3).order == 6
    assert OperatorRequest("divergence").order == 1
    assert OperatorRequest.from_config(lap(2, selected=5).to_config()) == lap(2, selected=5)
    for bad in (
        dict(kind="curl"),
        dict(kind="laplacian", power=0),
        dict(kind="laplacian", metric="weird"),
        dict(kind="laplacian", selected=0),
        dict(kind="laplacian", directions=-1),
        dict(kind="laplacian", differentiate="everything"),
        dict(
            kind="laplacian", accuracy=0.1
        ),  # an accuracy sizes an estimator: it needs exact=False
        dict(kind="laplacian", exact=False, accuracy=0.0),
    ):
        with pytest.raises(ValueError):
            OperatorRequest(**bad)


def test_descriptor_fingerprint_is_stable_and_discriminating():
    a = Descriptor(lap(), chain(), 128)
    assert a.fingerprint() == Descriptor(lap(), chain(), 128).fingerprint()
    assert a.fingerprint() != Descriptor(lap(), chain(), 129).fingerprint()
    assert a.fingerprint() != Descriptor(lap(2), chain(), 128).fingerprint()
    assert a.fingerprint() != Descriptor(lap(), chain(D=999), 128).fingerprint()
    assert a.fingerprint() != Descriptor(lap(), chain(act="sin"), 128).fingerprint()
    with pytest.raises(ValueError):
        Descriptor(lap(), chain(), 0)


# -- feasibility and ranking ---------------------------------------------------------------------------------


def test_projected_wins_for_wide_inputs_and_ties_with_forward_below_width():
    wide = select(Descriptor(lap(), chain(D=1000), 128), backend="jax")
    assert wide.algorithm == "projected"
    narrow = select(Descriptor(lap(), chain(D=16), 128), backend="jax")
    assert narrow.algorithm in ("forward_laplacian", "projected")
    times = {c["name"]: c["seconds"] for c in narrow.candidates}
    assert times["projected"] == pytest.approx(times["forward_laplacian"], rel=0.05)


def test_exact_request_excludes_stochastic_and_records_why():
    plan = select(Descriptor(lap(), chain(), 64), backend="jax")
    assert "hutchinson" not in {c["name"] for c in plan.candidates}
    assert any(r.algorithm == "hutchinson" and "stochastic" in r.reason for r in plan.refusals)
    approx = select(Descriptor(lap(exact=False), chain(), 64), backend="jax")
    assert "hutchinson" in {c["name"] for c in approx.candidates}


def test_refusal_reasons_name_the_cause():
    relu = select(Descriptor(lap(), chain(act="relu"), 32), backend="jax")
    reasons = {r.algorithm: r.reason for r in relu.refusals}
    assert "a.e.-zero second derivative" in reasons["projected"]
    assert relu.algorithm in ("taylor_jet", "nested_autodiff")  # generic fallbacks remain
    opted = select(
        Descriptor(lap(allow_almost_everywhere=True), chain(act="relu"), 32), backend="jax"
    )
    assert opted.algorithm == "projected"
    unknown = select(Descriptor(lap(), chain(act="my_custom"), 32), backend="jax")
    assert "no closed-form" in next(
        r.reason for r in unknown.refusals if r.algorithm == "projected"
    )
    bi = select(Descriptor(lap(2), chain(act="elu"), 32), backend="jax")
    assert "jump below order four" in next(
        r.reason for r in bi.refusals if r.algorithm == "projected"
    )
    assert "affine" in next(
        r.reason
        for r in select(
            Descriptor(lap(), ChainDescriptor.from_widths([8, 8, 1], ["tanh", "tanh"]), 8),
            backend="jax",
            candidates=None,
        ).refusals
        if r.algorithm == "projected"
    )


def test_memory_budget_refuses_and_reports_reason_and_nothing_feasible_raises():
    d = Descriptor(lap(2), chain(D=1000), 128)
    plan = select(d, backend="jax", memory_budget=2**30)
    assert plan.algorithm == "projected"
    assert any(
        r.algorithm == "nested_autodiff" and r.reason.startswith("memory_budget")
        for r in plan.refusals
    )
    with pytest.raises(ValueError, match="no algorithm is feasible"):
        select(d, backend="jax", memory_budget=1)


def test_force_requires_feasibility_and_overrides_rank():
    d = Descriptor(lap(), chain(D=1000), 64)
    assert select(d, backend="jax", force="taylor_jet").algorithm == "taylor_jet"
    assert select(d, backend="jax", force="taylor_jet").mode == "forced"
    with pytest.raises(ValueError, match="cannot force"):
        select(d, backend="jax", force="hutchinson")
    with pytest.raises(ValueError, match="cannot force"):
        select(d, backend="jax", force="nonsense")


def test_find_mode_times_leading_candidates_and_keeps_predictions():
    d = Descriptor(lap(), chain(D=1000), 64)
    seen = []

    lead = [c["name"] for c in select(d, backend="jax").candidates][:2]

    def timer(name):
        seen.append(name)
        return {lead[0]: 3.0, lead[1]: 1.0}.get(name, 9.0)

    plan = select(d, backend="jax", mode="find", timer=timer, top_k=2)
    assert plan.mode == "find" and plan.algorithm == lead[1]
    assert seen == lead
    assert set(plan.measured) == set(lead)
    assert plan.candidates[0]["name"] == lead[0]  # prediction order is kept for comparison
    with pytest.raises(ValueError, match="needs a timer"):
        select(d, backend="jax", mode="find")


def test_find_bounds_the_candidates_after_the_leader_by_what_the_leader_measured():
    d = Descriptor(lap(), chain(D=1000), 64)
    lead = [c["name"] for c in select(d, backend="jax").candidates][:2]
    seen = []

    def timer(name, deadline=None):
        seen.append((name, deadline))
        return 2.0 if name == lead[0] else 50.0  # the second candidate is far slower than the leader

    plan = select(d, backend="jax", mode="find", timer=timer, top_k=2)
    assert seen == [(lead[0], None), (lead[1], FIND_CUT_FACTOR * 2.0)]
    assert plan.algorithm == lead[0]
    row = next(c for c in plan.candidates if c["name"] == lead[1])
    assert "cannot win" in row["find"]
    # a candidate inside the bound is measured in full and can win: the minimum is unchanged
    plan = select(d, backend="jax", mode="find", timer=lambda n, deadline=None: 3.0 if n == lead[0] else 1.0, top_k=2)
    assert plan.algorithm == lead[1]
    assert all("find" not in c for c in plan.candidates)


def test_find_keeps_calling_a_one_argument_timer_as_before():
    d = Descriptor(lap(), chain(D=1000), 64)
    lead = [c["name"] for c in select(d, backend="jax").candidates][:2]
    calls = []

    def timer(name):
        calls.append(name)
        return 1.0 if name == lead[1] else 2.0

    plan = select(d, backend="jax", mode="find", timer=timer, top_k=2)
    assert calls == lead and plan.algorithm == lead[1]


def test_find_does_not_run_a_follower_predicted_to_exhaust_memory_but_always_runs_the_leader():
    d = Descriptor(lap(), chain(D=1000), 64)
    preds = select(d, backend="jax").candidates
    lead = [c["name"] for c in preds][:2]
    limit = float(preds[1]["peak_bytes"]) / 2.0
    ran = []

    def timer(name):
        ran.append(name)
        return 1.0

    plan = select(d, backend="jax", mode="find", timer=timer, top_k=2, find_memory_limit=limit)
    assert ran == lead[:1] and plan.algorithm == lead[0]
    assert "not timed" in next(c for c in plan.candidates if c["name"] == lead[1])["find"]
    ran.clear()  # a leader above the limit is still the user's request: it is timed
    plan = select(d, backend="jax", mode="find", timer=timer, top_k=2, find_memory_limit=1.0)
    assert ran == lead[:1]
    # when the leader fails, the follower is timed whatever its size: something must be measured
    ran.clear()
    plan = select(d, backend="jax", mode="find", timer=lambda n: ran.append(n) or (float("inf") if n == lead[0] else 1.0),
                  top_k=2, find_memory_limit=1.0)
    assert ran == lead and plan.algorithm == lead[1]


def test_find_journal_resumes_an_interrupted_find_and_a_finished_find_starts_fresh(tmp_path):
    d = Descriptor(lap(), chain(D=1000), 64)
    lead = [c["name"] for c in select(d, backend="jax").candidates][:2]
    cache = PlanCache(tmp_path / "plans.json")
    calls = []

    def dies(name):
        calls.append(name)
        if name == lead[1]:
            raise KeyboardInterrupt  # the job is killed while the second candidate runs
        return 2.0

    with pytest.raises(KeyboardInterrupt):
        select(d, backend="jax", mode="find", timer=dies, top_k=2, cache=cache, versions={"v": "1"})
    assert calls == lead and len(cache) == 0
    calls.clear()

    def timer(name):
        calls.append(name)
        return 1.0

    plan = select(d, backend="jax", mode="find", timer=timer, top_k=2, cache=cache, versions={"v": "1"})
    assert calls == lead[1:]  # the leader's measurement is read back, not repeated
    assert plan.measured[lead[0]] == 2.0 and plan.algorithm == lead[1]
    assert "journal" in next(c for c in plan.candidates if c["name"] == lead[0])["find"]
    assert len(cache) == 1
    # the plan is kept and the journal closed: a fresh cache file measures everything again
    calls.clear()
    fresh = PlanCache(tmp_path / "plans.json")
    fresh._entries.clear()
    select(d, backend="jax", mode="find", timer=timer, top_k=2, cache=fresh, versions={"v": "1"})
    assert calls == lead


def test_find_journal_ignores_other_versions_failures_and_torn_lines(tmp_path):
    d = Descriptor(lap(), chain(D=1000), 64)
    lead = [c["name"] for c in select(d, backend="jax").candidates][:2]
    cache = PlanCache(tmp_path / "plans.json")
    calls = []

    def fails_second(name):
        calls.append(name)
        return 2.0 if name == lead[0] else float("inf")

    select(d, backend="jax", mode="find", timer=fails_second, top_k=2, cache=cache, versions={"v": "1"})
    assert len(cache) == 0  # a plan with a failed candidate is not kept
    journal = cache.journal_path
    assert journal is not None
    with journal.open("a") as handle:
        handle.write("{torn line\n")
    assert list(cache.journal_get("any", "any")) == []
    calls.clear()
    select(d, backend="jax", mode="find", timer=fails_second, top_k=2, cache=cache, versions={"v": "2"})
    assert calls == lead  # another version: nothing is reused
    calls.clear()
    select(d, backend="jax", mode="find", timer=fails_second, top_k=2, cache=cache, versions={"v": "2"})
    assert calls == lead[1:]  # the success is reused, the failure is retried
    assert "inf" not in journal.read_text().lower()


def test_find_with_top_k_none_times_every_feasible_candidate_and_keeps_the_measured_best():
    d = Descriptor(lap(), chain(D=1000), 64)
    ranked = [c["name"] for c in select(d, backend="jax").candidates]
    assert len(ranked) > FIND_TOP_K  # the property below is vacuous on a pool of two
    seen = []

    def timer(name):
        seen.append(name)
        return 1.0 if name == ranked[-1] else 5.0  # the cost model ranks the fastest program last

    default = select(d, backend="jax", mode="find", timer=timer)
    assert seen == ranked[:FIND_TOP_K] and default.algorithm != ranked[-1]
    seen.clear()
    exhaustive = select(d, backend="jax", mode="find", timer=timer, top_k=None)
    assert seen == ranked and exhaustive.algorithm == ranked[-1]
    assert set(exhaustive.measured) == set(ranked)
    assert [c["name"] for c in exhaustive.candidates] == ranked  # the prediction order is kept for comparison


def test_exhaustive_find_still_cuts_slow_followers_and_skips_what_exceeds_host_memory():
    d = Descriptor(lap(), chain(D=1000), 64)
    preds = select(d, backend="jax").candidates
    ranked = [c["name"] for c in preds]
    deadlines = {}

    def timer(name, deadline=None):
        deadlines[name] = deadline
        return 2.0 if name == ranked[0] else 80.0

    plan = select(d, backend="jax", mode="find", timer=timer, top_k=None)
    assert deadlines[ranked[0]] is None
    assert all(deadlines[n] == FIND_CUT_FACTOR * 2.0 for n in ranked[1:])
    assert plan.algorithm == ranked[0]
    assert all("cannot win" in c["find"] for c in plan.candidates[1:])
    deadlines.clear()
    limit = min(float(c["peak_bytes"]) for c in preds[1:]) / 2.0  # every follower is predicted above the limit
    plan = select(d, backend="jax", mode="find", timer=timer, top_k=None, find_memory_limit=limit)
    assert list(deadlines) == ranked[:1]
    assert all("not timed" in c["find"] for c in plan.candidates[1:])


def test_top_k_and_library_versions_are_part_of_the_find_key_and_the_defaults_are_not(tmp_path):
    d = Descriptor(lap(), chain(D=1000), 64)
    calls = []

    def timer(name):
        calls.append(name)
        return 1.0

    cache = PlanCache()
    select(d, backend="jax", mode="find", timer=timer, cache=cache)
    n = len(calls)
    select(d, backend="jax", mode="find", timer=timer, cache=cache, top_k=FIND_TOP_K)
    assert cache.hits == 1 and len(calls) == n and len(cache) == 1  # the default top_k is the key it always had
    select(d, backend="jax", mode="find", timer=timer, cache=cache, top_k=None)
    assert len(cache) == 2 and len(calls) > n  # an exhaustive find is another measurement
    after_all = len(calls)
    select(d, backend="jax", mode="find", timer=timer, cache=cache, top_k=None)
    assert cache.hits == 2 and len(calls) == after_all
    select(d, backend="jax", mode="find", timer=timer, cache=cache, top_k=1)
    assert len(cache) == 3  # so is a different count
    # a heuristic plan does not time anything: top_k does not change its key
    select(d, backend="jax", cache=cache)
    select(d, backend="jax", cache=cache, top_k=None)
    assert len(cache) == 4

    # library versions: a plan measured under other versions is measured again, under the same ones it is reused
    versioned = PlanCache(tmp_path / "plans.json")
    select(d, backend="jax", mode="find", timer=timer, cache=versioned, versions={"jax": "0.10.2", "omnibias-jax": "1"})
    before = len(calls)
    select(d, backend="jax", mode="find", timer=timer, cache=versioned, versions={"omnibias-jax": "1", "jax": "0.10.2"})
    assert len(calls) == before and versioned.hits == 1  # the order of the mapping is not part of the key
    select(d, backend="jax", mode="find", timer=timer, cache=versioned, versions={"jax": "0.11.0", "omnibias-jax": "1"})
    assert len(calls) > before and len(versioned) == 2
    reloaded = PlanCache(tmp_path / "plans.json")
    select(d, backend="jax", mode="find", timer=timer, cache=reloaded, versions={"jax": "0.11.0", "omnibias-jax": "1"})
    assert reloaded.hits == 1  # persisted
    # a heuristic key carries no versions: the cost model is the whole input
    select(d, backend="jax", cache=versioned, versions={"jax": "0.10.2"})
    select(d, backend="jax", cache=versioned, versions={"jax": "0.11.0"})
    assert len(versioned) == 3


def test_a_failure_that_repeats_is_a_refusal_of_the_plan_and_the_plan_is_cached(tmp_path):
    d = Descriptor(lap(), chain(D=1000), 64)
    ranked = [c["name"] for c in select(d, backend="jax").candidates]
    cache = PlanCache(tmp_path / "plans.json")
    calls = []

    def timer(name):
        calls.append(name)
        if name == ranked[0]:
            raise FindRefusal("MemoryBudgetExceeded: compiled size 3.0 GiB exceeds 1.0 GiB")
        return 2.0 if name == ranked[1] else 4.0

    plan = select(d, backend="jax", mode="find", timer=timer, top_k=None, cache=cache, versions={"v": "1"})
    assert calls == ranked and plan.algorithm == ranked[1]
    assert ranked[0] not in plan.measured
    refusal = next(r for r in plan.refusals if r.algorithm == ranked[0])
    assert refusal.reason.startswith("find: MemoryBudgetExceeded")
    row = next(c for c in plan.candidates if c["name"] == ranked[0])
    assert "refused when timed" in row["find"]
    assert plan.explain()["refused"][-1]["algorithm"] == ranked[0]
    assert len(cache) == 1  # kept: the failure repeats, so there is nothing to retry
    # the next call, in another process, reads the plan and times nothing
    calls.clear()
    again = select(d, backend="jax", mode="find", timer=timer, top_k=None, cache=PlanCache(tmp_path / "plans.json"), versions={"v": "1"})
    assert calls == [] and again.algorithm == plan.algorithm
    assert any(r.algorithm == ranked[0] for r in again.refusals)
    assert Plan.from_config(plan.to_config()).refusals == plan.refusals  # no format change: refusals were always stored

    # an infinite time is a failure that may not repeat: the plan is still not kept
    transient = PlanCache()
    select(d, backend="jax", mode="find", timer=lambda n: float("inf") if n == ranked[0] else 1.0, top_k=2, cache=transient)
    assert len(transient) == 0
    # a refusal and a transient failure together: not kept either
    mixed = PlanCache()

    def both(name):
        if name == ranked[0]:
            raise FindRefusal("crash")
        return float("inf") if name == ranked[1] else 1.0

    select(d, backend="jax", mode="find", timer=both, top_k=None, cache=mixed)
    assert len(mixed) == 0


def test_find_with_every_candidate_refused_names_each_reason():
    d = Descriptor(lap(), chain(D=1000), 64)

    def timer(name):
        raise FindRefusal(f"{name} crashed")

    with pytest.raises(ValueError, match="every timed candidate failed") as info:
        select(d, backend="jax", mode="find", timer=timer, top_k=2)
    leader = select(d, backend="jax").algorithm
    assert f"{leader}: {leader} crashed" in str(info.value)


def test_cache_key_extra_and_find_use_bucket_leave_default_keys_unchanged():
    d = Descriptor(lap(), chain(D=64), 8)
    model = CostModel()
    base = PlanCache.key(d, "cpu", model, "find", None, 1000, "jax")
    assert PlanCache.key(d, "cpu", model, "find", None, 1000, "jax", "") == base
    assert PlanCache.key(d, "cpu", model, "find", None, 1000, "jax", "precision=tf32") != base
    cache = PlanCache()
    t = lambda name: 1.0  # noqa: E731
    first = select(d, backend="jax", mode="find", timer=t, expected_uses=1234, cache=cache)
    again = select(d, backend="jax", mode="find", timer=t, expected_uses=1000, cache=cache)
    assert again is not None and cache.hits == 1 and again.expected_uses == first.expected_uses
    other = select(d, backend="jax", mode="find", timer=t, expected_uses=1000, cache=cache, key_extra="precision=tf32")
    assert cache.hits == 1 and other is not None and len(cache) == 2
    heuristic = lambda uses: select(d, backend="jax", expected_uses=uses, cache=cache)  # noqa: E731
    heuristic(1234), heuristic(1000)
    assert len(cache) == 4  # a heuristic key keeps the exact use count


def test_host_memory_limit_is_a_fraction_of_a_positive_size():
    limit = host_memory_limit(0.5)
    assert limit is None or limit > 0
    full = host_memory_limit(1.0)
    assert limit is None or abs(full - 2 * limit) < 1.0


def test_expected_reuse_trades_compile_cost_against_steady_state():
    d = Descriptor(lap(), chain(D=64), 8)
    cheap_compile = CostModel(compile_s_per_unit=0.0)
    assert (
        select(d, backend="jax", cost_model=cheap_compile).candidates[0]["compile_seconds"] == 0.0
    )
    heavy = CostModel(compile_s_per_unit=1e3)
    once = select(d, backend="jax", cost_model=heavy, expected_uses=1)
    many = select(d, backend="jax", cost_model=heavy, expected_uses=10**9)
    assert once.algorithm in {c["name"] for c in once.candidates}
    assert many.candidates[0]["seconds"] <= once.candidates[0]["seconds"] * 1.0001


def test_residual_demand_decides_which_engine_applies():
    req = OperatorRequest("gram", directions=1)
    projected_only = Descriptor(req, chain(D=512, H=64), 256, Demand.of("lap", "directional"))
    plan = select(projected_only, backend="jax")
    assert plan.algorithm == "projected_streams"
    assert any(r.algorithm == "full_streams" for r in plan.refusals)
    full_only = Descriptor(
        OperatorRequest("gauss_newton"), chain(D=512, H=64), 256, Demand.of("grad", "lap")
    )
    plan = select(full_only, backend="jax")
    assert plan.algorithm == "full_streams"
    assert any(
        r.algorithm == "projected_streams" and "full gradient" in r.reason for r in plan.refusals
    )
    fourth = Descriptor(req, chain(D=512, H=64), 256, Demand.of("bilap", "lap"))
    assert select(fourth, backend="jax").algorithm == "projected_streams"


def test_projected_residual_gram_pays_the_qr_of_its_first_layer_only_when_it_reduces_columns():
    # ``_First`` of the projected engine runs a QR (one geqrf and one orgqr library call, counted in the compiled program)
    # of the first-layer metric matrix when more than H_1 inputs are selected, and none otherwise; full_streams has none
    req, demand = OperatorRequest("gram"), Demand.of("value", "lap")
    programs = {a.name: a for a in algorithms_for("gram", "jax")}
    reduces = Descriptor(req, chain(D=128, H=16, depth=2), 200, demand)
    keeps = Descriptor(req, chain(D=16, H=16, depth=2), 200, demand)
    assert programs["projected_streams"].work(reduces).solver_calls == 2
    assert programs["projected_streams"].work(keeps).solver_calls == 0
    assert programs["full_streams"].work(reduces).solver_calls == 0
    # a batch too small to amortize the two library calls goes to the full-gradient program; a large one does not
    cost = CostModel.for_device("a100", "jax")
    small = Descriptor(req, chain(D=78, H=16, depth=2), 54, demand)
    large = Descriptor(req, chain(D=512, H=64, depth=2), 256, demand)
    assert select(small, backend="jax", device="a100", cost_model=cost).algorithm == "full_streams"
    assert select(large, backend="jax", device="a100", cost_model=cost).algorithm == "projected_streams"


def test_vector_field_quantities_have_projected_and_generic_algorithms():
    sq = ChainDescriptor.from_widths([256, 32, 32, 256], ["tanh", "tanh", None])
    for kind in ("divergence", "logdet", "solve"):
        names = {a.name for a in algorithms_for(kind, "jax")}
        assert len(names) == 2
        plan = select(Descriptor(OperatorRequest(kind), sq, 64), backend="jax")
        assert plan.algorithm in names
    rect = ChainDescriptor.from_widths([256, 32, 8], ["tanh", None])
    plan = select(Descriptor(OperatorRequest("divergence"), rect, 64), backend="jax")
    assert any("vector field" in r.reason for r in plan.refusals)


# -- plan, cache, versions -----------------------------------------------------------------------------------


def test_plan_roundtrip_and_explain_keys():
    plan = select(
        Descriptor(lap(), chain(), 64), backend="jax", device="a100", versions={"omnibias": "1"}
    )
    again = Plan.from_config(json.loads(json.dumps(plan.to_config())))
    assert again == plan
    info = plan.explain()
    for key in (
        "selected_strategy",
        "candidates",
        "refused",
        "cold_cost_seconds",
        "predicted_seconds",
        "mode",
    ):
        assert key in info
    assert info["selected_strategy"] == plan.algorithm
    with pytest.raises(ValueError, match="plan format"):
        Plan.from_config({**plan.to_config(), "format": PLAN_FORMAT + 1})


def test_plan_cache_hits_persists_and_invalidates(tmp_path):
    path = tmp_path / "plans.json"
    cache = PlanCache(path)
    d = Descriptor(lap(), chain(), 64)
    first = select(d, backend="jax", device="a100", cache=cache)
    assert (cache.hits, cache.misses, len(cache)) == (0, 1, 1)
    assert select(d, backend="jax", device="a100", cache=cache) == first
    assert cache.hits == 1
    reloaded = PlanCache(path)
    assert select(d, backend="jax", device="a100", cache=reloaded) == first and reloaded.hits == 1
    # another device or another calibration is another key
    select(d, backend="jax", device="cpu", cache=cache)
    assert len(cache) == 2
    select(
        d,
        backend="jax",
        device="a100",
        cache=cache,
        cost_model=CostModel(efficiency={"projected": 1e3}),
    )
    assert len(cache) == 3
    # a file of another format is ignored, not misread
    path.write_text(json.dumps({"format": PLAN_FORMAT + 7, "plans": {"x": {}}}))
    assert len(PlanCache(path)) == 0
    path.write_text("not json")
    assert len(PlanCache(path)) == 0


def test_cost_model_digest_tracks_calibration_and_unknown_device_defaults():
    base = CostModel.for_device("some-new-gpu")
    assert base.source == "defaults"
    assert base.digest() != CostModel(efficiency={"projected": 2.0}).digest()
    assert CostModel.from_config(base.to_config()) == base


def test_explain_names_the_cost_model_that_ranked_the_plan():
    d = Descriptor(lap(), chain(D=200, H=32), 64)
    generic = select(d, device="generic", cost_model=CostModel(device="generic"))
    assert generic.explain()["cost_model"] == "defaults"
    fitted = CostModel(device="x", source="fitted on 3 measurements")
    plan = select(d, device="x", cost_model=fitted, versions={"jax": "0"})
    assert plan.explain()["cost_model"] == "fitted on 3 measurements"
    assert plan.to_config()["versions"]["jax"] == "0"
    assert Plan.from_config(plan.to_config()).explain()["cost_model"] == "fitted on 3 measurements"


# -- planner v2: objective, latency terms, auto modes, pool --------------------------------------------------


def _names(plan):
    return [c["name"] for c in plan.candidates]


def test_training_objective_costs_more_than_forward_and_is_recorded():
    fwd = select(Descriptor(lap(), chain(D=8), 256), backend="jax")
    trn = select(Descriptor(lap(differentiate="params"), chain(D=8), 256), backend="jax")
    assert trn.objective == "params" and fwd.objective == "forward"
    f = {c["name"]: c["seconds"] for c in fwd.candidates}
    t = {c["name"]: c["seconds"] for c in trn.candidates}
    assert all(t[n] > f[n] for n in f)
    assert trn.explain()["objective"] == "params"
    assert (
        Descriptor(lap(differentiate="params"), chain(), 8).fingerprint()
        != Descriptor(lap(), chain(), 8).fingerprint()
    )


def test_second_derivative_into_parameters_refuses_the_fused_kernels():
    d = Descriptor(lap(differentiate="params2"), chain(D=4, H=32), 256)
    plan = select(d, backend="jax", device="a100")
    assert "fused_streams" not in _names(plan)
    assert "second derivative" in next(
        r.reason for r in plan.refusals if r.algorithm == "fused_streams"
    )
    first = select(
        Descriptor(lap(differentiate="params"), chain(D=4, H=32, dtype="float32"), 256),
        backend="torch",
        device="a100",
    )
    assert "fused_streams" in _names(first)


def test_fused_streams_is_an_accelerator_program_with_named_limits():
    d = lambda **kw: Descriptor(lap(), chain(**{"D": 4, "H": 32, "dtype": "float32", **kw}), 128)  # noqa: E731
    cpu = select(d(), backend="torch", device="cpu")
    assert "accelerator" in next(r.reason for r in cpu.refusals if r.algorithm == "fused_streams")
    assert "fused_streams" in _names(select(d(), backend="torch", device="a100"))

    def why(descriptor):
        plan = select(descriptor, backend="torch", device="a100")
        return next(r.reason for r in plan.refusals if r.algorithm == "fused_streams")

    assert "exceeds the fused limit" in why(d(H=256))
    assert "stream channels" in why(d(D=64))
    assert "no fused kernel tower" in why(d(act="gelu"))
    assert "Laplacian" in why(Descriptor(lap(2), chain(D=4, H=32), 128))
    assert "drift" in why(Descriptor(lap(directions=1), chain(D=4, H=32, dtype="float32"), 128))


def test_fused_streams_runs_only_when_forced_where_it_measured_slower():
    f32 = Descriptor(lap(), chain(D=4, H=32, dtype="float32"), 128)
    f64 = Descriptor(lap(), chain(D=4, H=32, dtype="float64"), 128)

    def reason(desc, backend, device="a100", **kw):
        plan = select(desc, backend=backend, device=device, **kw)
        return plan, next((r.reason for r in plan.refusals if r.algorithm == "fused_streams"), None)

    plan, why = reason(f64, "torch")
    assert "float64" in why and "force" in why and plan.algorithm != "fused_streams"
    plan, why = reason(f32, "jax")
    assert "opt-in on JAX" in why and plan.algorithm != "fused_streams"
    # forcing overrides the measured preference, and the interpreter is exempt (tests, debugging)
    assert reason(f64, "torch", force="fused_streams")[0].algorithm == "fused_streams"
    assert reason(f32, "jax", force="fused_streams")[0].algorithm == "fused_streams"
    assert "fused_streams" in _names(select(f64, backend="jax", device="cpu-interpret"))


def test_fused_shared_memory_model_matches_the_a100_compile_table():
    """The static bound predicts every recorded Pallas compile outcome at the smallest tile and the default grid."""
    from pathlib import Path

    from omnibias.core.plan_algorithms import FUSED_SHARED_MEMORY_LIMIT, fused_shared_memory_bytes

    path = Path(__file__).resolve().parents[3] / "docs" / "benchmarks" / "fused_shared_memory.json"
    if not path.exists():
        pytest.skip("calibration table not shipped in this checkout")
    rows = json.loads(path.read_text())["rows"]
    assert len(rows) == 36
    # the one recorded cell where a larger tile needs more than the model says (float64, P=32, D=12, BB=32): the planner
    # and the kernels evaluate the smallest tile, and the kernels shrink an unnamed block
    known_under_prediction = {("float64", 12, 32, 32, "bwd")}
    for row in rows:
        for phase, backward in (("fwd", False), ("bwd", True)):
            need = fused_shared_memory_bytes(
                dtype=row["dtype"],
                width=row["H"],
                channels=row["S"],
                block=row["BB"],
                backward=backward,
            )
            predicted_fail = need > FUSED_SHARED_MEMORY_LIMIT
            key = (row["dtype"], row["D"], row["H"], row["BB"], phase)
            if key in known_under_prediction:
                assert row[phase]["ok"] is False and not predicted_fail
                continue
            assert predicted_fail == (not row[phase]["ok"]), key
            if not row[phase]["ok"] and row[phase].get("requested"):
                assert need >= 0.9 * row[phase]["requested"], key  # a bound, not a coin flip
    assert (
        fused_shared_memory_bytes(dtype="float64", width=128, channels=14) == 0
    )  # float64 forward: no dot tile


def test_fused_streams_refuses_shapes_that_exceed_shared_memory_even_when_forced():
    big = Descriptor(
        lap(differentiate="params"), chain(D=12, H=128, depth=2, dtype="float32", out=1), 2048
    )
    fits = Descriptor(
        lap(differentiate="params"), chain(D=2, H=64, depth=2, dtype="float32", out=1), 2048
    )
    for backend in ("jax", "torch"):
        plan = select(big, backend=backend, device="a100")
        why = next(r.reason for r in plan.refusals if r.algorithm == "fused_streams")
        assert "shared memory" in why and plan.algorithm != "fused_streams"
        with pytest.raises(ValueError, match="shared memory"):
            select(big, backend=backend, device="a100", force="fused_streams")
        # the interpreter has no shared-memory limit
        assert "fused_streams" in _names(select(big, backend=backend, device="cpu-interpret"))
    # a shape inside the limit is only subject to the measured-slower policy
    assert (
        select(fits, backend="torch", device="a100", force="fused_streams").algorithm
        == "fused_streams"
    )


def test_solver_and_sync_latencies_are_not_scaled_by_efficiency():
    algo = algorithms_for("laplacian", "jax")[0]
    w = Work(gemm_flops=1e9, launches=10, solver_calls=2, syncs=1)
    slow = CostModel(efficiency={algo.name: 10.0})
    base = CostModel()
    a = base.predict_work(algo.name, w).seconds
    b = slow.predict_work(algo.name, w).seconds
    latency = 10 * base.launch_s + 2 * base.solver_s + base.sync_s
    assert b - latency == pytest.approx(10.0 * (a - latency))


def test_cholesky_basis_pays_its_solver_calls_and_wins_only_when_gemm_bound():
    small = select(Descriptor(lap(), chain(D=16, H=32), 64), backend="jax")
    assert _names(small).index("projected") < _names(small).index("projected_cholesky")
    big = select(Descriptor(lap(), chain(D=1000, H=256, depth=3), 512), backend="jax")
    assert _names(big)[0] == "projected_cholesky"
    work = {a.name: a for a in algorithms_for("laplacian", "jax")}
    d = Descriptor(lap(), chain(D=16, H=32), 64)
    assert work["projected_cholesky"].work(d).solver_calls == 3.0
    assert work["projected"].work(d).solver_calls == 0.0


def test_divergence_and_logdet_models_follow_the_cyclic_trace():
    one = ChainDescriptor.from_widths([64, 32, 64], ["tanh", None])
    deep = ChainDescriptor.from_widths([64, 32, 32, 32, 32, 64], ["tanh"] * 4 + [None])
    algos = {a.name: a for a in algorithms_for("divergence", "jax")}
    shallow = algos["projected"].work(Descriptor(OperatorRequest("divergence"), one, 128))
    deeper = algos["projected"].work(Descriptor(OperatorRequest("divergence"), deep, 128))
    dense = algos["jacobian_trace"].work(Descriptor(OperatorRequest("divergence"), deep, 128))
    assert deeper.gemm_flops > shallow.gemm_flops and dense.gemm_flops > deeper.gemm_flops
    ld = {a.name: a for a in algorithms_for("logdet", "jax")}
    d = Descriptor(OperatorRequest("logdet"), deep, 128)
    assert ld["projected"].work(d).solver_calls == 1.0
    assert ld["projected"].work(d).gemm_flops < ld["dense_jacobian"].work(d).gemm_flops


def test_logdet_prices_the_carried_columns_by_their_shapes_and_leaves_the_dense_jacobian_to_small_d():
    """Measured on an A100 (float64, ``D -> H -> H -> D``, ``H = 256``): ``projected`` ran 1.7 and 4.6 times faster than the dense
    Jacobian at ``D = 301`` and ``512`` and was priced 3 and 1.4 times slower, because the model charged half a byte of traffic per
    flop of the carried-column products and counted their multiply-adds as elementwise work."""
    model = CostModel.for_device("a100", "jax")

    def prices(D, H, B):
        d = Descriptor(
            OperatorRequest("logdet"),
            ChainDescriptor.from_widths([D, H, H, D], ["tanh", "tanh", None], dtype="float64"),
            B,
        )
        return {a.name: model.predict(a, d).seconds for a in algorithms_for("logdet", "jax")}

    for D, B in ((301, 1503), (512, 1024)):
        price = prices(D, 256, B)
        assert price["projected"] < price["dense_jacobian"], price
    # the carried traffic is counted from shapes: it doubles with the batch
    projected = {a.name: a for a in algorithms_for("logdet", "jax")}["projected"]
    chain_ = ChainDescriptor.from_widths([128, 64, 64, 128], ["tanh", "tanh", None], dtype="float64")
    t1 = projected.work(Descriptor(OperatorRequest("logdet"), chain_, 128)).traffic_bytes
    t2 = projected.work(Descriptor(OperatorRequest("logdet"), chain_, 256)).traffic_bytes
    assert 1.9 < t2 / t1 < 2.0


def test_estimators_are_sized_by_the_accuracy_target():
    assert estimator_probes("hutchinson", None, 100) == 16
    assert estimator_probes("hutchinson", 0.1, 100) == 200  # ceil(2 / 0.01)
    assert estimator_probes("stde", 0.5, 100) == 8
    assert estimator_probes("sdgd", 0.5, 100) == 80  # ceil(100 / 1.25)
    assert estimator_probes("sdgd", None, 5) == 5
    with pytest.raises(ValueError):
        estimator_probes("nonsense", 0.1, 4)
    loose = select(Descriptor(lap(exact=False, accuracy=0.5), chain(D=500), 64), backend="jax")
    assert {"hutchinson", "stde", "sdgd"} <= set(_names(loose))
    tight = select(Descriptor(lap(exact=False, accuracy=0.01), chain(D=500), 64), backend="jax")
    assert "hutchinson" not in _names(tight)
    assert "4096 probes" in next(r.reason for r in tight.refusals if r.algorithm == "hutchinson")
    every = select(Descriptor(lap(exact=False, accuracy=0.001), chain(D=8), 64), backend="jax")
    assert "exact program is cheaper" in next(
        r.reason for r in every.refusals if r.algorithm == "sdgd"
    )
    # the estimators price their own probe count: a tighter target costs more
    a = {x.name: x for x in algorithms_for("laplacian", "jax")}["stde"]
    cheap = a.work(Descriptor(lap(exact=False, accuracy=0.5), chain(D=500), 64)).gemm_flops
    dear = a.work(Descriptor(lap(exact=False, accuracy=0.1), chain(D=500), 64)).gemm_flops
    assert dear > 10 * cheap


def test_taylor_closed_serves_low_powers_and_names_why_not_the_third():
    two = select(Descriptor(lap(2), chain(D=3, H=32), 64), backend="jax")
    assert "taylor_closed" in _names(two)
    three = select(Descriptor(lap(3), chain(D=3, H=32), 64), backend="jax")
    assert "order six" in next(r.reason for r in three.refusals if r.algorithm == "taylor_closed")
    metric = select(Descriptor(lap(2, metric="diagonal"), chain(D=3, H=32), 64), backend="jax")
    assert "Euclidean" in next(r.reason for r in metric.refusals if r.algorithm == "taylor_closed")


def test_taylor_closed_work_counts_the_channels_the_program_stacks():
    """``Delta^2`` carries ``4 m^2 + 1`` channels (the refusal rule's count): the work model prices ``m^2`` directions."""
    from omnibias.core import plan_algorithms as pa

    for m in (2, 5, 16):
        d = Descriptor(lap(2, selected=m), chain(D=m, H=32, depth=2), 8)
        w = pa._work_taylor_closed(d)
        channels = 4 * m * m + 1
        stacked = pa._jet_work(d, float(m * m), "taylor_closed")
        assert 1 + 2 * 2 * (m * m) == channels
        assert w.gemm_flops == stacked.gemm_flops and w.elem_flops > stacked.elem_flops  # plus the contraction
        assert w.peak_bytes == stacked.peak_bytes
    # the Laplacian streams are unchanged: m directions of order two, no contraction term
    one = Descriptor(lap(1, selected=7), chain(D=7, H=32, depth=2), 8)
    assert pa._work_taylor_closed(one) == pa._jet_work(one, 7, "taylor_closed")


def test_taylor_closed_refuses_a_bilaplacian_stack_beyond_the_measured_size():
    from omnibias.core.plan_algorithms import TAYLOR_CLOSED_MAX_CHANNELS

    assert 4 * 64 * 64 + 1 == TAYLOR_CLOSED_MAX_CHANNELS
    ok = select(Descriptor(lap(2), chain(D=64, H=64), 32), backend="jax")
    assert "taylor_closed" in _names(ok)
    wide = select(Descriptor(lap(2), chain(D=65, H=64), 32), backend="jax")
    reason = next(r.reason for r in wide.refusals if r.algorithm == "taylor_closed")
    assert "channels" in reason and "use `projected`" in reason
    assert "projected" in _names(wide) and "taylor_closed" not in _names(wide)
    # the Laplacian streams grow linearly and are not capped
    assert "taylor_closed" in _names(select(Descriptor(lap(1), chain(D=4096, H=64), 32), backend="jax"))


def test_refusal_reasons_of_shared_algorithms_are_identical_across_backends():
    """A program both backends implement is refused for the same reason on either: the rules live in one place."""
    cases = [
        Descriptor(lap(2), chain(D=65, H=64), 32),
        Descriptor(lap(3), chain(D=3, H=32), 64),
        Descriptor(lap(2, metric="diagonal"), chain(D=3, H=32), 64),
        Descriptor(lap(1), chain(D=8, H=32, act="relu"), 16),
        Descriptor(lap(2), chain(D=8, H=32, act="relu"), 16),
        Descriptor(lap(2), chain(D=8, H=32, act="not_an_activation"), 16),
        Descriptor(OperatorRequest("divergence"), ChainDescriptor.from_widths([8, 16, 8], ["tanh", None]), 16),
        Descriptor(OperatorRequest("divergence"), ChainDescriptor.from_widths([8, 16, 3], ["tanh", None]), 16),
    ]
    checked = 0
    for d in cases:
        reasons = {}
        for backend in ("jax", "torch"):
            try:
                plan = select(d, backend=backend, device="cpu")
                reasons[backend] = {r.algorithm: r.reason for r in plan.refusals}
            except ValueError as err:  # nothing feasible: the message lists the reasons, which must agree too
                reasons[backend] = {"<none feasible>": str(err)}
        shared = reasons["jax"].keys() & reasons["torch"].keys()
        for name in shared:
            assert reasons["jax"][name] == reasons["torch"][name], (name, d.request)
            checked += 1
    assert checked >= 8


def test_auto_resolution_of_uses_and_mode():
    assert resolve_planning("heuristic", 7) == ("heuristic", 7, "explicit")
    assert resolve_planning("auto", 7) == ("heuristic", 7, "explicit")
    assert resolve_planning("auto", AUTO_FIND_MIN_USES) == ("find", AUTO_FIND_MIN_USES, "explicit")
    assert resolve_planning("auto", "auto") == ("heuristic", 1, "auto")
    assert resolve_planning("auto", "auto", differentiate="params") == (
        "find",
        AUTO_TRAINING_USES,
        "auto",
    )
    assert resolve_planning("auto", "auto", differentiate="params", uses_hint=10) == (
        "heuristic",
        10,
        "hint",
    )
    assert (
        resolve_planning("auto", "auto", differentiate="params", can_time=False)[0] == "heuristic"
    )
    for bad in (("auto", "often"), ("auto", 0), ("sometimes", 3)):
        with pytest.raises(ValueError):
            resolve_planning(*bad)


def test_select_resolves_auto_and_records_where_the_uses_came_from():
    d = Descriptor(lap(differentiate="params"), chain(D=8), 128)
    seen = []

    def timer(name):
        seen.append(name)
        return 1.0 if len(seen) == 1 else 2.0

    plan = select(d, backend="jax", mode="auto", expected_uses="auto", timer=timer)
    assert (
        plan.mode == "find"
        and plan.uses_source == "auto"
        and plan.expected_uses == AUTO_TRAINING_USES
    )
    assert len(seen) == 2
    quiet = select(d, backend="jax", mode="auto", expected_uses="auto", uses_hint=3)
    assert quiet.mode == "heuristic" and quiet.uses_source == "hint" and quiet.expected_uses == 3
    forward = select(
        Descriptor(lap(), chain(D=8), 128), backend="jax", mode="auto", expected_uses="auto"
    )
    assert forward.mode == "heuristic" and forward.expected_uses == 1


def test_plan_cache_default_path_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("OMNIBIAS_PLAN_CACHE", raising=False)
    monkeypatch.setenv("OMNIBIAS_CACHE_DIR", str(tmp_path))
    assert default_plan_cache_path() == tmp_path / "plans.json"
    monkeypatch.setenv("OMNIBIAS_PLAN_CACHE", str(tmp_path / "x.json"))
    assert default_plan_cache_path() == tmp_path / "x.json"
    monkeypatch.setenv("OMNIBIAS_PLAN_CACHE", "off")
    assert default_plan_cache_path() is None


def test_plan_round_trips_the_v2_fields_and_old_plans_migrate():
    plan = select(
        Descriptor(lap(differentiate="params"), chain(), 64), backend="jax", expected_uses=5
    )
    cfg = plan.to_config()
    assert cfg["format"] == 2 and cfg["objective"] == "params" and cfg["uses_source"] == "explicit"
    assert Plan.from_config(cfg) == plan
    old = {k: v for k, v in cfg.items() if k not in ("objective", "uses_source")}
    old["format"] = 1
    migrated = Plan.from_config(old)
    assert migrated.objective == "forward" and migrated.uses_source == "explicit"


def test_the_polylaplacian_is_priced_by_its_support_jets_and_refused_when_it_cannot_be_traced():
    """``Delta^p`` by local jets unrolls ``C(D, <= p)`` supports: at small ``D`` it is cheap, then it falls behind quickly.

    A ranking that counted a handful of streams put it ahead of the projected programs at ``D = 12`` and the program then
    took minutes to trace (the coverage matrix of the architecture rows ran out of memory on it).
    """
    from omnibias.core.plan_algorithms import POLYLAPLACIAN_MAX_SUPPORTS

    for power in (2, 3):
        for D in (6, 12, 50):
            plan = select(Descriptor(lap(power), chain(D=D, H=5, depth=2), 4), backend="jax")
            assert plan.algorithm == "projected", (power, D)
    times = {
        D: next(
            c["seconds"] + c["compile_seconds"]
            for c in select(
                Descriptor(lap(2), chain(D=D, H=5, depth=2), 4), backend="jax"
            ).candidates
            if c["name"] == "forward_polylaplacian"
        )
        for D in (3, 6, 12, 24)
    }
    assert times[3] < times[6] < times[12] < times[24]
    assert times[24] > 5 * times[6]
    big = select(Descriptor(lap(2), chain(D=1000), 128), backend="jax")
    reason = next(r.reason for r in big.refusals if r.algorithm == "forward_polylaplacian")
    assert "support jets" in reason and str(POLYLAPLACIAN_MAX_SUPPORTS) in reason


def test_torch_on_an_accelerator_ranks_the_triton_kernel_where_it_was_measured_and_counts_the_kernels_pytorch_launches():
    """Measured on an A100 (float32, ``D -> H x 2 -> 1``): the Triton kernel ran 1.5 to 3 times faster than the stream engine for
    hidden width <= 64 and D <= 16, and 3 to 26 times slower at width 128; it compiles for 4 to 7 s on a cold Triton cache. The
    registry counts XLA-fused kernels, so PyTorch's 4 to 7 times larger kernel counts were absent from the ranking and the kernel
    sat behind two equal candidates."""
    long_run = 10**5  # the compile is paid once: it wins when the program is used many times, not in a handful of calls

    def pick(D, H, dtype="float32", device="a100", backend="torch", diff="none", L=2, uses=long_run):
        d = Descriptor(OperatorRequest("laplacian", differentiate=diff), chain(D=D, H=H, depth=L, dtype=dtype), 256)
        return select(d, backend=backend, device=device, expected_uses=uses)

    for D, H in ((2, 64), (4, 32), (12, 64), (16, 64)):
        assert pick(D, H).algorithm == "fused_streams", (D, H)
    assert pick(2, 64, diff="params").algorithm == "fused_streams"
    for uses in (1, 1000):
        assert pick(2, 64, uses=uses).algorithm != "fused_streams", uses  # a short run does not repay 5 s of compilation
        assert pick(2, 64, diff="params", uses=uses).algorithm != "fused_streams", uses
    top2 = [c["name"] for c in pick(2, 64, diff="params", uses=20000).candidates[:2]]
    assert "fused_streams" in top2  # a find (``mode="auto"``) over tens of thousands of steps times it
    for D, H in ((2, 128), (32, 32), (64, 64)):
        assert pick(D, H).algorithm != "fused_streams", (D, H)
    reason = next(r.reason for r in pick(2, 128).refusals if r.algorithm == "fused_streams")
    assert "outside that region" in reason and "force" in reason
    # a shared-memory-limited backward and the other precisions and executors keep their earlier ranking
    assert pick(16, 64, diff="params").algorithm != "fused_streams"
    assert pick(2, 64, dtype="float64").algorithm != "fused_streams"
    assert pick(2, 64, device="cpu").algorithm != "fused_streams"
    assert pick(2, 64, backend="jax").algorithm != "fused_streams"
    assert CostModel.for_device("cpu", "torch").launch_scale == {}
    assert CostModel.for_device("a100", "jax").launch_scale == {}
    assert CostModel.for_device("a100", "jax").compile_seconds == {}
    assert CostModel.for_device("a100", "torch").launch_scale["laplacian1/projected"] > 4
    forced = select(
        Descriptor(OperatorRequest("laplacian"), chain(D=2, H=128, depth=2, dtype="float32"), 256),
        backend="torch", device="a100", force="fused_streams",
    )
    assert forced.algorithm == "fused_streams" and forced.mode == "forced"


def test_a_cpu_ranks_by_flops_with_a_class_prior_and_the_accelerator_model_is_unchanged():
    cpu = CostModel.for_device("cpu", "jax")
    generic = CostModel()
    assert cpu.source == "defaults+cpu class prior"
    assert cpu.rate_gemm < generic.rate_gemm / 100 and cpu.launch_s < generic.launch_s and cpu.sync_s == 0.0
    assert CostModel.for_device("cpu").digest() == cpu.digest()  # no backend named: the JAX-measured class
    # an accelerator, another backend on a CPU and an unknown device keep the generic constants
    assert CostModel.for_device("a100", "jax").rate_gemm == generic.rate_gemm
    assert CostModel.for_device("cpu", "torch").rate_gemm == generic.rate_gemm
    assert CostModel.for_device("some-device", "jax").digest() == CostModel(device="some-device").digest()

    # D > H on a CPU: the Cholesky basis carries about half the channels and measured 1.1 to 2.6 times faster forward
    # and 1.6 to 3.4 times in value-and-gradient in 20 of 20 such cells; the flop term must decide it there
    d = Descriptor(lap(), chain(D=64, H=32, depth=2), 1024)
    assert select(d, backend="jax", device="cpu").algorithm == "projected_cholesky"
    assert select(d, backend="jax", device="a100").algorithm == "projected"  # launches and solver calls decide on a GPU
