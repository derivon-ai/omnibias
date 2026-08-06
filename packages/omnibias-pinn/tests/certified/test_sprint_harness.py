# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The internal Navier-Stokes sprint harness: it runs, and it stays private.

Two jobs.  The smoke tests check the ledger has the shape it promises and that
every gate stayed blocked.  The guard tests check the harness never leaks into the
public API -- which matters more, because a "sprint" surface in ``__all__`` is the
kind of thing that gets read as a result.
"""

from __future__ import annotations

import json

import pytest
from omnibias.pinn.certified._sprint import (
    SPRINT_PHASES,
    SPRINT_SCHEMA_VERSION,
    SprintConfig,
    run_candidate_sprint,
    sprint_honesty_errors,
    sprint_summary,
)

SPRINT_SYMBOLS = (
    "SPRINT_PHASES",
    "SPRINT_SCHEMA_VERSION",
    "SprintConfig",
    "run_candidate_sprint",
    "sprint_honesty_errors",
    "sprint_summary",
)


@pytest.fixture(scope="module")
def ledger() -> dict[str, object]:
    return run_candidate_sprint(SprintConfig(seed=11, notes="ci smoke"))


# --------------------------------------------------------------------------- #
# The guard: this must never become public API
# --------------------------------------------------------------------------- #
def test_the_sprint_symbols_are_not_exported_from_the_certified_package() -> None:
    """The whole reason the module is underscore-prefixed."""
    import omnibias.pinn.certified as certified

    for name in SPRINT_SYMBOLS:
        assert name not in certified.__all__
        assert not hasattr(certified, name)


def test_the_sprint_module_is_not_reachable_as_a_public_submodule() -> None:
    """Underscore-prefixed, so it is absent from the package's public surface."""
    import omnibias.pinn.certified as certified

    assert "_sprint" not in certified.__all__
    public = [name for name in dir(certified) if not name.startswith("_")]
    assert "sprint" not in public


def test_no_public_omnibias_module_imports_the_sprint_harness() -> None:
    """Importing it from public code would re-export it by the back door."""
    from pathlib import Path

    root = Path(certified_package_root())
    offenders = [
        path.name
        for path in root.rglob("*.py")
        if not path.name.startswith("_") and "_sprint" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def certified_package_root() -> str:
    import omnibias.pinn.certified as certified

    return str(certified.__path__[0])


# --------------------------------------------------------------------------- #
# The smoke test
# --------------------------------------------------------------------------- #
def test_the_ledger_has_every_phase_in_order(ledger: dict[str, object]) -> None:
    phases = ledger["phases"]
    assert isinstance(phases, dict)
    assert set(phases) == set(SPRINT_PHASES)
    assert ledger["schema_version"] == SPRINT_SCHEMA_VERSION


def test_the_ledger_claims_nothing(ledger: dict[str, object]) -> None:
    """Every honesty flag false, and re-derived from the phases rather than trusted."""
    assert sprint_honesty_errors(ledger) == []
    assert ledger["theorem_claimed"] is False

    honesty = ledger["honesty"]
    assert isinstance(honesty, dict)
    assert honesty["global_regularity_claim"] is False
    assert honesty["finite_time_blowup_claim"] is False
    assert honesty["theorem_prover_verified"] is False


def test_every_terminal_gate_is_blocked(ledger: dict[str, object]) -> None:
    """A gate that opened would be a bug in the harness, not a result."""
    blocked = ledger["blocked_gates"]
    assert isinstance(blocked, list)
    for gate in ("theorem_gate", "review_gate"):
        assert gate in blocked

    phases = ledger["phases"]
    assert isinstance(phases, dict)
    assert phases["theorem_gate"]["unproven_claim"] is False
    assert phases["review_gate"]["unproven_claim"] is False


def test_the_ledger_records_the_open_obligations(ledger: dict[str, object]) -> None:
    """The output is a list of what is still missing; an empty one would be suspicious."""
    obligations = ledger["open_obligations"]
    assert isinstance(obligations, list)
    assert len(obligations) > 10
    assert obligations == sorted(set(obligations))
    assert all(isinstance(item, str) and item for item in obligations)


def test_the_ledger_is_json_serialisable(ledger: dict[str, object]) -> None:
    """It is an artifact; an artifact that cannot be written down is not one."""
    text = json.dumps(ledger, default=str)
    assert json.loads(text)["schema_version"] == SPRINT_SCHEMA_VERSION


def test_the_run_is_reproducible_from_its_seed() -> None:
    """Seeded, so a recorded ledger can be re-derived rather than merely believed."""
    config = SprintConfig(seed=5)
    first = run_candidate_sprint(config)
    second = run_candidate_sprint(config)
    assert first["open_obligations"] == second["open_obligations"]
    assert first["blocked_gates"] == second["blocked_gates"]
    assert first["config"] == second["config"]


def test_the_summary_says_plainly_that_nothing_is_claimed(
    ledger: dict[str, object],
) -> None:
    text = sprint_summary(ledger)
    assert "theorem claimed   : False" in text
    assert "open obligations" in text


# --------------------------------------------------------------------------- #
# The self-check has to be able to fail
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda led: led.update(theorem_claimed=True), "theorem_claimed"),
        (
            lambda led: led["honesty"].update(global_regularity_claim=True),
            "global_regularity_claim",
        ),
        (
            lambda led: led["phases"]["theorem_gate"].update(unproven_claim=True),
            "asserts a proved claim",
        ),
        (lambda led: led.update(open_obligations=[]), "no open obligations"),
        (lambda led: led["phases"].pop("review_gate"), "missing phase"),
        (lambda led: led.update(schema_version="forged"), "schema_version"),
    ],
)
def test_the_honesty_check_rejects_a_doctored_ledger(mutate, expected: str) -> None:
    """A self-check that cannot fail is decoration; each of these must trip it."""
    led = json.loads(json.dumps(run_candidate_sprint(SprintConfig(seed=3)), default=str))
    assert sprint_honesty_errors(led) == []

    mutate(led)
    errors = sprint_honesty_errors(led)
    assert any(expected in error for error in errors), errors


def test_an_unavailable_replay_backend_is_recorded_not_hidden() -> None:
    """An un-replayed artifact must not read as a replayed one."""
    import builtins

    real_import = builtins.__import__

    def fail_symbolic(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("omnibias.symbolic"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fail_symbolic
    try:
        led = run_candidate_sprint(SprintConfig(seed=2))
    finally:
        builtins.__import__ = real_import

    replay = led["phases"]["replay"]
    assert replay["available"] is False
    assert replay["replayed"] is False
    assert "independent_replay_backend_unavailable" in led["open_obligations"]
