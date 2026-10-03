# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from omnibias.core.proof.discovery import ExactCheck
from omnibias.geometry.algebraic_route2_search import Route2Candidate

from benchmarks.hilbert16_route2_sweep import _run_best_first_tier


class _TinyRoute2Family:
    name = "tiny_route2"
    symmetry = "d4"
    anchor_count = 1

    def __init__(self) -> None:
        self._scores: dict[Route2Candidate, Fraction] = {}

    def origin(self) -> Route2Candidate:
        return Route2Candidate((0,), 0, 0, "d4")

    def neighbors(self, candidate: Route2Candidate) -> tuple[Route2Candidate, ...]:
        return (
            (Route2Candidate((1,), 0, 0, "d4"),)
            if candidate.coordinates == (0,)
            else ()
        )

    def score(self, candidate: Route2Candidate) -> Fraction:
        score = Fraction(candidate.coordinates[0])
        self._scores[candidate] = score
        return score

    def check(self, candidate: Route2Candidate) -> ExactCheck:
        return ExactCheck(
            False,
            {
                "margin_float": float(candidate.coordinates[0]),
                "exact_lift": None,
                "farkas_digest": None,
                "honesty": {"full_hilbert16_solved": False},
            },
        )

    def cardinality(self) -> int:
        return 2

    def load_score_checkpoint(self, rows: list[dict[str, object]]) -> None:
        del rows

    def score_checkpoint(self) -> list[dict[str, object]]:
        return [
            {"candidate": candidate.as_dict(), "score": str(score)}
            for candidate, score in self._scores.items()
        ]

    def top_scores(self) -> list[dict[str, object]]:
        return self.score_checkpoint()

    def lattice_spec(self) -> dict[str, list[str]]:
        return {"x": ["0", "1"]}


def test_best_first_checkpoint_resumes_without_restarting(tmp_path: Path) -> None:
    state: dict[str, object] = {}
    checkpoint = tmp_path / "frontier.json"
    first = _run_best_first_tier(
        _TinyRoute2Family(),  # type: ignore[arg-type]
        budget=1,
        checkpoint=checkpoint,
        state=state,
        seed_all_anchors=False,
        stagnation_window=2,
        stagnation_tolerance=0.0,
    )
    second = _run_best_first_tier(
        _TinyRoute2Family(),  # type: ignore[arg-type]
        budget=2,
        checkpoint=checkpoint,
        state=state,
        seed_all_anchors=False,
        stagnation_window=2,
        stagnation_tolerance=0.0,
    )
    assert first["evaluated"] == 1
    assert second["evaluated"] == 2
    assert second["exhausted"] is True
    assert len(second["history"]) == 2
    assert checkpoint.is_file()
