# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Guardrail: honest capability labels must never regress into over-claims.

Several packages make precise, load-bearing claims that a marketing edit could
easily over-state. This backend-free test pins the honest wording so a future
change that re-introduces an over-claim fails CI. It is the repo-level companion
that ``omnibias-struct``'s ``test_concept_terminology`` mirrors for the
founding (``delta -> 0``) vs feasibility (``beta -> inf``) axis.

Claims guarded here:

* ``omnibias-keras`` -- "bit-identical" is scoped to the *closed-form activation
  / derivative math* (shared coefficients), not end-to-end layer numerics.
* ``omnibias.score.flow`` -- the trace-of-Jacobian is exact; the ODE
  time-integration is a numerical solver.
* ``omnibias-control`` -- the safety certificate is *model-relative*, not an
  unconditional "certified safe" guarantee.
* ``omnibias-convex`` -- the LP/QP "active set" is the feasibility sense, not the
  founding bias collapse.
* ``omnibias-symbolic`` -- the derivative columns are exact closed form; the
  STLSQ sparse fit is numerical / non-differentiable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    path = REPO_ROOT / rel
    if not path.is_file():
        pytest.skip(f"{rel} not present (installed as a wheel?); skipping filesystem check")
    return " ".join(path.read_text(encoding="utf-8").split())


def test_keras_bit_identical_is_preconditioned() -> None:
    for rel in (
        "packages/omnibias-keras/src/omnibias/keras/__init__.py",
        "packages/omnibias-keras/README.md",
    ):
        text = _read(rel)
        assert "bit-identical across backends by construction" in text, rel
        assert "selected Keras backend" in text, rel
        # the bare, unqualified over-claim must be gone
        assert "so all backends are bit-identical by construction" not in text, rel


def test_score_flow_exact_is_scoped_to_the_trace() -> None:
    text = _read("packages/omnibias-score/src/omnibias/score/flow/__init__.py")
    assert "trace term is exact" in text
    assert "numerical solver" in text


def test_control_certificate_is_model_relative() -> None:
    text = _read("packages/omnibias-control/src/omnibias/control/__init__.py")
    assert "model-relative safety certificate" in text
    assert "certified safe control" not in text


def test_convex_active_set_is_not_founding_bias_collapse() -> None:
    text = _read("packages/omnibias-convex/src/omnibias/convex/warm_start.py")
    assert "delta -> 0" in text
    assert "**not** the founding" in text
    assert "do not conflate" in text.lower()


@pytest.mark.parametrize(
    "rel",
    [
        "packages/omnibias-symbolic/src/omnibias/symbolic/discovery.py",
        "packages/omnibias-symbolic/src/omnibias/symbolic/field_discovery.py",
    ],
)
def test_symbolic_stlsq_labelled_numerical_non_differentiable(rel: str) -> None:
    text = _read(rel)
    assert "exact closed form" in text, rel
    assert "non-differentiable" in text, rel


#: Modules that mention both collapse senses and must keep the disambiguation.
PENALTY_FILES = (
    "packages/omnibias-discrete/src/omnibias/discrete/evolution/_core.py",
    "packages/omnibias-discrete/src/omnibias/discrete/evolution/torch.py",
    "packages/omnibias-discrete/src/omnibias/discrete/evolution/jax.py",
    "packages/omnibias-discrete/src/omnibias/discrete/evolution/__init__.py",
    "packages/omnibias-convex/src/omnibias/convex/arrangement/_core.py",
    "packages/omnibias-convex/src/omnibias/convex/arrangement/torch.py",
    "packages/omnibias-convex/src/omnibias/convex/arrangement/jax.py",
    "packages/omnibias-convex/src/omnibias/convex/arrangement/__init__.py",
    "packages/omnibias-discrete/src/omnibias/discrete/csp/_core.py",
    "packages/omnibias-discrete/src/omnibias/discrete/csp/torch.py",
    "packages/omnibias-discrete/src/omnibias/discrete/csp/jax.py",
    "packages/omnibias-discrete/src/omnibias/discrete/csp/__init__.py",
    "packages/omnibias-measure/src/omnibias/measure/transport/_core.py",
    "packages/omnibias-measure/src/omnibias/measure/transport/torch.py",
    "packages/omnibias-measure/src/omnibias/measure/transport/jax.py",
    "packages/omnibias-measure/src/omnibias/measure/transport/__init__.py",
    "packages/omnibias-shape/src/omnibias/shape/morphology/_core.py",
    "packages/omnibias-shape/src/omnibias/shape/morphology/torch.py",
    "packages/omnibias-shape/src/omnibias/shape/morphology/jax.py",
    "packages/omnibias-shape/src/omnibias/shape/morphology/__init__.py",
    "packages/omnibias-core/src/omnibias/core/cubature.py",
    "packages/omnibias-fields/src/omnibias/fields/_core/quadrature.py",
    "packages/omnibias-core/src/omnibias/core/scale.py",
    "packages/omnibias-fields/src/omnibias/fields/scale.py",
    "packages/omnibias-verify/src/omnibias/verify/_core/localization.py",
    "packages/omnibias-verify/src/omnibias/verify/localization.py",
    "packages/omnibias-shape/src/omnibias/shape/topology/_core.py",
    "packages/omnibias-shape/src/omnibias/shape/topology/torch.py",
    "packages/omnibias-shape/src/omnibias/shape/topology/jax.py",
    "packages/omnibias-shape/src/omnibias/shape/topology/__init__.py",
    "packages/omnibias-difference/src/omnibias/difference/_core/singularity.py",
    "packages/omnibias-difference/src/omnibias/difference/singularity.py",
    "packages/omnibias-fields/src/omnibias/fields/singularity.py",
)


@pytest.mark.parametrize("rel", PENALTY_FILES)
def test_penalty_files_label_both_collapse_senses(rel: str) -> None:
    text = _read(rel)
    assert "founding bias collapse" in text, rel
    assert "delta -> 0" in text, rel
    assert "beta -> inf" in text, rel
    assert "feasibility" in text.lower(), rel
    assert "do not conflate" in text.lower(), rel
