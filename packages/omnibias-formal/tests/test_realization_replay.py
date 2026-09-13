# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
from fractions import Fraction as Q

import pytest
from omnibias.core.proof.realization_algebra_replay import (
    algebraic_realization_replay_certificate,
    laurent_closure_replay_certificate,
    polynomial_box_replay_certificate,
    sturm_isolation_replay_certificate,
)
from omnibias.core.proof.realization_replay import (
    interval_error_budget_certificate,
    interval_ldlt_replay_certificate,
    krawczyk_replay_certificate,
    polynomial_evaluation_certificate,
    polynomial_inequality_certificate,
    rank_replay_certificate,
    strict_box_inclusion_certificate,
)
from omnibias.core.realization.algebraic import RealAlgebraicField
from omnibias.core.realization.membership import classify_binary_quadratic
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.realization.rank import certify_matrix_rank
from omnibias.formal.mathlib_check import (
    check_certificate,
    classify_obligation,
    generate_obligation,
    mathlib_check_available,
)


def _certificates():
    real = classify_binary_quadratic([[1, 0, 2], [0, 1, 0]])
    boundary = classify_binary_quadratic([[1, 0, 0], [0, 1, 0]])
    x = SparsePolynomial.variable(1, 0)
    return [
        sturm_isolation_replay_certificate(RealAlgebraicField.sqrt(2)),
        algebraic_realization_replay_certificate(real.source, real.witness),
        laurent_closure_replay_certificate(boundary.source, boundary.closure_path),
        polynomial_box_replay_certificate(
            (x * x + 1,), ((0, 1),), {"equation": 0, "enclosure": [[1, 1], [2, 1]]}
        ),
        polynomial_evaluation_certificate((x * x,), (2,), (4,)),
        polynomial_inequality_certificate((x * x,), (2,), ("gt",)),
        rank_replay_certificate(certify_matrix_rank([[1, 0], [0, 1]])),
        interval_ldlt_replay_certificate([[(2, 3), (0, 0)], [(0, 0), (2, 3)]]),
        strict_box_inclusion_certificate(((1, 2),), ((0, 3),)),
        interval_error_budget_certificate((-Q(1, 10), Q(1, 10)), Q(1, 5)),
        krawczyk_replay_certificate(
            (0,), ((0, 0),), (((2, 2),),), ((Q(1, 2),),), ((-1, 1),), stationary_box=((0, 0),)
        ),
    ]


@pytest.mark.parametrize("certificate", _certificates())
def test_mathlib_operand_replay_generation_and_real_build(certificate):
    assert classify_obligation(certificate) == "realization_replay"
    assert generate_obligation(certificate) is not None
    if mathlib_check_available():
        assert check_certificate(certificate).verified
