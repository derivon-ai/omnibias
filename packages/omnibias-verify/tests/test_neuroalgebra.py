# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
from dataclasses import replace
from fractions import Fraction as Q

from omnibias.core.proof.realization_replay import verify_replay_certificate
from omnibias.core.realization.polynomial import (
    PolynomialNetworkSpec,
    SparsePolynomial,
    compile_coefficient_map,
)
from omnibias.verify.neuroalgebra import bounded_realization_search, polynomial_interval


def test_interval_polynomial_contains_dense_rational_values():
    x, y = SparsePolynomial.variable(2, 0), SparsePolynomial.variable(2, 1)
    f = (x * x + x * y - 2 * y) ** 2 + x * x + 1
    box = ((-Q(1, 3), Q(1, 2)), (-2, 1))
    lo, hi = polynomial_interval(f, box)
    for i in range(31):
        for j in range(31):
            assert (
                lo
                <= f.evaluate(
                    (
                        box[0][0] + (box[0][1] - box[0][0]) * Q(i, 30),
                        box[1][0] + (box[1][1] - box[1][0]) * Q(j, 30),
                    )
                )
                <= hi
            )


def test_bounded_realization_exact_witness_and_box_scope():
    source = compile_coefficient_map(PolynomialNetworkSpec.monomial((1, 1)))
    witness = bounded_realization_search(source, (2,), ((1, 3),))
    assert witness.status == "realizable_in_box" and witness.witness.verify(source)
    outside = bounded_realization_search(source, (2,), ((-1, 1),))
    assert outside.status == "excluded_on_box" and outside.certificate.verify(source)
    assert verify_replay_certificate(outside.certificate.to_replay_certificate(source))
    assert not replace(outside.certificate, target=(Q(0),)).verify(source)


def test_budget_does_not_become_global_exclusion():
    source = compile_coefficient_map(PolynomialNetworkSpec.monomial((1, 1, 1)))
    result = bounded_realization_search(source, (2,), ((0, 2), (0, 2)), max_nodes=1)
    assert result.status == "inconclusive" and result.certificate is None
    assert result.unresolved_boxes
