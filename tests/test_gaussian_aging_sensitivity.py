# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certify true inverse gradients from untrusted aging-autodiff proposals.

The finite aging polynomial and the exact inverse have different derivatives.
Root and adjoint residuals bridge that gap without an inverse iteration or a
matrix factorization during certification.
"""

from fractions import Fraction as F
from importlib import import_module

import pytest
import torch
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    verify_certificate_digest,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_sensitivity import certify_gaussian_target_vjp
from omnibias.core.verified.mean_map import MeanMap
from omnibias.torch.gaussian_aging import gaussian_heat_inverse_aging


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _mp(mp, value):
    exact = F(value)
    return mp.mpf(exact.numerator)/exact.denominator


def _pair(value):
    exact = F(value)
    return [exact.numerator, exact.denominator]


def _independent_inverse_vjp(mp, model, target, upstream):
    """Direct Gaussian root, differentiated physical velocity, transpose solve.

    This oracle runs before the no-solve patch and does not call any library
    inverse, posterior, covariance, aging, or sensitivity implementation.
    """
    centers = [tuple(_mp(mp, value) for value in row) for row in model.centers]
    biases = tuple(_mp(mp, value) for value in model.biases)
    nu, age = _mp(mp, model.viscosity), _mp(mp, model.time)
    wanted = mp.matrix(tuple(_mp(mp, value) for value in target))

    def velocity(*point):
        logits = [bias-sum((x-a)**2 for x, a in zip(point, row, strict=True))/(4*nu*age)
                  for bias, row in zip(biases, centers, strict=True)]
        weights = [mp.exp(value-max(logits)) for value in logits]
        mean = [sum(p*row[axis] for p, row in zip(weights, centers, strict=True))/sum(weights)
                for axis in range(model.dimension)]
        return mp.matrix(tuple((x-mu)/age for x, mu in zip(point, mean, strict=True)))

    root = mp.findroot(lambda *point: velocity(*point)-wanted,
                       tuple(age*value for value in wanted), tol=mp.eps*32, maxsteps=80)

    def entry(i, j):
        def along_coordinate(z):
            point = tuple(z if axis == j else value for axis, value in enumerate(root))
            return velocity(*point)[i]
        return mp.diff(along_coordinate, root[j])

    jacobian = mp.matrix([[entry(i, j) for j in range(model.dimension)] for i in range(model.dimension)])
    gradient = mp.lu_solve(jacobian.T, mp.matrix(tuple(_mp(mp, value) for value in upstream)))
    return tuple(root), tuple(gradient)


def _proposal(model, target, upstream, order):
    wanted = torch.tensor(tuple(float(value) for value in target), dtype=torch.float64, requires_grad=True)
    centers = torch.tensor(tuple(tuple(float(value) for value in row) for row in model.centers),
                           dtype=torch.float64)
    biases = torch.tensor(tuple(float(value) for value in model.biases), dtype=torch.float64)
    result = gaussian_heat_inverse_aging(wanted, centers, biases, float(model.viscosity),
                                        float(model.time), order=order)
    assert result.valid.item()
    loss = (result.value*torch.tensor(upstream, dtype=torch.float64)).sum()
    proposal = tuple(torch.autograd.grad(loss, wanted)[0].detach().tolist())
    scaled_point = tuple((result.value.detach()/float(model.time)).tolist())
    return scaled_point, proposal


def _forbid_solves(patch):
    def forbidden(*args, **kwargs):
        pytest.fail("inverse iteration, factorization, or dense Jacobian used during certification")

    inverse = import_module("omnibias.core.verified.inverse_gaussian_heat")
    verified = import_module("omnibias.core.verified")
    exact_linear = import_module("omnibias.core._rational_linear")
    interval_linear = import_module("omnibias.core.verified.linalg")
    patch.setattr(inverse, "inverse_gaussian_heat", forbidden)
    patch.setattr(verified, "inverse_gaussian_heat", forbidden)
    patch.setattr(exact_linear, "factor_positive_definite", forbidden)
    patch.setattr(exact_linear.RationalLDLT, "solve", forbidden)
    for name in ("_float_inverse", "_symmetric_ldlt_solve", "interval_solve"):
        patch.setattr(interval_linear, name, forbidden)
    for name in ("solve", "cholesky", "cholesky_ex", "lu_factor"):
        patch.setattr(torch.linalg, name, forbidden)
    patch.setattr(MeanMap, "jacobian", forbidden)
    patch.setattr(GaussianHeatMap, "scaled_jacobian", forbidden)
    patch.setattr(GaussianHeatMap, "float_scaled_fixed_point", forbidden)


FAMILIES = (
    (((0,), (1,), (3,)), (F(1, 8), F(-1, 4), F(3, 8)), F(1), F(8), (0.25,), (0.75,)),
    (((0, 0), (1, 0), (0, 2), (2, 1)), (F(1, 8), F(-1, 4), F(3, 8), F(-1, 8)),
     F(3, 2), F(4), (0.25, -0.125), (0.75, -0.5)),
)


@pytest.mark.parametrize("family", FAMILIES, ids=("scalar", "coupled_2d"))
def test_aging_ad_certifies_true_inverse_vjp_without_iteration_or_factorization(monkeypatch, family):
    mp = pytest.importorskip("mpmath")
    centers, biases, viscosity, time, floating_target, upstream = family
    # Bind the exact problem to the actual binary64 target and upstream values.
    target = tuple(F(value) for value in floating_target)
    exact_upstream = tuple(F(value) for value in upstream)
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time)
    proposals = {order: _proposal(model, target, upstream, order) for order in (1, 4, 8)}
    with mp.workdps(90):
        root, expected = _independent_inverse_vjp(mp, model, target, exact_upstream)

        # Residual certification remains available; only proposal-generating
        # solves and matrix constructions are prohibited inside this context.
        with monkeypatch.context() as patch:
            _forbid_solves(patch)
            certificates = {order: certify_gaussian_target_vjp(model, target, point, upstream, proposal)
                            for order, (point, proposal) in proposals.items()}

        errors = {}
        for order, certificate in certificates.items():
            assert certificate is not None
            point, proposed = proposals[order]
            assert certificate.proposal == proposed
            assert certificate.root_certificate.scaled_point == point
            assert verify_certificate_digest(certificate.certificate)
            assert certificate_is_unconditional(certificate.certificate)
            assert certificate.certificate["payload"]["upstream"] == list(upstream)
            assert certificate.certificate["payload"]["proposal"] == list(proposed)
            payload = certificate.root_certificate.certificate["payload"]
            assert payload["centers"] == [[_pair(value) for value in row] for row in model.centers]
            assert payload["biases"] == [_pair(value) for value in model.biases]
            assert payload["target"] == [_pair(value) for value in target]
            assert payload["time"] == _pair(time) and payload["viscosity"] == _pair(viscosity)
            for true, interval in zip(root, certificate.root_certificate.enclosure, strict=True):
                assert _mp(mp, interval.lo) <= true <= _mp(mp, interval.hi)
            for true, interval in zip(expected, certificate.enclosure, strict=True):
                assert _mp(mp, interval.lo) <= true <= _mp(mp, interval.hi)
            errors[order] = mp.sqrt(sum((_mp(mp, value)-true)**2
                                        for value, true in zip(proposed, expected, strict=True)))
            assert errors[order] <= _mp(mp, certificate.error_bound)

        # These two fixed examples improve with degree. This is intentionally
        # not a monotonicity claim for arbitrary families or every adjacent N.
        assert certificates[8].error_bound < certificates[1].error_bound/1000
        assert errors[8] < errors[1]/1000
        assert errors[1] > mp.mpf("1e-5")  # Finite-polynomial AD is not the exact inverse VJP.

        # No claimed gradient identity is trusted: a deliberately bad proposal
        # still certifies, but pays for its much larger adjoint residual.
        point, proposed = proposals[8]
        perturbed = tuple(value+0.25 for value in proposed)
        with monkeypatch.context() as patch:
            _forbid_solves(patch)
            loose = certify_gaussian_target_vjp(model, target, point, upstream, perturbed)
        assert loose is not None and loose.error_bound > certificates[8].error_bound*1000
        for true, interval in zip(expected, loose.enclosure, strict=True):
            assert _mp(mp, interval.lo) <= true <= _mp(mp, interval.hi)
