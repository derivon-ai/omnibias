# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Train-then-certify bridge: seal a trained network's certified read-out.

:func:`certify_trained_network` runs the certified global minimum enclosure over an
input box (and, on request, the certified flatness / strict-local-min curvature
read-out) and **seals** the result into a tamper-evident v1
:class:`~omnibias.core.proof.certificate.Cert`.  The tests cover

* **roundtrip / soundness** -- the sealed certificate carries the same enclosure as the
  :class:`GlobalMinResult`, its digest verifies, and the enclosure is sound vs a grid;
* **tamper detection** -- editing any sealed bound or a ``meta`` field breaks the digest;
* **honest ``converged`` flag** -- a starved ``max_boxes`` budget yields ``converged is
  False`` (and ``honesty["global_min_certified"] is False``) while the enclosure stays
  sound;
* **provenance** -- the ingested-weight digest matches :func:`verified_layer_bundle`;
* **bundled curvature** -- ``flatness`` / ``strict_local_min`` land in the result and the
  certificate ``meta``.
"""

from __future__ import annotations

import copy
import math

import pytest
from omnibias.core.proof.certificate import decode_interval, verify_certificate_digest
from omnibias.verify import (
    NetworkCertificate,
    certify_trained_network,
    verified_layer_bundle,
)

# --------------------------------------------------------------------------- #
# fixtures: duck-typed JetMLP-like nets (a `_layer_specs()` is all the bridge needs)
# --------------------------------------------------------------------------- #

Layer = tuple[list[list[float]], list[float] | None, str | None]


def _eval_net(layers: list[Layer], x: list[float]) -> list[float]:
    v = list(x)
    for weight, bias, name in layers:
        pre = []
        for i, row in enumerate(weight):
            s = sum(row[j] * v[j] for j in range(len(v)))
            if bias is not None:
                s += bias[i]
            pre.append(s)
        if name == "tanh":
            v = [math.tanh(z) for z in pre]
        elif name == "gaussian":
            v = [math.exp(-z * z / 2.0) for z in pre]
        elif name is None:
            v = pre
        else:  # pragma: no cover - defensive
            raise ValueError(name)
    return v


def _grid_min(layers: list[Layer], box: list[tuple[float, float]], n: int = 41) -> float:
    (xlo, xhi), (ylo, yhi) = box
    best = math.inf
    for i in range(n):
        for j in range(n):
            x = xlo + (xhi - xlo) * i / (n - 1)
            y = ylo + (yhi - ylo) * j / (n - 1)
            best = min(best, _eval_net(layers, [x, y])[0])
    return best


# u(x, y) = -exp(-x^2/2) - exp(-y^2/2): min -2 at the origin, Hessian ~ I near it (exact
# at the min -> converges in one box; the easy soundness/curvature fixture).
BUMP: list[Layer] = [
    ([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], "gaussian"),
    ([[-1.0, -1.0]], [0.0], None),
]
# A tanh MLP whose interval-bound-propagation value enclosure is genuinely loose, so
# branch-and-bound must refine (a starved budget can't converge) -- the honesty fixture.
TANH: list[Layer] = [
    ([[1.5, -0.7], [0.8, 1.2], [-1.1, 0.5], [0.3, -1.4]], [0.1, -0.2, 0.05, 0.2], "tanh"),
    ([[0.9, -1.3, 0.6, 0.7]], [0.15], None),
]


class _Net:
    """Minimal duck-typed ``JetMLP``: a ``_layer_specs()`` is all the bridge needs."""

    def __init__(self, layers: list[Layer]) -> None:
        self._layers = layers

    def _layer_specs(self) -> list[Layer]:
        return list(self._layers)


BOX = [(-2.0, 2.0), (-2.0, 2.0)]
NEAR_ORIGIN = [(-0.3, 0.3), (-0.3, 0.3)]


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #


def test_roundtrip_sealed_cert_verifies_and_is_sound() -> None:
    nc = certify_trained_network(_Net(BUMP), BOX, tol=1e-3)
    assert isinstance(nc, NetworkCertificate)
    assert nc.converged
    assert nc.verified  # digest matches body

    # the sealed certificate carries exactly the GlobalMinResult enclosure ...
    sealed = decode_interval(nc.certificate["payload"]["interval"])
    assert sealed.lo == nc.result.f_lower
    assert sealed.hi == nc.result.f_upper
    # ... which is sound vs a dense grid and brackets the true min (-2).
    assert nc.result.f_lower <= _grid_min(BUMP, BOX)
    assert nc.result.f_lower <= -2.0 <= nc.result.f_upper

    honesty = nc.certificate["honesty"]
    assert honesty["unproven_claim"] is False
    assert honesty["global_min_certified"] is True


def test_provenance_digest_matches_bundle() -> None:
    net = _Net(BUMP)
    nc = certify_trained_network(net, BOX, tol=1e-3)
    expected = verified_layer_bundle(net).metadata["layers_digest"]
    assert nc.layers_digest == expected
    assert nc.certificate["meta"]["layers_digest"] == expected
    # meta records the reproducible read-out context.
    meta = nc.certificate["meta"]
    assert meta["kind"] == "trained_network_readout"
    assert meta["component"] == 0
    assert meta["box"] == [[-2.0, 2.0], [-2.0, 2.0]]


def test_digest_tamper_on_bound_is_detected() -> None:
    nc = certify_trained_network(_Net(BUMP), BOX, tol=1e-3)
    assert nc.verified
    tampered = copy.deepcopy(nc.certificate)
    # forge a *tighter* (fraudulently better) lower bound
    tampered["payload"]["interval"]["lo"] = "0.0"
    assert not verify_certificate_digest(tampered)


def test_digest_tamper_on_meta_is_detected() -> None:
    nc = certify_trained_network(_Net(BUMP), BOX, tol=1e-3)
    tampered = copy.deepcopy(nc.certificate)
    # flip the honest convergence record without re-sealing
    tampered["meta"]["converged"] = not tampered["meta"]["converged"]
    assert not verify_certificate_digest(tampered)


def test_converged_flag_is_honest_under_starved_budget() -> None:
    # the loose tanh enclosure needs real refinement; a 1-box budget can't converge
    nc = certify_trained_network(_Net(TANH), BOX, tol=1e-9, max_boxes=1)
    assert not nc.converged
    assert nc.certificate["honesty"]["global_min_certified"] is False
    assert nc.certificate["meta"]["converged"] is False
    # the enclosure is still unconditionally sound (lower bound under-estimates the grid)
    assert nc.verified
    assert nc.result.f_lower <= _grid_min(TANH, BOX)


def test_flatness_bundled_and_pd_near_origin() -> None:
    nc = certify_trained_network(_Net(BUMP), NEAR_ORIGIN, tol=1e-3, flatness=True)
    assert nc.flatness is not None
    assert nc.flatness.certified_positive_definite  # curvature ~ diag(1, 1) near origin
    meta_flat = nc.certificate["meta"]["flatness"]
    assert meta_flat["certified_positive_definite"] is True
    assert meta_flat["eig_min"][0] > 0.0
    assert 0.0 < meta_flat["sharpness"] < 1.5


def test_strict_local_min_bundled() -> None:
    nc = certify_trained_network(
        _Net(BUMP), NEAR_ORIGIN, tol=1e-3, strict_local_min=True
    )
    assert nc.strict_local_min is True
    assert nc.certificate["meta"]["strict_local_min"] is True


def test_accepts_raw_layer_list_like_net() -> None:
    # a raw [(W, b, name), ...] list certifies identically to the wrapped net
    from_list = certify_trained_network(BUMP, BOX, tol=1e-3)
    from_net = certify_trained_network(_Net(BUMP), BOX, tol=1e-3)
    assert from_list.verified
    assert from_list.result.f_lower == from_net.result.f_lower
    assert from_list.result.f_upper == from_net.result.f_upper
    assert from_list.layers_digest == from_net.layers_digest


def test_torch_jetmlp_train_then_certify_end_to_end() -> None:
    # a genuinely *trained* omnibias network, certified + sealed end to end.
    torch = pytest.importorskip("torch")
    from omnibias.torch.architectures.pinn import JetMLP

    torch.manual_seed(0)
    net = JetMLP(in_dim=2, hidden=4, out_dim=1, depth=2, base="tanh").double()

    # a couple of optimizer steps so the weights are "trained", not just init
    xs = torch.rand(64, 2, dtype=torch.float64) * 3.0 - 1.5
    target = torch.zeros(64, 1, dtype=torch.float64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(15):
        opt.zero_grad()
        loss = ((net(xs) - target) ** 2).mean()
        loss.backward()
        opt.step()

    box = [(-1.5, 1.5), (-1.5, 1.5)]
    nc = certify_trained_network(net, box, tol=1e-2, max_boxes=200_000)
    assert nc.verified
    assert nc.converged

    @torch.no_grad()
    def fwd(px: float, py: float) -> float:
        z = torch.tensor([px, py], dtype=torch.float64)
        for w, b, spec in net._layer_specs():
            z = z @ w.T
            if b is not None:
                z = z + b
            if spec is not None:
                z = torch.tanh(z)
        return float(z.squeeze(-1))

    axis = torch.linspace(-1.5, 1.5, 41).tolist()
    gmin = min(fwd(px, py) for px in axis for py in axis)
    assert nc.result.f_lower <= gmin  # sealed lower bound is sound vs the torch net
    assert nc.result.f_upper <= gmin + 5e-2
