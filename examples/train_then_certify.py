# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Train a tiny omnibias network, then **seal** a certified read-out over an input box.

Run::

    python examples/train_then_certify.py

The "train-then-certify" bridge: train a small ``tanh`` ``JetMLP`` (or, when
``omnibias-torch`` is unavailable, fall back to a fixed analytic net), then call
:func:`omnibias.verify.certify_trained_network` to *rigorously* enclose the
network's minimum over an input box, optionally read off its certified curvature
(flatness), and seal everything into a tamper-evident v1 certificate whose digest
recomputes from the body (so any post-hoc edit to a bound is detected).

Honest scope (inherited from interval branch-and-bound): **small** networks over
**low-dimensional** input boxes, activations ``tanh`` / ``sigmoid`` / ``gaussian``.
This is a certified read-out over an input region -- not million-parameter
training and not a continuum / global-regularity-grade statement.
"""

from __future__ import annotations

from typing import Any

from omnibias.core.proof.certificate import decode_interval
from omnibias.verify import certify_trained_network


def trained_net() -> tuple[Any, str]:
    """A briefly-trained ``tanh`` ``JetMLP`` if torch is available, else a fixed net."""
    try:
        import torch
        from omnibias.torch.architectures.pinn import JetMLP
    except ImportError:
        # duck-typed fallback (no torch): u = -exp(-x^2/2) - exp(-y^2/2), min -2 at origin
        class _Bump:
            def _layer_specs(self) -> list[tuple[list[list[float]], list[float] | None, str | None]]:
                return [
                    ([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], "gaussian"),
                    ([[-1.0, -1.0]], [0.0], None),
                ]

        return _Bump(), "fixed gaussian bump (install omnibias-torch to train instead)"

    torch.manual_seed(0)
    net = JetMLP(in_dim=2, hidden=4, out_dim=1, depth=2, base="tanh").double()
    xs = torch.rand(128, 2, dtype=torch.float64) * 3.0 - 1.5
    target = torch.zeros(128, 1, dtype=torch.float64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(50):
        opt.zero_grad()
        ((net(xs) - target) ** 2).mean().backward()
        opt.step()
    return net, "tanh JetMLP (trained 50 Adam steps)"


def main() -> None:
    net, note = trained_net()
    box = [(-1.5, 1.5), (-1.5, 1.5)]

    nc = certify_trained_network(net, box, tol=1e-2, max_boxes=200_000, flatness=True)

    enc = decode_interval(nc.certificate["payload"]["interval"])
    print(f"net:            {note}")
    print(f"box:            {box}")
    print(f"min enclosure:  [{enc.lo:.6f}, {enc.hi:.6f}]")
    print(f"argmin:         {tuple(round(v, 4) for v in nc.result.x)}")
    print(f"converged:      {nc.converged}  (gap {nc.result.gap:.2e}, tol {nc.result.tol:.0e})")
    if nc.flatness is not None:
        f = nc.flatness
        print(
            f"curvature:      Hessian eig in [{f.eig_min.lo:.4f}, {f.eig_max.hi:.4f}]  "
            f"PD={f.certified_positive_definite}"
        )
    print(f"layers digest:  {nc.layers_digest}")
    print(f"cert claim:     {nc.certificate['claim']}")
    print(f"cert verified:  {nc.verified}  (recomputed digest matches sealed body)")


if __name__ == "__main__":
    main()
