# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""P4: certified read-outs of a trained model (H5) via ``omnibias.verify``.

The ``mse_tanh`` model is a plain ``Linear``/``Tanh`` stack (``MLP1D.net`` is an
:class:`torch.nn.Sequential`), so :func:`omnibias.verify.torch.network_from_sequential`
ingests it into a backend-neutral :class:`~omnibias.verify.Network` and the pure-Python
verifier produces *sound* enclosures:

* a rigorous Lipschitz upper bound over the standardised input box (cheap: one interval
  Jacobian), and
* a certified robustness margin at correctly-classified test points (branch-and-bound on
  ``out[true] - out[j]`` over an L-inf ball).

Honest scope (inherited from interval branch-and-bound): these are *local* certificates
over an input region for a *small* net. The input-space Hessian *flatness* enclosure is
exposed but off by default -- over the full 40-dim box it is expensive and typically
returns a loose / ``UNKNOWN`` bracket; it is meaningful only on very small nets / boxes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from examples.mnist1d_double_descent.data import DataBundle, one_hot
from examples.mnist1d_double_descent.models import build_model, count_parameters


@dataclass(frozen=True)
class CertifiedReadout:
    """Sound enclosures for one trained model (all bounds are rigorous)."""

    register: str
    width: int
    n_params: int
    eps: float
    box_halfwidth: float
    n_points: int
    lipschitz_inf: float
    robust_frac: float
    min_margin_mean: float
    flatness_eig_min: float | None
    flatness_eig_max: float | None
    certified_pd: bool | None
    train_err: float
    test_err: float
    notes: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _correct_points(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, n_points: int
) -> tuple[list[list[float]], list[int]]:
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    correct = (pred == y).nonzero(as_tuple=True)[0][:n_points]
    xs = [x[i].double().tolist() for i in correct.tolist()]
    ys = [int(y[i]) for i in correct.tolist()]
    return xs, ys


def certify_model(
    model: nn.Module,
    *,
    register: str,
    width: int,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    eps: float = 0.05,
    box_halfwidth: float = 3.0,
    n_points: int = 8,
    max_boxes: int = 256,
    order: int = 2,
    flatness: bool = False,
    flatness_dim_cap: int = 8,
    train_err: float = float("nan"),
    test_err: float = float("nan"),
) -> CertifiedReadout:
    """Produce sound Lipschitz / robustness (and optional flatness) enclosures for ``model``."""
    from omnibias.core.verified import Interval
    from omnibias.verify import certified_network_flatness, certify_robustness, lipschitz_bound
    from omnibias.verify.torch import network_from_sequential

    sequential = model.net if hasattr(model, "net") else model
    net = network_from_sequential(sequential)
    seq_len = int(x_test.shape[1])
    box = [Interval(-box_halfwidth, box_halfwidth) for _ in range(seq_len)]
    lip = float(lipschitz_bound(net, box, norm="inf"))

    xs, ys = _correct_points(model, x_test, y_test, n_points)
    certified = 0
    margins: list[float] = []
    for x0, y0 in zip(xs, ys, strict=True):
        cert = certify_robustness(net, x0, eps, y0, order=order, max_boxes=max_boxes)
        certified += int(cert.certified)
        m = cert.min_margin
        if m == m and abs(m) != float("inf"):
            margins.append(float(m))
    robust_frac = certified / max(len(xs), 1)
    min_margin_mean = sum(margins) / len(margins) if margins else float("nan")

    eig_min: float | None = None
    eig_max: float | None = None
    certified_pd: bool | None = None
    notes = "Lipschitz (l-inf) + robustness margin; sound local enclosures."
    if flatness:
        if seq_len <= flatness_dim_cap:
            flat = certified_network_flatness(net, box, component=0)
            eig_min = float(flat.eig_min.lo)
            eig_max = float(flat.eig_max.hi)
            certified_pd = bool(flat.certified_positive_definite)
            notes += " Flatness = certified input-space Hessian eigenvalue bracket."
        else:
            notes += f" Flatness skipped: dim {seq_len} > cap {flatness_dim_cap} (too expensive)."

    return CertifiedReadout(
        register=register,
        width=width,
        n_params=count_parameters(model),
        eps=eps,
        box_halfwidth=box_halfwidth,
        n_points=len(xs),
        lipschitz_inf=lip,
        robust_frac=robust_frac,
        min_margin_mean=min_margin_mean,
        flatness_eig_min=eig_min,
        flatness_eig_max=eig_max,
        certified_pd=certified_pd,
        train_err=train_err,
        test_err=test_err,
        notes=notes,
    )


def train_and_certify(
    width: int,
    bundle: DataBundle,
    *,
    seed: int = 0,
    steps: int = 200,
    lr: float = 1e-2,
    eps: float = 0.05,
    n_points: int = 8,
    max_boxes: int = 256,
    order: int = 2,
    device: str = "cpu",
    flatness: bool = False,
) -> CertifiedReadout:
    """Train one small ``mse_tanh`` model at ``width`` (float64) and certify it."""
    model, metrics = _train_mse_tanh(width, bundle, seed=seed, steps=steps, lr=lr, device=device)
    return certify_model(
        model,
        register="mse_tanh",
        width=width,
        x_test=bundle.x_test.to(torch.float64).to(device),
        y_test=bundle.y_test.to(device),
        eps=eps,
        n_points=n_points,
        max_boxes=max_boxes,
        order=order,
        flatness=flatness,
        train_err=metrics["train_err"],
        test_err=metrics["test_err"],
    )


def _train_mse_tanh(
    width: int, bundle: DataBundle, *, seed: int, steps: int, lr: float, device: str
) -> tuple[nn.Module, dict[str, float]]:
    """Train an mse_tanh model in float64 (so ingestion is exact) and report train/test error."""
    model = build_model(
        "mse_tanh", in_dim=bundle.in_dim, hidden=width, num_classes=bundle.num_classes,
        seed=seed, device=device, dtype=torch.float64,
    )
    x = bundle.x_train.to(torch.float64).to(device)
    y = bundle.y_train.to(device)
    target = one_hot(y, bundle.num_classes).to(torch.float64)
    x_te = bundle.x_test.to(torch.float64).to(device)
    y_te = bundle.y_test.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        train_err = float((model(x).argmax(dim=1) != y).to(torch.float64).mean())
        test_err = float((model(x_te).argmax(dim=1) != y_te).to(torch.float64).mean())
    return model, {"train_err": train_err, "test_err": test_err}


def main(argv: list[str] | None = None) -> None:
    """CLI: train + certify a list of ``mse_tanh`` widths and write ``certified.json``."""
    import argparse
    import json
    from pathlib import Path

    from examples.mnist1d_double_descent.data import Mnist1DConfig, load_mnist1d

    p = argparse.ArgumentParser(description="P4 certified read-outs of small mse_tanh nets.")
    p.add_argument("--widths", nargs="+", type=int, default=[4, 8, 12, 16])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--n-points", type=int, default=8)
    p.add_argument("--max-boxes", type=int, default=64)
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--label-noise", type=float, default=0.15)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--flatness", action="store_true", help="also enclose input-space Hessian (small nets only)")
    p.add_argument("--scratch-dir", default="artifacts/omnibias_mnist1d/data")
    p.add_argument("--out", default="examples/mnist1d_double_descent/results/p4/certified.json")
    args = p.parse_args(argv)

    bundle = load_mnist1d(
        Mnist1DConfig(n_train=args.n_train, n_test=args.n_test),
        label_noise=args.label_noise, noise_seed=args.seed, scratch_dir=args.scratch_dir,
    )
    readouts = []
    for width in args.widths:
        r = train_and_certify(
            width, bundle, seed=args.seed, steps=args.steps, lr=args.lr, eps=args.eps,
            n_points=args.n_points, max_boxes=args.max_boxes, order=args.order,
            flatness=args.flatness,
        )
        readouts.append(r.as_dict())
        print(
            f"w={width:4d} P={r.n_params:5d} lipschitz={r.lipschitz_inf:.3g} "
            f"robust_frac={r.robust_frac:.3f} margin={r.min_margin_mean:.3g} "
            f"test_err={r.test_err:.3f}"
        )
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(readouts, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()


__all__ = ["CertifiedReadout", "certify_model", "main", "train_and_certify"]
