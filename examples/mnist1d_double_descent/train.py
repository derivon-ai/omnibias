# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""One full-batch training loop shared by every optimizer arm.

MNIST-1D is tiny (4000x40), so every arm trains on the **full batch**: this gives
a single, well-defined training-objective Hessian to instrument and removes
minibatch noise as a confound in the optimizer comparison (a random subset is
drawn per step only when ``batch_size`` is set, for the first-order arms).

The loop dispatches on ``arm.driver`` (see
:mod:`~examples.mnist1d_double_descent.arms`) to honour each optimizer's closure
convention, logs train/test error per step, detects the **interpolation step**
(train error first reaches zero on the noisy training labels), and takes periodic
exact-curvature snapshots (:func:`~examples.mnist1d_double_descent.curvature.spectrum_snapshot`)
including one at interpolation and one at the end. Results serialise to JSON under
a scratch directory.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from omnibias.torch.optim import gauss_newton_fisher
from torch import Tensor, nn
from torch.func import functional_call

from examples.mnist1d_double_descent.arms import OptimizerArm
from examples.mnist1d_double_descent.curvature import CurvatureSnapshot, spectrum_snapshot
from examples.mnist1d_double_descent.data import DataBundle, one_hot
from examples.mnist1d_double_descent.eos import EdgeOfStabilityLR
from examples.mnist1d_double_descent.models import Register, build_model, count_parameters

from omnibias.curvature.torch import sam_objective, sharpness_aware_loss  # isort: skip


@dataclass
class RunConfig:
    """The knobs that define one training run."""

    register: Register
    arm: str
    width: int
    depth: int
    seed: int
    label_noise: float
    steps: int
    lr: float
    batch_size: int | None = None
    log_every: int = 1  # record full train/test metrics every k steps (+ step 0 and final)
    curvature: bool = True  # take exact-curvature snapshots (disable for fast sweeps / tests)
    curvature_every: int = 0  # 0 -> auto (~5 evenly spaced snapshots)
    curv_batch: int = 0  # 0 -> full train set; else a fixed subsample for the Hessian
    curv_power_iters: int = 40  # matrix-free power-iteration count (wide nets)
    curv_hutch: int = 8  # matrix-free Hutchinson samples (wide nets)
    dense_max_params: int = 1500
    device: str = "cpu"
    metric_batch: int = 512
    source: str = ""


@dataclass
class RunResult:
    """Outcome + full instrumented history of one run."""

    config: dict[str, object]
    n_params: int
    final_train_loss: float
    final_train_err: float
    final_test_err: float
    best_test_err: float
    interpolation_step: int  # -1 if never interpolated
    err_at_interpolation: float
    wall_time: float
    opt_state_bytes: int = 0  # persistent optimiser-state memory (excludes the model params)
    update_time_s: float = 0.0  # summed wall-clock of just the optimiser update steps
    status: str = "ok"
    error: str = ""
    history: list[dict[str, object]] = field(default_factory=list)
    curvature_at_interpolation: dict[str, object] | None = None
    curvature_final: dict[str, object] | None = None


def _optimizer_state_bytes(opt: object) -> int:
    """Total bytes of *persistent optimiser state* -- the memory-footprint telemetry.

    Counts the built-in ``.state`` buffers (Adam's ``m``/``v``, SGD's momentum) and the
    private buffer attributes the omnibias optimisers keep (``_m`` / ``_c`` / ``_d`` /
    ``_v`` / ``_gS``). The model parameters themselves are deliberately excluded, so this
    is exactly the extra memory an optimizer costs -- e.g. Adam ``2P``, FrugalCurvature
    ``P + #tensors``.
    """
    seen: set[int] = set()
    total = 0

    def _add(t: object) -> None:
        nonlocal total
        if isinstance(t, Tensor) and id(t) not in seen:
            seen.add(id(t))
            total += t.numel() * t.element_size()

    state = getattr(opt, "state", None)
    if isinstance(state, dict):
        for entry in state.values():
            if isinstance(entry, dict):
                for v in entry.values():
                    _add(v)
    for name in ("_m", "_c", "_d", "_v", "_gS"):
        val = getattr(opt, name, None)
        if isinstance(val, Tensor):
            _add(val)
        elif isinstance(val, list | tuple):
            for item in val:
                _add(item)
    return total


# ---------------------------------------------------------------------------
# Losses / metrics
# ---------------------------------------------------------------------------


def canonical_loss(
    model: nn.Module, x: Tensor, y: Tensor, onehot: Tensor, register: Register
) -> Tensor:
    """The per-register training objective whose Hessian we instrument."""
    out = model(x)
    if register == "ce_relu":
        return torch.nn.functional.cross_entropy(out, y)
    return ((out - onehot) ** 2).mean()


def _residual(model: nn.Module, x: Tensor, onehot: Tensor) -> Tensor:
    return (model(x) - onehot).reshape(-1)


@torch.no_grad()
def _accuracy(out: Tensor, y: Tensor) -> float:
    return float((out.argmax(dim=1) == y).to(torch.float64).mean())


@torch.no_grad()
def _weight_norm(params: list[Tensor]) -> float:
    sq = torch.zeros((), dtype=torch.float64)
    for p in params:
        sq = sq + (p.detach().to(torch.float64) ** 2).sum()
    return float(torch.sqrt(sq))


# ---------------------------------------------------------------------------
# Driver-specific update
# ---------------------------------------------------------------------------


def _sharpness_objective(
    arm: OptimizerArm, base: Tensor, params: list[Tensor], gen: torch.Generator | None
) -> Tensor:
    kind = arm.sharpness_kind
    if kind == "sam_exact":
        return sam_objective(
            base, params, rho=float(arm.hypers["rho"]), iters=int(arm.hypers.get("iters", 20)),
            generator=gen,
        )
    if kind == "sharpness_reg":
        return sharpness_aware_loss(
            base, params, lam=float(arm.hypers["lam"]),
            measure=str(arm.hypers.get("measure", "frobenius")),
            n_samples=int(arm.hypers.get("n_samples", 4)), generator=gen,
        )
    if kind == "sam_stochastic":
        # First-order (linear ascent term only) SAM proxy: penalise ||grad L||.
        grads = torch.autograd.grad(base, params, create_graph=True)
        gnorm = torch.sqrt(sum(((g * g).sum() for g in grads), start=base.new_zeros(())))
        return base + float(arm.hypers["rho"]) * gnorm
    raise ValueError(f"unknown sharpness_kind {kind!r}")


def _attach_gn_metric(
    opt: torch.optim.Optimizer, model: nn.Module, x: Tensor, onehot: Tensor, *, metric_batch: int
) -> None:
    """Give a ``NaturalGradient`` optimizer a (dense) Gauss-Newton Fisher metric provider."""
    params = list(opt._params)  # type: ignore[attr-defined]  # matches model.parameters() order
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    n = int(x.shape[0])
    xs = x[:metric_batch] if 0 < metric_batch < n else x
    os = onehot[:metric_batch] if 0 < metric_batch < n else onehot

    def residual_of_flat(flat: Tensor) -> Tensor:
        pd: dict[str, Tensor] = {}
        idx = 0
        for nm, sh, ne in zip(names, shapes, numels, strict=True):
            pd[nm] = flat[idx : idx + ne].reshape(sh)
            idx += ne
        out = functional_call(model, pd, (xs,))
        return (out - os).reshape(-1)

    def metric(flat: Tensor) -> Tensor:
        fisher, _g = gauss_newton_fisher(residual_of_flat, flat)
        return fisher

    opt.metric = metric  # type: ignore[attr-defined]


def _update(
    arm: OptimizerArm,
    opt: torch.optim.Optimizer,
    model: nn.Module,
    xb: Tensor,
    yb: Tensor,
    ob: Tensor,
    register: Register,
    params: list[Tensor],
    gen: torch.Generator | None,
    controller: EdgeOfStabilityLR | None = None,
) -> dict[str, float] | None:
    """Take one optimizer step; return optional per-step telemetry (EoS eta / lambda_max)."""
    driver = arm.driver
    if driver == "standard":
        opt.zero_grad(set_to_none=True)
        canonical_loss(model, xb, yb, ob, register).backward()
        opt.step()
    elif driver in ("scalar", "kfac", "natural"):

        def closure_scalar() -> Tensor:
            return canonical_loss(model, xb, yb, ob, register)

        opt.step(closure_scalar)  # type: ignore[arg-type]
    elif driver == "residual":

        def closure_res() -> Tensor:
            return _residual(model, xb, ob)

        opt.step(closure_res)  # type: ignore[arg-type]
    elif driver == "sharpness":
        opt.zero_grad(set_to_none=True)
        base = canonical_loss(model, xb, yb, ob, register)
        _sharpness_objective(arm, base, params, gen).backward()
        opt.step()
    elif driver == "eos":
        if controller is None:
            raise ValueError("eos driver requires an EdgeOfStabilityLR controller")
        # One forward: the controller probes the exact top eigenvalue off this loss's
        # (retained) graph to set the rate, then the same loss is backpropped for the
        # plain-GD step at that rate (lr overridden in-place).
        opt.zero_grad(set_to_none=True)
        loss = canonical_loss(model, xb, yb, ob, register)
        telemetry = controller.rate(loss, params)
        loss.backward()
        for group in opt.param_groups:
            group["lr"] = controller.last_eta
        opt.step()
        return telemetry
    else:
        raise ValueError(f"unknown driver {driver!r}")
    return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _auto_curvature_every(steps: int, curvature_every: int) -> int:
    if curvature_every > 0:
        return curvature_every
    return max(1, steps // 5)


def train_run(bundle: DataBundle, arm: OptimizerArm, cfg: RunConfig, *, log: bool = False) -> RunResult:
    """Train one ``(arm, register, width, seed)`` run and return its instrumented result."""
    torch.manual_seed(cfg.seed)
    device = cfg.device
    bundle = bundle.to(device)
    x_tr, y_tr = bundle.x_train, bundle.y_train
    x_te, y_te = bundle.x_test, bundle.y_test
    ob_tr = one_hot(y_tr, bundle.num_classes)

    model = build_model(
        cfg.register, in_dim=bundle.in_dim, hidden=cfg.width,
        num_classes=bundle.num_classes, depth=cfg.depth, seed=cfg.seed, device=device,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = count_parameters(model)
    opt = arm.build(model, lr=cfg.lr)
    if arm.driver == "natural":
        _attach_gn_metric(opt, model, x_tr, ob_tr, metric_batch=cfg.metric_batch)
    controller: EdgeOfStabilityLR | None = None
    if arm.driver == "eos":
        h = arm.hypers
        controller = EdgeOfStabilityLR(
            c=float(h.get("c", 0.9)),
            momentum=float(h.get("momentum", 0.0)),
            eta_min=float(h.get("eta_min", 1e-4)),
            eta_max=float(h.get("eta_max", 1.0)),
            probe_iters=int(h.get("probe_iters", 12)),
            measure_every=int(h.get("measure_every", 1)),
            ema=float(h.get("ema", 0.5)),
            seed=cfg.seed + 3,
        )

    cgen: torch.Generator | None = None
    if device == "cpu":
        cgen = torch.Generator().manual_seed(cfg.seed + 1)

    every = _auto_curvature_every(cfg.steps, cfg.curvature_every)
    n = int(x_tr.shape[0])
    batch_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 7)

    # A fixed subsample of the training set for the Hessian (0 -> full batch). Using a
    # fixed subset keeps the exact-curvature snapshots affordable on CPU and is the
    # standard "Hessian on a held batch" convention; it is deterministic per seed.
    if 0 < cfg.curv_batch < n:
        cidx = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed + 11))[
            : cfg.curv_batch
        ].to(device)
        xc, yc, obc = x_tr[cidx], y_tr[cidx], ob_tr[cidx]
    else:
        xc, yc, obc = x_tr, y_tr, ob_tr

    history: list[dict[str, object]] = []
    interp_step = -1
    err_interp = -1.0
    curv_interp: dict[str, object] | None = None
    t0 = time.perf_counter()

    def snapshot() -> CurvatureSnapshot:
        loss = canonical_loss(model, xc, yc, obc, cfg.register)
        return spectrum_snapshot(
            loss, params, dense_max_params=cfg.dense_max_params,
            power_iters=cfg.curv_power_iters, hutch_samples=cfg.curv_hutch, seed=cfg.seed + 2,
        )

    def log_metrics(step: int, take_curv: bool) -> dict[str, object]:
        nonlocal interp_step, err_interp, curv_interp
        with torch.no_grad():
            out_tr = model(x_tr)
            out_te = model(x_te)
            train_loss = float(canonical_loss(model, x_tr, y_tr, ob_tr, cfg.register))
            test_loss = float(canonical_loss(model, x_te, y_te, one_hot(y_te, bundle.num_classes), cfg.register))
        acc_tr = _accuracy(out_tr, y_tr)
        acc_tr_clean = _accuracy(out_tr, bundle.y_train_clean)
        acc_te = _accuracy(out_te, y_te)
        entry: dict[str, object] = {
            "step": step,
            "train_loss": train_loss,
            "train_err": 1.0 - acc_tr,
            "train_err_clean": 1.0 - acc_tr_clean,
            "test_loss": test_loss,
            "test_err": 1.0 - acc_te,
            "test_acc": acc_te,
            "weight_norm": _weight_norm(params),
        }
        if interp_step < 0 and acc_tr >= 1.0 - 1e-9:
            interp_step = step
            err_interp = 1.0 - acc_te
            take_curv = cfg.curvature
        if take_curv:
            snap = snapshot().as_dict()
            entry["curvature"] = snap
            if step == interp_step and curv_interp is None:
                curv_interp = snap
        return entry

    history.append(log_metrics(0, take_curv=cfg.curvature))
    update_time_s = 0.0
    for step in range(1, cfg.steps + 1):
        if cfg.batch_size and cfg.batch_size < n and arm.driver in ("standard", "sharpness"):
            idx = torch.randperm(n, generator=batch_gen)[: cfg.batch_size].to(device)
            xb, yb, ob = x_tr[idx], y_tr[idx], ob_tr[idx]
        else:
            xb, yb, ob = x_tr, y_tr, ob_tr
        t_update = time.perf_counter()
        telemetry = _update(arm, opt, model, xb, yb, ob, cfg.register, params, cgen, controller)
        update_time_s += time.perf_counter() - t_update
        take_curv = cfg.curvature and ((step % every == 0) or (step == cfg.steps))
        do_log = take_curv or (step % max(cfg.log_every, 1) == 0) or (step == cfg.steps)
        if not do_log:
            continue
        history.append(log_metrics(step, take_curv=take_curv))
        if telemetry:
            history[-1].update(telemetry)
        if log:
            last = history[-1]
            extra_str = ""
            if "eos_eta" in last:
                extra_str = (
                    f"  eta={float(last['eos_eta']):.3g}  "
                    f"lam*eta={float(last['eos_lambda_eta']):.3f}"
                )
            print(
                f"[{cfg.register} {arm.name:>15s} w={cfg.width:<4d} s={cfg.seed}] "
                f"step {step:>4d}/{cfg.steps}  train_err={last['train_err']:.3f}  "
                f"test_err={last['test_err']:.3f}{extra_str}"
            )

    wall = time.perf_counter() - t0
    opt_state_bytes = _optimizer_state_bytes(opt)
    test_errs = [float(h["test_err"]) for h in history]
    curv_final = history[-1].get("curvature")
    return RunResult(
        config=_config_dict(cfg, bundle),
        n_params=n_params,
        final_train_loss=float(history[-1]["train_loss"]),
        final_train_err=float(history[-1]["train_err"]),
        final_test_err=float(history[-1]["test_err"]),
        best_test_err=min(test_errs),
        interpolation_step=interp_step,
        err_at_interpolation=err_interp,
        wall_time=wall,
        opt_state_bytes=opt_state_bytes,
        update_time_s=update_time_s,
        history=history,
        curvature_at_interpolation=curv_interp,
        curvature_final=curv_final if isinstance(curv_final, dict) else None,
    )


def _config_dict(cfg: RunConfig, bundle: DataBundle) -> dict[str, object]:
    d = asdict(cfg)
    d["data_source"] = bundle.source
    return d


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def run_filename(cfg: RunConfig) -> str:
    """Deterministic, filesystem-safe file name for one run."""
    noise = f"{cfg.label_noise:.2f}".replace(".", "p")
    return (
        f"{cfg.register}__{cfg.arm}__w{cfg.width}__d{cfg.depth}"
        f"__s{cfg.seed}__n{noise}.json"
    )


def save_run(result: RunResult, out_dir: str | Path) -> Path:
    """Write one :class:`RunResult` to ``out_dir`` as JSON; return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    noise = f"{float(cfg['label_noise']):.2f}".replace(".", "p")
    name = (
        f"{cfg['register']}__{cfg['arm']}__w{cfg['width']}__d{cfg['depth']}"
        f"__s{cfg['seed']}__n{noise}.json"
    )
    path = out / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh)
    return path


__all__ = [
    "RunConfig",
    "RunResult",
    "canonical_loss",
    "run_filename",
    "save_run",
    "train_run",
]
